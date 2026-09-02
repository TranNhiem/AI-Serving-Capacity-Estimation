"""Measured code_agent profiles from OpenCode exports, pinned against the wrong numbers.

The two failure modes this suite guards are structural: counting assistant messages
instead of step-finish parts, which under-reports requests per session by the whole
tool-calling factor, and summing tool intervals instead of unioning them, which can
push kv_residency above 1.0 and make the emitted workload unloadable by capacity_at.
"""

import pytest

from ascep.agent_profile import (
    SHAPES_VERSION,
    _merge_intervals,
    aggregate,
    parse_session,
    to_ascep_workload,
    to_replay_shapes,
)

SESSION_ID = "s1"


def _step(message_id, input_tokens, output_tokens=100, reasoning_tokens=0, cache_read=0):
    return {
        "id": f"{message_id}-sf-{input_tokens}",
        "sessionID": SESSION_ID,
        "messageID": message_id,
        "type": "step-finish",
        "reason": "stop",
        "tokens": {
            "input": input_tokens,
            "output": output_tokens,
            "reasoning": reasoning_tokens,
            "cache": {"read": cache_read, "write": 0},
        },
    }


def _tool(message_id, name, start_ms=None, end_ms=None):
    if end_ms is not None:
        status = "completed"
    elif start_ms is not None:
        status = "running"
    else:
        status = "pending"
    state = {"status": status, "input": {}}
    if start_ms is not None:
        state["time"] = {"start": start_ms}
        if end_ms is not None:
            state["time"]["end"] = end_ms
    return {
        "id": f"{message_id}-tool-{name}-{start_ms}",
        "sessionID": SESSION_ID,
        "messageID": message_id,
        "type": "tool",
        "callID": f"call-{name}-{start_ms}",
        "tool": name,
        "state": state,
    }


def _assistant(message_id, created_ms, *, steps=(), tools=(), completed_ms=None, summary=False):
    time = {"created": created_ms}
    if completed_ms is not None:
        time["completed"] = completed_ms
    info = {
        "id": message_id,
        "sessionID": SESSION_ID,
        "role": "assistant",
        "time": time,
        "modelID": "test-model",
        "providerID": "test-provider",
        "cost": 0.0,
    }
    if summary:
        info["summary"] = True
    return {"info": info, "parts": [*steps, *tools]}


def _compaction_message(message_id, created_ms):
    part = {
        "id": f"{message_id}-c",
        "sessionID": SESSION_ID,
        "messageID": message_id,
        "type": "compaction",
        "auto": True,
    }
    return _assistant(message_id, created_ms, steps=[], tools=[part], summary=False)


def _export(*messages):
    return {"info": {"id": SESSION_ID}, "messages": list(messages)}


def _realistic_export():
    """Three turns: 4 steps, 2 completed tools, hand-checkable timings in ms."""
    m1 = _assistant(
        "m1",
        1_000,
        completed_ms=11_000,
        steps=[_step("m1", 1_000, 100, 50), _step("m1", 1_600, 120, 60, cache_read=300)],
        tools=[_tool("m1", "bash", 2_000, 5_000)],
    )
    m2 = _assistant("m2", 12_000, completed_ms=20_000, steps=[_step("m2", 2_500, 150, 70)])
    m3 = _assistant(
        "m3",
        21_000,
        completed_ms=30_000,
        steps=[_step("m3", 3_000, 200, 80, cache_read=500)],
        tools=[_tool("m3", "read", 22_000, 25_000)],
    )
    return _export(m1, m2, m3)


def test_requests_are_counted_per_step_finish_part_not_per_assistant_message():
    """Two step-finish parts inside one message are two API calls, not one.

    Counting messages would report 1 request here and understate requests per
    session by the tool-calling factor -- the exact quantity this module exists
    to measure.
    """
    profile = parse_session(
        _export(_assistant("m1", 0, steps=[_step("m1", 100), _step("m1", 200)]))
    )

    assert profile.turns_per_session == 1
    assert profile.requests_per_session == 2


@pytest.mark.parametrize(
    ("intervals", "expected"),
    [
        ([(0, 10), (5, 15)], [(0, 15)]),  # overlap
        ([(0, 20), (5, 10)], [(0, 20)]),  # nested
        ([(0, 10), (10, 20)], [(0, 20)]),  # adjacent
        ([(0, 10), (20, 30)], [(0, 10), (20, 30)]),  # disjoint
    ],
    ids=["overlap", "nested", "adjacent", "disjoint"],
)
def test_tool_intervals_are_merged_as_a_union(intervals, expected):
    """Concurrent tools must not be charged twice against the session wall clock.

    A sum instead of a union inflates tool_blocked_seconds past the wall clock
    and pushes kv_residency above 1.0, an obviously impossible residency figure.
    """
    assert _merge_intervals(intervals) == expected


def test_kv_residency_is_never_below_duty_cycle():
    """capacity_at raises ValueError on kv_residency < duty_cycle.

    Guarding the relation here prevents this module from emitting a workload
    fragment that the loader itself refuses to load.
    """
    profile = parse_session(_realistic_export())

    assert profile.kv_residency >= profile.duty_cycle
    assert min(p.duty_cycle for p in [profile]) >= 0.0


def test_a_delta_spanning_a_compaction_boundary_is_excluded_from_context_growth():
    """Compaction shrinks the prompt; that shrink and the resume step are not growth.

    Averaging the post-compaction restart (2,000 -> 800) into the growth figure
    would understate per-turn growth and let ASCEP under-price the KV floor.
    """
    export = _export(
        _assistant("m1", 0, steps=[_step("m1", 2_000)]),
        _compaction_message("c1", 10_000),
        _assistant("m2", 20_000, steps=[_step("m2", 800)]),
    )
    profile = parse_session(export)

    assert profile.context_growth_tokens_per_turn == 0.0
    assert profile.compaction_events == 1
    assert profile.turns_per_session == 2


def test_negative_deltas_within_a_run_are_dropped_not_clamped():
    """A shrinking prompt inside a run is pruning, negative growth.

    Deltas 1,000 -> 1,500 (+500), 1,500 -> 1,400 (dropped), 1,400 -> 2,000 (+600)
    must mean to 550.0; clamping the -100 to 0 would still drag the figure to 366.7.
    """
    steps = [_step("m1", 1_000), _step("m1", 1_500), _step("m1", 1_400), _step("m1", 2_000)]
    profile = parse_session(_export(_assistant("m1", 0, steps=steps)))

    assert profile.context_growth_tokens_per_turn == pytest.approx(550.0)


def test_compaction_resume_tokens_is_none_when_no_compaction_occurred():
    """None means not observed; 0.0 would claim the session resumed from an empty prompt."""
    profile = parse_session(_realistic_export())

    assert profile.compaction_events == 0
    assert profile.compaction_resume_tokens is None


def test_compaction_resume_tokens_is_the_first_step_after_the_boundary():
    """The resume cost of a compaction is the prompt the next API call pays for."""
    export = _export(
        _assistant("m1", 0, steps=[_step("m1", 2_000)]),
        _compaction_message("c1", 10_000),
        _assistant("m2", 20_000, steps=[_step("m2", 700), _step("m2", 1_300)]),
    )
    profile = parse_session(export)

    assert profile.compaction_resume_tokens == pytest.approx(700.0)


def test_a_transcript_with_no_completion_timestamps_still_produces_a_profile():
    """Missing optional fields degrade to zeros, never to an exception.

    With nothing completed, generating_seconds and duty_cycle are 0.0 and the
    session span falls back to the created timestamps.
    """
    export = _export(
        _assistant("m1", 0, steps=[_step("m1", 100)]),
        _assistant("m2", 5_000, steps=[_step("m2", 300)]),
    )
    profile = parse_session(export)

    assert profile.session_seconds == 5.0
    assert profile.generating_seconds == 0.0
    assert profile.duty_cycle == 0.0
    assert profile.requests_per_session == 2


def test_pending_and_running_tools_count_per_turn_but_block_no_time():
    """An unfinished call still consumed a model decision but holds no interval.

    Dropping it from the count understates tool_calls_per_turn; inventing an end
    for it would invent blocked time that never happened.
    """
    tools = [_tool("m1", "bash"), _tool("m1", "edit", start_ms=1_000)]
    profile = parse_session(_export(_assistant("m1", 0, steps=[_step("m1", 100)], tools=tools)))

    assert profile.tool_calls_per_turn == pytest.approx(2.0)
    assert profile.tool_blocked_seconds == 0.0


@pytest.mark.parametrize(
    "export",
    [
        {"info": {"id": SESSION_ID}},
        {"info": {"id": SESSION_ID}, "messages": [{"parts": []}]},
        {"info": {"id": SESSION_ID}, "messages": [{"info": {"id": "m1"}, "parts": []}]},
    ],
    ids=["no-messages-key", "message-without-info", "info-without-role"],
)
def test_malformed_exports_raise_value_error(export):
    """Only a structurally broken export raises; robustness must not hide misuse."""
    with pytest.raises(ValueError):
        parse_session(export)


def test_unknown_message_roles_are_ignored_rather_than_raising():
    """The format may gain roles; a profiler must not be the thing that breaks."""
    unknown = {"info": {"id": "x1", "role": "system", "time": {"created": 0}}, "parts": []}
    profile = parse_session(_export(unknown, _assistant("m1", 100, steps=[_step("m1", 100)])))

    assert profile.turns_per_session == 1


def test_a_three_turn_session_matches_hand_computed_values():
    """Every aggregate of the realistic export, checkable by hand.

    Steps: inputs 1,000 / 1,600 / 2,500 / 3,000 -> mean 2,025; outputs 100 / 120 /
    150 / 200 -> mean 142.5; reasoning 50 / 60 / 70 / 80 -> mean 65; cache reads
    0 / 300 / 0 / 500 -> mean 200. Growth deltas 600, 900, 500 -> mean 666.7.
    Session span 1,000 to 30,000 ms -> 29.0 s. Tool union [2,000, 5,000] +
    [22,000, 25,000] -> 6.0 s. Windows (10s, 8s, 9s) minus in-window tool overlap
    (3s, 0s, 3s) -> generating 21.0 s. Duty 21/29 = 0.724; residency 27/29 = 0.931.
    """
    profile = parse_session(_realistic_export())

    assert profile.turns_per_session == 3
    assert profile.requests_per_session == 4
    assert profile.tool_calls_per_turn == pytest.approx(2 / 3)
    assert profile.input_tokens_per_request == pytest.approx(2_025.0)
    assert profile.output_tokens_per_request == pytest.approx(142.5)
    assert profile.reasoning_tokens_per_request == pytest.approx(65.0)
    assert profile.cache_read_tokens_per_request == pytest.approx(200.0)
    assert profile.context_growth_tokens_per_turn == pytest.approx(2_000 / 3)
    assert profile.session_seconds == pytest.approx(29.0)
    assert profile.generating_seconds == pytest.approx(21.0)
    assert profile.tool_blocked_seconds == pytest.approx(6.0)
    assert profile.duty_cycle == pytest.approx(21 / 29)
    assert profile.kv_residency == pytest.approx(27 / 29)


def test_to_ascep_workload_emits_exactly_the_declared_key_set():
    """A stray or missing key makes the fragment fail downstream validation.

    The _provenance mirror must tag every numeric field as measured (M) and must
    not invent fields the workload schema does not declare.
    """
    fragment = to_ascep_workload([parse_session(_realistic_export())])

    assert set(fragment) == {
        "archetypes",
        "requests_per_session",
        "input_tokens_per_request",
        "output_tokens_per_request",
        "reasoning_tokens_per_request",
        "context_growth_tokens_per_turn",
        "avg_session_seconds",
        "duty_cycle",
        "kv_residency",
        "agent_loop",
        "_provenance",
    }
    assert set(fragment["agent_loop"]) == {
        "turns_per_session",
        "tool_calls_per_turn",
        "compaction_resume_tokens",
        "session_max_context_tokens",
    }
    assert fragment["archetypes"] == ["code_agent"]
    assert fragment["_provenance"]["agent_loop"]["session_max_context_tokens"] is None
    supplied = to_ascep_workload(
        [parse_session(_realistic_export())], session_max_context_tokens=200_000
    )
    assert supplied["_provenance"]["agent_loop"]["session_max_context_tokens"] == "M"


def test_rounding_is_half_away_from_zero_not_bankers():
    """A mean of 2.5 requests must publish as 3 on every Python build.

    Banker's rounding would publish 2 for the same data, so two identical reports
    computed on different builds would disagree on a printed integer.
    """
    two = parse_session(_export(_assistant("m1", 0, steps=[_step("m1", 1), _step("m1", 2)])))
    three = parse_session(_export(_assistant("m1", 0, steps=[_step("m1", i) for i in range(3)])))
    fragment = to_ascep_workload([two, three])

    assert fragment["requests_per_session"] == 3


def test_aggregate_skips_zero_turn_sessions_and_keeps_resume_tokens_over_observers_only():
    """Empty sessions are not a workload and must not drag the means to zero.

    The resume figure is measured only where compaction happened; folding in the
    None sessions as zeros would fabricate a cheap-resume number nobody observed.
    """
    real = parse_session(_realistic_export())
    empty = parse_session({"info": {"id": "s2"}, "messages": []})
    agg = aggregate([real, empty])

    assert agg["skipped_sessions"] == 1
    assert agg["turns_per_session"]["n"] == 1
    assert agg["compaction_resume_tokens"] is None


def _raw_tool(message_id, name, time_obj):
    """A tool part with an arbitrary time object, including shapes the schema forbids."""
    return {
        "id": f"{message_id}-tool-{name}",
        "sessionID": SESSION_ID,
        "messageID": message_id,
        "type": "tool",
        "callID": f"call-{name}",
        "tool": name,
        "state": {"status": "completed", "input": {}, "output": "", "time": time_obj},
    }


@pytest.mark.parametrize(
    ("label", "time_obj"),
    [
        ("end with no start", {"end": 6_000}),
        ("end before start", {"start": 6_000, "end": 2_000}),
    ],
)
def test_a_tool_time_that_cannot_be_an_interval_is_not_charged_as_blocked_time(label, time_obj):
    """A malformed tool clock must not manufacture a session-long block.

    Treating an end with no start as an interval starts it at the epoch, so
    tool_blocked_seconds saturates and kv_residency clamps to a tidy-looking 1.0 --
    the report then claims the engine held KV for the entire session on the strength
    of one bad field. Wrong here is worse than absent, so the end is dropped and the
    call reads as never completed.
    """
    export = _export(
        _assistant(
            "m1",
            1_000,
            steps=[_step("m1", 500)],
            completed_ms=5_000,
        )
    )
    export["messages"][0]["parts"].append(_raw_tool("m1", "shell", time_obj))
    profile = parse_session(export)

    assert profile.tool_blocked_seconds == 0.0, label
    # The call is still counted: it consumed a model decision even if it is untimed.
    assert profile.tool_calls_per_turn == 1.0
    assert profile.kv_residency <= 1.0
    assert profile.kv_residency >= profile.duty_cycle


# --- replay shapes --------------------------------------------------------------------


def _shapes(*messages, shared_prefix_tokens=0):
    profile = parse_session(_export(*messages))
    return to_replay_shapes([profile], shared_prefix_tokens=shared_prefix_tokens)


def test_a_replayable_shape_matches_the_hand_computed_sequence():
    """The whole file pinned at once, because every field of it changes a measured floor.

    Prompt sizes drive the prefill floor, output sizes the decode time, gaps the KV
    residency and the session count that fits a window. A shape that is wrong in any one
    of them replays a workload the capture never contained, and nothing downstream can
    tell: the run completes, the numbers are plausible, and they are about something else.
    """
    out = _shapes(*_realistic_export()["messages"])

    assert out["sessions"][0]["steps"] == [
        # m1 has two steps, so its 3 s of bash time is the gap between them.
        {
            "turn_index": 0,
            "prompt_tokens": 1_000,
            "output_tokens": 150,
            "gap_s": 3.0,
            "resets_prefix": False,
        },
        # Last step of m1: the gap runs to m2 arriving, 11,000 ms -> 12,000 ms.
        {
            "turn_index": 0,
            "prompt_tokens": 1_600,
            "output_tokens": 180,
            "gap_s": 1.0,
            "resets_prefix": False,
        },
        {
            "turn_index": 1,
            "prompt_tokens": 2_500,
            "output_tokens": 220,
            "gap_s": 1.0,
            "resets_prefix": False,
        },
        # Last step of the session: nothing follows it, so there is no gap to serve.
        {
            "turn_index": 2,
            "prompt_tokens": 3_000,
            "output_tokens": 280,
            "gap_s": 0.0,
            "resets_prefix": False,
        },
    ]
    assert out["ascep_shapes_version"] == SHAPES_VERSION
    assert out["_summary"] == {
        "sessions": 1,
        "steps": 4,
        "prefix_resets": 0,
        "skipped_sessions": 0,
    }


def test_a_turns_tool_time_is_spread_across_its_gaps_not_charged_to_one():
    """Three steps and 6 s of tool time replay as two 3 s gaps, not as 6 s and 0 s.

    Both hold KV for the same total, so the residency figure is identical either way. The
    arrival pattern is not: charging it all to one gap offers the engine two back-to-back
    requests and then a stall, which is a burstier workload than the session produced and
    a harsher test of the scheduler than the capture justifies.
    """
    message = _assistant(
        "m1",
        1_000,
        completed_ms=20_000,
        steps=[_step("m1", 500), _step("m1", 900), _step("m1", 1_400)],
        tools=[_tool("m1", "bash", 2_000, 5_000), _tool("m1", "read", 6_000, 9_000)],
    )

    gaps = [s["gap_s"] for s in _shapes(message)["sessions"][0]["steps"]]

    assert gaps == [3.0, 3.0, 0.0]


def test_reasoning_tokens_are_replayed_as_generated_output():
    """They occupy decode steps and KV exactly as visible output does.

    Omitting them makes the replay generate 100 tokens where the session generated 900,
    so the run finishes each step roughly nine times too fast and reports a throughput
    ceiling the model cannot reach on the real workload.
    """
    message = _assistant(
        "m1",
        1_000,
        completed_ms=2_000,
        steps=[_step("m1", 500, output_tokens=100, reasoning_tokens=800)],
    )

    assert _shapes(message)["sessions"][0]["steps"][0]["output_tokens"] == 900


def test_the_step_after_a_compaction_starts_cold_but_the_compaction_itself_does_not():
    """The summarising request sends the conversation it is about to throw away.

    So it continues the prefix, and only the step after it starts from a summary. Marking
    the compaction step a reset instead would charge a full prefill of the largest prompt
    in the session -- the one immediately before a compaction always is -- and report a
    prefill floor tighter than the cost the session actually paid.
    """
    m1 = _assistant("m1", 1_000, completed_ms=2_000, steps=[_step("m1", 8_000)])
    boundary = _assistant(
        "m2",
        3_000,
        completed_ms=4_000,
        steps=[_step("m2", 8_400)],
        tools=[
            {
                "id": "m2-c",
                "sessionID": SESSION_ID,
                "messageID": "m2",
                "type": "compaction",
                "auto": True,
            }
        ],
    )
    m3 = _assistant("m3", 5_000, completed_ms=6_000, steps=[_step("m3", 1_200)])

    steps = _shapes(m1, boundary, m3)["sessions"][0]["steps"]

    assert [s["resets_prefix"] for s in steps] == [False, False, True]


def test_a_prompt_that_shrank_without_a_compaction_is_still_a_prefix_reset():
    """OpenCode prunes old tool output in place, which shortens the prompt mid-session.

    Replayed as growth it would be impossible to construct, and replayed as a continuation
    it would claim a cached prefix the engine does not hold -- reporting prefill work the
    server would really have to do as work it had already done.
    """
    m1 = _assistant("m1", 1_000, completed_ms=2_000, steps=[_step("m1", 30_000)])
    m2 = _assistant("m2", 3_000, completed_ms=4_000, steps=[_step("m2", 12_000)])

    steps = _shapes(m1, m2)["sessions"][0]["steps"]

    assert [s["resets_prefix"] for s in steps] == [False, True]
    assert _shapes(m1, m2)["_summary"]["prefix_resets"] == 1


def test_a_step_recording_no_prompt_at_all_is_refused_by_name():
    """A zero-token prompt is a broken export, not a very cheap request.

    Replayed, it costs the engine nothing and completes instantly, so the rung it appears
    on reports a server faster than any real one -- and the session it came from is the
    only place a reader could look to find out why.
    """
    message = _assistant("m1", 1_000, completed_ms=2_000, steps=[_step("m1", 0)])

    with pytest.raises(ValueError, match=f"session {SESSION_ID}"):
        _shapes(message)


def test_a_session_that_produced_no_steps_is_dropped_and_counted():
    """An empty session in the file would be a replay slot that issues nothing.

    A virtual user holding it would idle for the whole window while the report counts it
    as a user under load, which understates per-user demand by exactly its share.
    """
    empty = parse_session(_export(_assistant("m1", 1_000, completed_ms=2_000)))
    real = parse_session(
        _export(_assistant("m2", 1_000, completed_ms=2_000, steps=[_step("m2", 500)]))
    )

    out = to_replay_shapes([empty, real])

    assert len(out["sessions"]) == 1
    assert out["_summary"]["skipped_sessions"] == 1


def test_a_clock_that_ran_backwards_between_turns_yields_no_gap_rather_than_a_negative_one():
    """Exports carry wall-clock timestamps, and wall clocks are adjusted.

    A negative gap passed through to the driver becomes a negative sleep -- an immediate
    resubmission at best, an exception at worst -- and either way the session replays
    faster than it ran.
    """
    m1 = _assistant("m1", 1_000, completed_ms=11_000, steps=[_step("m1", 500)])
    m2 = _assistant("m2", 10_000, completed_ms=12_000, steps=[_step("m2", 900)])

    assert _shapes(m1, m2)["sessions"][0]["steps"][0]["gap_s"] == 0.0


def test_the_shared_prefix_declaration_is_carried_into_the_file():
    """The system prompt is identical across every session and is not in any step's count.

    Dropped here, the replay builds prompts short by the whole system prompt -- for a
    coding agent commonly several thousand tokens on every single request -- and the
    prefill floor comes out proportionally too kind.
    """
    message = _assistant("m1", 1_000, completed_ms=2_000, steps=[_step("m1", 500)])

    assert _shapes(message, shared_prefix_tokens=3_400)["shared_prefix_tokens"] == 3_400
