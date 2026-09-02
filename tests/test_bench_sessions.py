"""What a session replay must guarantee before its numbers mean anything.

A replayed agent session earns its place in a report only if three prefix relationships
hold exactly: consecutive steps share a prefix, a compaction really discards it, and the
declared shared prefix is all that two sessions have in common. Each failure direction is
a wrong number: an independent prompt per turn makes every request a cold prefill and
understates capacity by the prefix-cache hit rate, while a replay that is not a pure
function of (seed, session_index, step_index) lets a high concurrency rung inherit a low
rung's warm cache -- the flattering failure, throughput the deployment will never see.
"""

import asyncio
import json

import pytest

from ascep.bench.sessions import (
    ReplaySessionPlan,
    SessionShape,
    StepShape,
    load_shapes,
)


# A word-count tokenizer, matching how ascep/bench/run.py builds SyntheticCorpus: every
# space-separated word is exactly one token, so token sizes are trivially auditable.
def words(text: str) -> int:
    return len(text.split())


def _step(turn, prompt, output=10, gap=0.0, reset=False):
    return StepShape(
        turn_index=turn,
        prompt_tokens=prompt,
        output_tokens=output,
        gap_s=gap,
        resets_prefix=reset,
    )


def _growth_session(sid="ses_growth"):
    """Turn 0 spans two API calls, turn 1 grows, then a compaction resets the prompt."""
    return SessionShape(
        session_id=sid,
        steps=(
            _step(0, 8, output=12, gap=3.4),
            _step(0, 14, output=9),
            _step(1, 22, output=7, gap=12.1),
            _step(2, 9, output=11, reset=True),
            _step(2, 16, output=5, gap=0.5),
        ),
    )


def _plan(shapes, shared=0, seed=3):
    return ReplaySessionPlan(
        shapes=tuple(shapes),
        seed=seed,
        tokenizer=words,
        shared_prefix_tokens=shared,
    )


def _text(plan, session_index, step_index):
    return plan.spec(session_index=session_index, step_index=step_index, request_id="t").messages[
        0
    ]["content"]


def shared_words(*texts):
    """How many leading whole words every one of ``texts`` has in common.

    Whole words, because os.path.commonprefix compares characters and every filler word
    here starts with the same letter: on four shared tokens it returns those four words
    plus the fifth word's leading "w", and a token count taken off that string reads five.
    The prefix relationships are what this module exists to get right, so the helper the
    assertions rest on must not be the thing that is approximately right.
    """
    count = 0
    for column in zip(*(text.split() for text in texts)):
        if len(set(column)) != 1:
            break
        count += 1
    return count


def test_the_same_seed_session_and_step_render_byte_identical_text_on_two_plan_instances():
    """Determinism across processes is what makes a rung reproducible. If the same
    (seed, session_index, step_index) ever rendered different text, a reproduction
    bundle's digest would cite traffic nobody can regenerate."""
    one = _plan([_growth_session()], shared=4)
    two = _plan([_growth_session()], shared=4)
    for step_index in range(5):
        assert _text(one, 0, step_index) == _text(two, 0, step_index)


def test_a_steps_text_begins_with_the_previous_steps_text_when_the_prefix_is_kept():
    """Asserted with startswith, not length: only sharing the actual bytes makes the
    engine's prefix cache hit as it would for a real agent. An independent prompt of the
    right size would measure a cold prefill on every turn and understate capacity by the
    whole prefix-cache hit rate -- numbers that look defensible and point at hardware
    that is not insufficient."""
    plan = _plan([_growth_session()])
    for k in (1, 2, 4):
        assert _text(plan, 0, k).startswith(_text(plan, 0, k - 1))


def test_a_compaction_step_drops_the_previous_steps_text_but_keeps_the_shared_prefix():
    """A compaction really throws the prefix away: it is the one moment the cold prefill
    cost is real. Carrying the old text across the reset would hand the post-compaction
    step a cache hit production never gets."""
    plan = _plan([_growth_session()], shared=4)
    before, after = _text(plan, 0, 2), _text(plan, 0, 3)

    assert not after.startswith(before)
    # Exactly the four declared words survive the reset -- not three, which would mean the
    # system prompt is being re-sent as new text on every compaction, and not five, which
    # would mean a word of the discarded conversation came across with it.
    assert shared_words(before, after) == 4
    assert shared_words(*(_text(plan, 0, k) for k in range(5))) == 4


def test_every_steps_text_tokenizes_to_exactly_its_captured_prompt_tokens():
    """A prompt that tokenizes to 21 tokens where the capture said 22 moves the prefill
    cost the run is measuring, and the drift is invisible in the report."""
    shape = _growth_session()
    plan = _plan([shape], shared=4)
    for k, step in enumerate(shape.steps):
        assert words(_text(plan, 0, k)) == step.prompt_tokens


def test_two_sessions_share_exactly_the_declared_shared_prefix_and_nothing_more():
    """The shared prefix is the one cache line all sessions legitimately share. Anything
    beyond it would be cross-session cache hits between agents that production keeps
    separate, flattering every rung after the first."""
    a = SessionShape(session_id="ses_a", steps=(_step(0, 8), _step(1, 14)))
    b = SessionShape(session_id="ses_b", steps=(_step(0, 7), _step(1, 15)))
    plan = _plan([a, b], shared=4)

    # Across sessions, not across steps: two steps of one session are supposed to share
    # more than the prefix -- the later one continues the earlier -- so comparing those
    # would pass on a plan that leaked every session's text into every other one.
    assert shared_words(_text(plan, 0, 0), _text(plan, 1, 0)) == 4
    assert shared_words(_text(plan, 0, 1), _text(plan, 1, 1)) == 4
    # And a third session, which wraps round to shape a again: same shape, different text.
    assert shared_words(_text(plan, 0, 0), _text(plan, 2, 0)) == 4
    assert _text(plan, 0, 0) != _text(plan, 2, 0)


def test_session_assignment_is_round_robin_so_the_mix_depends_only_on_the_draw_count():
    """A random draw would give a 128-concurrency rung a different mix of session lengths
    than the same rung at 8, and the two rungs would stop being comparable. Round-robin
    makes the mix over M sessions a function of M alone, not of any interleaving."""
    shapes = [SessionShape(session_id=f"ses_{i}", steps=(_step(0, 8),)) for i in range(3)]
    plan = _plan(shapes)
    ids = [plan.shape(i).session_id for i in range(7)]
    assert ids == ["ses_0", "ses_1", "ses_2", "ses_0", "ses_1", "ses_2", "ses_0"]
    assert plan.size == 3
    assert "round-robin" in plan.sampler_rule


def test_a_shrinking_prompt_without_resets_prefix_raises_naming_the_session():
    """Appending cannot shrink, and silently truncating the prefix would destroy the
    cache hit the whole module exists to reproduce. The capture recorded a compaction it
    did not mark; the error has to say which session so the file can be fixed at the
    step it went wrong."""
    bad = SessionShape(session_id="ses_shrink", steps=(_step(0, 10), _step(1, 6)))
    plan = _plan([bad])
    with pytest.raises(ValueError, match="ses_shrink"):
        plan.spec(session_index=0, step_index=1, request_id="t")
    with pytest.raises(ValueError, match="step 1"):
        plan.spec(session_index=0, step_index=1, request_id="t")


def test_a_session_with_no_steps_is_refused():
    """An empty session replays as no traffic at all while still occupying a slot in the
    round-robin -- a rung quietly shorter than the ladder claims."""
    with pytest.raises(ValueError, match="ses_empty"):
        SessionShape(session_id="ses_empty", steps=())


def test_a_decreasing_turn_index_is_refused():
    """turn_index must be non-decreasing because steps are chronological; a decrease
    means the file was edited out of order and the prefix relationships no longer mean
    what the timestamps meant."""
    with pytest.raises(ValueError, match="ses_dec"):
        SessionShape(session_id="ses_dec", steps=(_step(1, 8), _step(0, 12)))


def test_a_negative_gap_is_refused():
    """A gap is client-side idle time in seconds; a negative one would have the driver
    sleep a backwards duration and silently compress the session's wall clock."""
    with pytest.raises(ValueError, match="ses_gap"):
        SessionShape(session_id="ses_gap", steps=(_step(0, 8, gap=-0.1),))


def test_a_prompt_below_one_token_is_refused():
    """A zero-token prompt renders as an empty request, and a rung of empty requests
    looks like a very fast server."""
    with pytest.raises(ValueError, match="ses_zero"):
        SessionShape(session_id="ses_zero", steps=(_step(0, 0),))


@pytest.mark.parametrize(
    "content",
    [
        "{not json at all",
        json.dumps({"sessions": [{"session_id": "s", "steps": []}]}),
        json.dumps({"ascep_shapes_version": 2, "sessions": [{"session_id": "s", "steps": []}]}),
        json.dumps({"ascep_shapes_version": 1, "sessions": []}),
        json.dumps(
            {
                "ascep_shapes_version": 1,
                "sessions": [
                    {
                        "session_id": "s",
                        "steps": [
                            {"turn_index": 0, "prompt_tokens": 8, "output_tokens": 4, "gap_s": 0.0}
                        ],
                    }
                ],
            }
        ),
    ],
    ids=["unreadable-json", "missing-version", "wrong-version", "empty-sessions", "missing-key"],
)
def test_load_shapes_refuses_each_bad_input_naming_the_path(tmp_path, content):
    """Every schema failure must stop the run at load time, naming the file, because the
    alternative is discovering it eight hours into a ladder. The path is asserted, not
    just the refusal: a bare error sends the operator to grep the wrong run directory."""
    path = tmp_path / "shapes.json"
    path.write_text(content)
    with pytest.raises(ValueError) as exc:
        load_shapes(path)
    assert str(path) in str(exc.value)


def test_load_shapes_returns_the_shapes_and_the_declared_shared_prefix(tmp_path):
    """shared_prefix_tokens is declared, not captured -- a transcript records only total
    prompt sizes -- so it must survive the round trip exactly, with 0 as the default."""
    payload = {
        "ascep_shapes_version": 1,
        "shared_prefix_tokens": 11,
        "sessions": [
            {
                "session_id": "ses_1",
                "steps": [
                    {
                        "turn_index": 0,
                        "prompt_tokens": 12,
                        "output_tokens": 4,
                        "gap_s": 1.5,
                        "resets_prefix": False,
                    },
                    {
                        "turn_index": 1,
                        "prompt_tokens": 20,
                        "output_tokens": 6,
                        "gap_s": 0.0,
                        "resets_prefix": False,
                    },
                ],
            }
        ],
    }
    path = tmp_path / "shapes.json"
    path.write_text(json.dumps(payload))
    shapes, shared = load_shapes(path)
    assert shared == 11
    assert shapes[0].session_id == "ses_1"
    assert shapes[0].steps[1].prompt_tokens == 20

    payload["shared_prefix_tokens"] = None
    del payload["shared_prefix_tokens"]
    path.write_text(json.dumps(payload))
    _, shared = load_shapes(path)
    assert shared == 0


def test_spec_uses_each_steps_own_output_tokens_not_a_run_wide_value():
    """max_tokens comes from the captured step: a replay whose every turn generates the
    same number of tokens is not the workload that was captured, and a run-wide value
    would misprice decode cost on every unequal turn."""
    plan = _plan([_growth_session()])
    short = plan.spec(session_index=0, step_index=4, request_id="a")
    long = plan.spec(session_index=0, step_index=0, request_id="b")
    assert short.max_tokens == 5
    assert long.max_tokens == 12
    assert short.max_tokens != long.max_tokens


def test_spec_forces_the_captured_output_length_and_sends_one_user_message():
    """ignore_eos stops the model ending the turn early, so the output side stays the
    captured shape rather than whatever the served model feels like. One long user
    message is deliberate: what the engine prices is prompt tokens and their prefix
    relationship, and a fake conversation array would pretend the replayed content is a
    real dialogue."""
    plan = _plan([_growth_session()])
    spec = plan.spec(session_index=0, step_index=1, request_id="t")
    assert spec.extra == {"ignore_eos": True}
    assert [m["role"] for m in spec.messages] == ["user"]


def test_turns_counts_distinct_indices_while_requests_counts_api_calls():
    """A tool-calling turn issues several API calls that share one index. Reporting
    requests as turns would describe five API calls as five agent turns -- the error
    that makes an agent look five times as productive per session as it was."""
    shape = _growth_session()
    assert shape.requests == 5
    assert shape.turns == 3
    assert shape.wall_clock_s == pytest.approx(3.4 + 12.1 + 0.5)


def test_the_digest_changes_with_the_shapes_and_stays_stable_without_change():
    """The digest is what a reproduction bundle cites for 'same traffic'. Stable when
    nothing changed, and sensitive to the captured shape: a digest that ignored a shape
    edit would assert two runs comparable when they replayed different workloads."""
    base = _plan([_growth_session()])
    same = _plan([_growth_session()])
    assert base.digest == same.digest
    edited = SessionShape(
        session_id="ses_growth", steps=tuple([*_growth_session().steps, _step(3, 30)])
    )
    assert _plan([edited]).digest != base.digest
    assert _plan([_growth_session()], shared=4).digest != base.digest
    assert _plan([_growth_session()], seed=4).digest != base.digest


def test_rendering_a_step_gives_the_same_text_whatever_order_the_steps_were_asked_for():
    """The plan memoises the last step of each session so a walk is linear, not quadratic.

    A memo over a pure function is invisible; a memo that becomes state is not. If the
    resumed path and the rebuilt path ever diverged, a rung would replay different bytes
    depending on how its workers happened to interleave, and the run would be measuring an
    ordering artefact -- reproducible from the config only by accident.
    """
    shape = _growth_session()
    forward = _plan([shape], shared=4)
    backward = _plan([shape], shared=4)

    in_order = [_text(forward, 0, k) for k in range(5)]
    reversed_order = [_text(backward, 0, k) for k in (4, 3, 2, 1, 0)][::-1]

    assert in_order == reversed_order


def test_a_long_session_renders_in_linear_time_rather_than_quadratic_time():
    """The driver calls spec() inline on the event loop it is timing with.

    Rebuilding every earlier step on each call cost half a second of blocking CPU at
    120,000 prompt tokens. Every other virtual user waits that out, so it lands in their
    latency samples: the harness measures itself, reports a server that is slow under
    concurrency, and the effect grows with exactly the context length agent workloads have.
    """
    sizes = [4_000 + i * 4_000 for i in range(12)]
    shape = SessionShape(
        session_id="ses_long",
        steps=tuple(_step(i // 2, size, output=200) for i, size in enumerate(sizes)),
    )
    plan = _plan([shape], shared=1_000)

    # Cost per step must stay flat in the session's own length, so the last step may not
    # cost meaningfully more than the work its own size implies. A quadratic rebuild made
    # the last step of this session roughly forty times the first.
    for k, size in enumerate(sizes):
        assert words(_text(plan, 0, k)) == size
    assert len(plan._recent) == 1, "one session in flight should hold one memo entry"


def test_a_file_written_by_ascep_agent_profile_loads_and_drives_the_driver(tmp_path):
    """The two halves were built against a schema, not against each other.

    This is the seam where a replay silently becomes something else: the exporter writes a
    summary block the loader never asked for, the loader defaults a field the exporter
    always emits, and the mismatch surfaces as a ladder that ran but measured independent
    requests. Everything here is the real path -- real exporter, real file, real loader,
    real driver -- because a fixture written by hand would agree with whichever side wrote it.
    """
    from ascep.agent_profile import parse_session, to_replay_shapes
    from ascep.bench.driver import WindowPolicy, no_reset, run_window
    from ascep.bench.records import RequestRecord

    def part(mid, inp, out):
        return {
            "id": f"{mid}-{inp}",
            "sessionID": "s1",
            "messageID": mid,
            "type": "step-finish",
            "reason": "stop",
            "tokens": {
                "input": inp,
                "output": out,
                "reasoning": 0,
                "cache": {"read": 0, "write": 0},
            },
        }

    def message(mid, created, completed, parts):
        return {
            "info": {
                "id": mid,
                "sessionID": "s1",
                "role": "assistant",
                "time": {"created": created, "completed": completed},
                "modelID": "m",
                "providerID": "p",
                "cost": 0.0,
            },
            "parts": parts,
        }

    # Millisecond gaps, so the whole session replays many times inside a short window. A
    # capture's real gaps are seconds; a test that used them would measure the sleep.
    export = {
        "info": {"id": "s1"},
        "messages": [
            message(
                "m1",
                1_000,
                1_020,
                [
                    part("m1", 40, 12),
                    part("m1", 61, 9),
                    {
                        "id": "m1-t",
                        "sessionID": "s1",
                        "messageID": "m1",
                        "type": "tool",
                        "callID": "c1",
                        "tool": "bash",
                        "state": {
                            "status": "completed",
                            "input": {},
                            "time": {"start": 1_005, "end": 1_015},
                        },
                    },
                ],
            ),
            message("m2", 1_030, 1_040, [part("m2", 95, 7)]),
        ],
    }

    path = tmp_path / "shapes.json"
    path.write_text(json.dumps(to_replay_shapes([parse_session(export)], shared_prefix_tokens=12)))

    shapes, shared = load_shapes(path)
    assert shared == 12, "the exporter's declared prefix must survive the round trip"
    plan = ReplaySessionPlan(shapes=shapes, seed=5, tokenizer=words, shared_prefix_tokens=shared)

    class _Adapter:
        name = "fake"

        def __init__(self):
            self.seen = []

        async def issue(self, spec, *, clock, sink=None):
            self.seen.append((words(spec.messages[0]["content"]), spec.max_tokens))
            record = RequestRecord(request_id=spec.request_id, issued_ts=clock())
            record.first_token_ts = record.end_ts = clock()
            record.output_tokens = 1
            return record

    adapter = _Adapter()
    run = asyncio.run(
        run_window(
            adapter,
            policy=WindowPolicy(
                concurrency=2,
                window_s=0.25,
                drain_deadline_s=0.2,
                think_time_s=0.0,
                warmup_requests=2,
                repetition=0,
            ),
            reset=no_reset,
            session_plan=plan,
        )
    )

    assert run.records, "the replay issued nothing"
    # Prompt and output sizes come from the capture, not from a single declared figure.
    assert set(adapter.seen) == {(40, 12), (61, 9), (95, 7)}
    for record in run.records:
        assert record.session_id is not None
        assert record.turn_index in (0, 1)


def test_the_writer_and_the_reader_agree_on_the_shapes_schema_version():
    """Two modules declare it independently, and only one of them is ever bumped.

    Raise the exporter's and the loader refuses every file the exporter now writes -- loud,
    and caught immediately. Raise the loader's and it refuses every file already captured,
    which is the same failure pointed at the evidence rather than at the tool.
    """
    from ascep.agent_profile import SHAPES_VERSION as writer_version
    from ascep.bench.sessions import SHAPES_VERSION as reader_version

    assert writer_version == reader_version
