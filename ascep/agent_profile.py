"""Measure ASCEP ``code_agent`` workloads from OpenCode session exports.

An OpenCode export already contains everything an agent-loop profile needs, but one
assistant message can wrap several API calls, so counting messages instead of
step-finish parts would under-report ``requests_per_session`` by exactly the
tool-calling factor this module exists to measure. Every aggregate it emits is a
measured (M) value suitable for an ASCEP workload declaration.
"""

from __future__ import annotations

import statistics
from collections.abc import Sequence
from dataclasses import dataclass, field
from decimal import ROUND_HALF_UP, Decimal


@dataclass(frozen=True)
class ToolCall:
    """One tool invocation from a session transcript.

    Timestamps are epoch milliseconds. ``end_ms`` is None for pending and running
    calls; those occupied the agent loop and count toward ``tool_calls_per_turn``
    but have no interval to block.
    """

    name: str
    start_ms: int
    end_ms: int | None

    @property
    def duration_s(self) -> float | None:
        """Wall-clock duration in seconds, or None when the call never ended."""
        if self.end_ms is None:
            return None
        return (self.end_ms - self.start_ms) / 1000.0


@dataclass(frozen=True)
class Step:
    """One step-finish part, i.e. one API call to the model. Token counts in tokens."""

    message_id: str
    input_tokens: int
    output_tokens: int
    reasoning_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int


@dataclass(frozen=True)
class Turn:
    """One assistant message: a possibly multi-step tool-calling turn.

    ``created_ms``/``completed_ms`` are epoch milliseconds; ``completed_ms`` is None
    for a message that never finished. A turn can hold zero steps -- it then is an
    aborted turn that still counts as a turn and contributes zero requests.
    """

    message_id: str
    created_ms: int
    completed_ms: int | None
    steps: list[Step] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)
    is_compaction: bool = False


@dataclass(frozen=True)
class SessionProfile:
    """One measured session. Times in seconds, token figures in tokens."""

    session_id: str
    model_id: str
    provider_id: str
    turns: list[Turn]
    turns_per_session: int
    requests_per_session: int
    tool_calls_per_turn: float
    input_tokens_per_request: float
    output_tokens_per_request: float
    reasoning_tokens_per_request: float
    cache_read_tokens_per_request: float
    context_growth_tokens_per_turn: float
    compaction_events: int
    compaction_resume_tokens: float | None
    session_seconds: float
    generating_seconds: float
    tool_blocked_seconds: float
    duty_cycle: float
    kv_residency: float


def _merge_intervals(intervals: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
    """Merge ``(start_ms, end_ms)`` intervals into a disjoint, sorted union.

    Summing raw intervals counts concurrent tools twice and can push
    kv_residency above 1.0, silently claiming the engine held more KV than the
    session existed for.
    """
    merged: list[tuple[int, int]] = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _round_half_away(value: float) -> int:
    """Round to an int half away from zero, so reports are reproducible across builds."""
    return int(Decimal(str(value)).quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def _round_places(value: float, places: int) -> float:
    """Round to ``places`` decimals half away from zero, for the same reason."""
    quantum = Decimal(1).scaleb(-places)
    return float(Decimal(str(value)).quantize(quantum, rounding=ROUND_HALF_UP))


_AGGREGATE_FIELDS = (
    "turns_per_session",
    "requests_per_session",
    "tool_calls_per_turn",
    "input_tokens_per_request",
    "output_tokens_per_request",
    "reasoning_tokens_per_request",
    "cache_read_tokens_per_request",
    "context_growth_tokens_per_turn",
    "compaction_events",
    "compaction_resume_tokens",
    "session_seconds",
    "generating_seconds",
    "tool_blocked_seconds",
    "duty_cycle",
    "kv_residency",
)


def parse_session(export: dict) -> SessionProfile:
    """Turn one ``opencode export`` dict into a measured SessionProfile.

    Raises ValueError only when the export is not a session export at all:
    missing ``messages``, a message without ``info``, or an ``info`` without
    ``role``. Unknown roles are skipped so a newer OpenCode cannot break the
    profiler; missing optional fields degrade to zeros, never to exceptions.
    """
    messages = export.get("messages")
    if messages is None:
        raise ValueError(
            "export is missing the 'messages' key; this is not an `opencode export` payload"
        )

    session_id = export.get("info", {}).get("id", "")
    model_id = provider_id = ""
    turns: list[Turn] = []
    all_created: list[int] = []
    end_times: list[int] = []
    # Assistant message windows [created, completed], for generating_seconds.
    message_windows: list[tuple[int, int]] = []

    for entry in messages:
        info = entry.get("info")
        if info is None:
            raise ValueError("a message in the export has no 'info' key")
        role = info.get("role")
        if role is None:
            raise ValueError("a message in the export has an 'info' with no 'role'")
        if role not in ("user", "assistant"):
            continue
        created = info.get("time", {}).get("created")
        if created is not None:
            all_created.append(created)
        if role == "user":
            continue

        parts = entry.get("parts") or []
        steps: list[Step] = []
        tool_calls: list[ToolCall] = []
        is_compaction = bool(info.get("summary"))
        for part in parts:
            part_type = part.get("type")
            if part_type == "compaction":
                is_compaction = True
            elif part_type == "step-finish":
                tokens = part.get("tokens") or {}
                cache = tokens.get("cache") or {}
                steps.append(
                    Step(
                        message_id=info["id"],
                        input_tokens=tokens.get("input", 0),
                        output_tokens=tokens.get("output", 0),
                        reasoning_tokens=tokens.get("reasoning", 0),
                        cache_read_tokens=cache.get("read", 0),
                        cache_write_tokens=cache.get("write", 0),
                    )
                )
            elif part_type == "tool":
                # Pending states carry no time object, so start degrades to 0 with
                # no end; such a call still consumed a model decision.
                time = (part.get("state") or {}).get("time") or {}
                start = time.get("start")
                end = time.get("end")
                # An end without a start, or before its start, would build an
                # interval running from the epoch that swallows the whole session:
                # tool_blocked_seconds saturates, kv_residency clamps to a tidy-looking
                # 1.0, and the resulting workload is confidently wrong. Drop the end
                # instead, which reports the call as never having completed.
                if end is not None and (start is None or end < start):
                    end = None
                if end is not None:
                    end_times.append(end)
                tool_calls.append(
                    ToolCall(
                        name=part.get("tool", ""),
                        start_ms=start if start is not None else 0,
                        end_ms=end,
                    )
                )
        completed = info.get("time", {}).get("completed")
        if completed is not None:
            end_times.append(completed)
            if created is not None:
                message_windows.append((created, completed))
        if not model_id:
            model_id = info.get("modelID", "")
            provider_id = info.get("providerID", "")
        if not session_id:
            session_id = info.get("sessionID", "")
        turns.append(
            Turn(
                message_id=info["id"],
                created_ms=created if created is not None else 0,
                completed_ms=completed,
                steps=steps,
                tool_calls=tool_calls,
                is_compaction=is_compaction,
            )
        )

    # Chronological ordering, not export ordering: an export that serialises
    # messages out of order would otherwise invert context deltas.
    turns.sort(key=lambda turn: turn.created_ms)

    non_compaction = [turn for turn in turns if not turn.is_compaction]
    turns_per_session = len(non_compaction)
    steps = [step for turn in non_compaction for step in turn.steps]
    requests_per_session = len(steps)
    tool_parts = sum(len(turn.tool_calls) for turn in non_compaction)

    def _mean(values: Sequence[int]) -> float:
        return statistics.mean(values) if values else 0.0

    # One chronological stream of steps and compaction markers, so a delta that
    # spans a boundary can be excluded and a resume step can be found.
    events: list[tuple[int, int, Step | None]] = []
    for turn in turns:
        if turn.is_compaction:
            events.append((turn.created_ms, 0, None))
        else:
            for index, step in enumerate(turn.steps):
                events.append((turn.created_ms, index + 1, step))
    events.sort(key=lambda event: (event[0], event[1] == 0))

    growth_deltas: list[int] = []
    resume_tokens: list[int] = []
    previous_input: int | None = None
    crossed_compaction = False
    for position, (_, _, step) in enumerate(events):
        if step is None:
            crossed_compaction = True
            for _, _, later in events[position + 1 :]:
                if later is not None:
                    resume_tokens.append(later.input_tokens)
                    break
            continue
        # A negative delta means the prompt shrank; averaging it in would
        # understate growth and ASCEP would under-price the KV floor.
        if previous_input is not None and not crossed_compaction:
            delta = step.input_tokens - previous_input
            if delta > 0:
                growth_deltas.append(delta)
        previous_input = step.input_tokens
        crossed_compaction = False

    compaction_events = sum(1 for turn in turns if turn.is_compaction)

    if all_created:
        first_ms = min(all_created)
        # Tool ends can postdate message completions, and a session with no
        # completions anywhere still has a span measured from created times.
        last_ms = max(end_times) if end_times else max(all_created)
        session_seconds = (last_ms - first_ms) / 1000.0
    else:
        session_seconds = 0.0

    completed_intervals = [
        (tc.start_ms, tc.end_ms)
        for turn in turns
        for tc in turn.tool_calls
        if tc.end_ms is not None
    ]
    merged = _merge_intervals(completed_intervals)
    tool_blocked_seconds = sum(end - start for start, end in merged) / 1000.0

    # A message window spans generation AND its tool executions, so subtract the
    # tool union falling inside each window rather than the message durations.
    generating_ms = 0
    for window_start, window_end in message_windows:
        overlap = sum(
            max(0, min(end, window_end) - max(start, window_start)) for start, end in merged
        )
        generating_ms += max(0, (window_end - window_start) - overlap)
    generating_seconds = max(0.0, generating_ms / 1000.0)

    if session_seconds > 0:
        duty_cycle = min(1.0, max(0.0, generating_seconds / session_seconds))
        kv_residency = min(
            1.0, max(0.0, (generating_seconds + tool_blocked_seconds) / session_seconds)
        )
    else:
        duty_cycle = 0.0
        kv_residency = 0.0
    # capacity_at raises on kv_residency < duty_cycle; floating-point noise after
    # clamping must not be allowed to fabricate that pair.
    kv_residency = max(kv_residency, duty_cycle)

    return SessionProfile(
        session_id=session_id,
        model_id=model_id,
        provider_id=provider_id,
        turns=turns,
        turns_per_session=turns_per_session,
        requests_per_session=requests_per_session,
        tool_calls_per_turn=tool_parts / turns_per_session if turns_per_session else 0.0,
        input_tokens_per_request=_mean([s.input_tokens for s in steps]),
        output_tokens_per_request=_mean([s.output_tokens for s in steps]),
        reasoning_tokens_per_request=_mean([s.reasoning_tokens for s in steps]),
        cache_read_tokens_per_request=_mean([s.cache_read_tokens for s in steps]),
        context_growth_tokens_per_turn=_mean(growth_deltas),
        compaction_events=compaction_events,
        # None, not 0.0: 0.0 would claim the session resumed from an empty prompt.
        compaction_resume_tokens=statistics.mean(resume_tokens) if resume_tokens else None,
        session_seconds=session_seconds,
        generating_seconds=generating_seconds,
        tool_blocked_seconds=tool_blocked_seconds,
        duty_cycle=duty_cycle,
        kv_residency=kv_residency,
    )


def _stats(values: list[float]) -> dict:
    if not values:
        return {"mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0, "n": 0}
    return {
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "n": len(values),
    }


def aggregate(profiles: Sequence[SessionProfile]) -> dict:
    """Aggregate several SessionProfiles into mean/median/min/max/n per field.

    Sessions with zero turns are excluded and reported under ``skipped_sessions``;
    they would otherwise drag token-per-request means toward zero without
    describing a real workload. ``compaction_resume_tokens`` aggregates only over
    sessions that observed a compaction, and is None when none did.
    """
    included = [profile for profile in profiles if profile.turns_per_session > 0]
    result: dict = {}
    for name in _AGGREGATE_FIELDS:
        values = [getattr(profile, name) for profile in included]
        values = [value for value in values if value is not None]
        if name == "compaction_resume_tokens" and not values:
            result[name] = None
        else:
            result[name] = _stats(values)
    result["skipped_sessions"] = len(profiles) - len(included)
    return result


def to_ascep_workload(
    profiles: Sequence[SessionProfile], *, session_max_context_tokens: int | None = None
) -> dict:
    """Emit an ASCEP workload declaration fragment from measured profiles.

    All values carry provenance (M). Every mean is rounded half away from zero
    rather than with banker's rounding, so a published report is reproducible
    across Python builds. Raises ValueError if rounding ever pushes kv_residency
    below duty_cycle, because capacity_at refuses to load such a pair.
    """
    agg = aggregate(profiles)

    def mean(name: str) -> float:
        entry = agg[name]
        return entry["mean"] if entry is not None else 0.0

    resume = agg["compaction_resume_tokens"]
    duty_cycle = _round_places(mean("duty_cycle"), 3)
    kv_residency = _round_places(mean("kv_residency"), 3)
    if kv_residency < duty_cycle:
        raise ValueError(
            f"rounded kv_residency {kv_residency} is below rounded duty_cycle {duty_cycle}; "
            "capacity_at would refuse to load this workload"
        )

    return {
        "archetypes": ["code_agent"],
        "requests_per_session": _round_half_away(mean("requests_per_session")),
        "input_tokens_per_request": _round_half_away(mean("input_tokens_per_request")),
        "output_tokens_per_request": _round_half_away(mean("output_tokens_per_request")),
        "reasoning_tokens_per_request": _round_half_away(mean("reasoning_tokens_per_request")),
        "context_growth_tokens_per_turn": _round_half_away(mean("context_growth_tokens_per_turn")),
        "avg_session_seconds": _round_places(mean("session_seconds"), 1),
        "duty_cycle": duty_cycle,
        "kv_residency": kv_residency,
        "agent_loop": {
            "turns_per_session": _round_half_away(mean("turns_per_session")),
            "tool_calls_per_turn": _round_places(mean("tool_calls_per_turn"), 2),
            "compaction_resume_tokens": (
                _round_half_away(resume["mean"]) if resume is not None else None
            ),
            "session_max_context_tokens": session_max_context_tokens,
        },
        "_provenance": {
            "requests_per_session": "M",
            "input_tokens_per_request": "M",
            "output_tokens_per_request": "M",
            "reasoning_tokens_per_request": "M",
            "context_growth_tokens_per_turn": "M",
            "avg_session_seconds": "M",
            "duty_cycle": "M",
            "kv_residency": "M",
            "agent_loop": {
                "turns_per_session": "M",
                "tool_calls_per_turn": "M",
                "compaction_resume_tokens": "M",
                "session_max_context_tokens": (
                    "M" if session_max_context_tokens is not None else None
                ),
            },
        },
    }


#: Schema version of the replay shapes file. It is emitted and checked rather than assumed
#: because a shapes file outlives the run that produced it: a bundle cited six months later
#: must fail loudly against a changed reader instead of replaying a shape nobody intended.
SHAPES_VERSION = 1


def _intra_turn_gaps(turn: Turn) -> float:
    """Tool-blocked seconds to charge to each gap between two steps of one turn.

    A turn's transcript records when each tool ran but not when each step finished, so the
    turn's merged tool time is spread evenly over its S-1 internal gaps. Charging it all to
    one gap instead would replay a burst of back-to-back requests followed by one long
    stall, which holds KV for the same total time but offers the engine a different arrival
    pattern than the session actually produced.
    """
    steps = len(turn.steps)
    if steps < 2:
        return 0.0
    intervals = [(c.start_ms, c.end_ms) for c in turn.tool_calls if c.end_ms is not None]
    blocked_ms = sum(end - start for start, end in _merge_intervals(intervals))
    return blocked_ms / 1000.0 / (steps - 1)


def to_replay_shapes(profiles: Sequence[SessionProfile], *, shared_prefix_tokens: int = 0) -> dict:
    """Emit the per-session, per-step shape file the closed-loop replay driver consumes.

    This is not the aggregate: :func:`to_ascep_workload` reduces many sessions to one set of
    means, and a mean cannot be replayed. A load generator needs the actual sequence -- how
    the prompt grew, where it collapsed at a compaction, and how many seconds the client sat
    on a tool between two calls -- because those are the three things that separate an agent
    loop from a stream of independent requests. Times are seconds, token figures tokens.

    ``gap_s`` is the client's idle time after a step. Where a step is not the last of its
    turn the gap is tool execution, spread evenly across the turn (see
    :func:`_intra_turn_gaps`); where it is the last, the gap runs to the next turn's arrival
    and is a human composing the next instruction. The two are told apart by ``turn_index``
    rather than by a field, and they are not the same thing for KV: this module's
    ``kv_residency`` counts tool time as resident and inter-turn time as not.

    Raises ValueError naming the session when a step records a prompt below one token: such
    a transcript cannot be replayed, and a zero-token prompt would offer the engine a rung of
    empty requests that reads as a very fast server.
    """
    sessions = []
    reset_steps = 0
    for profile in profiles:
        steps_out: list[dict] = []
        turn_index = -1
        previous_prompt: int | None = None
        # A compaction turn discards the conversation, so the next application step cannot
        # continue the prefix that preceded it.
        pending_reset = False
        for position, turn in enumerate(profile.turns):
            if not turn.is_compaction:
                turn_index += 1
            index = max(turn_index, 0)
            gap_each = _intra_turn_gaps(turn)
            # The wall clock between this turn finishing and the next one arriving is the
            # human, not the agent. It is replayed because it is real occupancy of a
            # session slot, and dropping it would compress every session into its busy
            # part and overstate how many sessions a deployment can carry.
            trailing = 0.0
            if turn.completed_ms is not None and position + 1 < len(profile.turns):
                trailing = max(0.0, (profile.turns[position + 1].created_ms - turn.completed_ms))
                trailing /= 1000.0
            for step_position, step in enumerate(turn.steps):
                if step.input_tokens < 1:
                    raise ValueError(
                        f"session {profile.session_id}: step {len(steps_out)} records a "
                        f"prompt of {step.input_tokens} tokens, which cannot be replayed"
                    )
                # A prompt that did not grow cannot have been built by appending to the one
                # before it. Compaction is the usual cause; OpenCode's tool-output pruning
                # is the other, and both invalidate the cached prefix from that point on,
                # so both are replayed as a reset rather than as impossible growth.
                #
                # The compaction turn's own request is not one of them, which is why
                # is_compaction is absent here and appears only as pending_reset below. That
                # request sends the conversation it is about to discard, plus an instruction
                # to summarise it, so it continues the prefix like any other step and the
                # step after it is the one that starts cold. Marking the compaction step
                # itself a reset would charge the replay a full prefill of the largest
                # prompt in the session -- always the one right before a compaction -- and
                # tighten the prefill floor against a cost the session never paid. If a
                # build does trim before summarising, the prompt shrinks and the rule below
                # catches it anyway.
                resets = pending_reset or (
                    previous_prompt is not None and step.input_tokens <= previous_prompt
                )
                pending_reset = False
                reset_steps += bool(resets)
                is_last = step_position == len(turn.steps) - 1
                steps_out.append(
                    {
                        "turn_index": index,
                        "prompt_tokens": step.input_tokens,
                        # Reasoning tokens are generated and cost decode time exactly as
                        # visible output does; a replay that omits them generates a
                        # fraction of the tokens the session did.
                        "output_tokens": step.output_tokens + step.reasoning_tokens,
                        "gap_s": round(trailing if is_last else gap_each, 3),
                        "resets_prefix": resets,
                    }
                )
                previous_prompt = step.input_tokens
            if turn.is_compaction:
                pending_reset = True
        if steps_out:
            sessions.append({"session_id": profile.session_id, "steps": steps_out})

    return {
        "ascep_shapes_version": SHAPES_VERSION,
        "shared_prefix_tokens": shared_prefix_tokens,
        "sessions": sessions,
        "_summary": {
            "sessions": len(sessions),
            "steps": sum(len(s["steps"]) for s in sessions),
            "prefix_resets": reset_steps,
            "skipped_sessions": len(profiles) - len(sessions),
        },
    }
