"""Concurrency-ladder grading: what each rung means and what the ladder licenses.

Chapter 7 sections 5 to 7 are the most prescriptive part of the protocol, and nearly every
rule there exists because the obvious implementation is wrong in a direction that flatters
the result: majority votes that sell intermittently failing windows as Sustainable,
first-answer-wins resolution, collapse tested on goodput until every gate failure looks
like a queueing death, an exhausted ladder reported as a measured maximum, and a boundary
rung believed on the strength of the search that selected it. Grading is therefore kept as
pure logic over already-reduced :class:`~ascep.bench.metrics.WindowSummary` values, so
every one of those rules is reachable by a test without standing up a server. Nothing here
runs anything, draws randomness, touches a clock, or performs I/O.
"""

from __future__ import annotations

import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from enum import Enum

# From the sibling module, never the package: the package initialiser is where re-exports
# accumulate, and the first one that reaches back here turns a working import into a cycle.
from ascep.bench.metrics import SloGates, WindowSummary


class RungOutcome(Enum):
    """The closed abort-condition vocabulary of section 7, fixed before timing.

    The vocabulary is closed so a true system limit cannot be laundered into an
    instrumentation excuse, nor a broken harness into a capacity boundary. COMPLETE
    licenses the (M) tier only -- Sustainable additionally needs every gate passed; FAILED
    is a real negative boundary and must never be softened into INVALID; INVALID claims no
    operating point, only diagnostic evidence; ABORTED is failure evidence by cause, again
    not an operating point.
    """

    COMPLETE = "COMPLETE"
    FAILED = "FAILED"
    INVALID = "INVALID"
    ABORTED = "ABORTED"


@dataclass(frozen=True)
class LadderPolicy:
    """The pre-run declarations grading needs: the section 1 C7 table, as data.

    Every value must be fixed before the first timed request, because each one exists to
    stop a selection made after the result was visible -- gates fitted to the percentile
    found, a collapse ratio loosened until the curve looks healthy, a repetition count
    trimmed to the windows that passed, a cache policy left blank so a hit-warmed workload
    reads as production throughput.
    """

    gates: SloGates
    repetitions: int = 3
    throughput_collapse_ratio: float = 0.5
    cache_policy: str = ""

    def __post_init__(self) -> None:
        if not self.cache_policy.strip():
            # Silence is the failure mode: a harness that never asked looks identical to
            # one that disabled the cache, and only one of them measured production.
            raise ValueError(
                "cache_policy must be declared before timing (section 3): 'disabled', "
                "'cleared', 'unique-prefixes', a workload declaration, or the explicit "
                "string 'unknown' -- an empty declaration hides a warm-cache measurement"
            )
        if self.repetitions < 3:
            raise ValueError(
                "repetitions must be at least 3 (section 6): two good windows are not "
                "evidence of stability, they are two good windows"
            )
        if self.throughput_collapse_ratio < 0.5:
            # A laxer ratio lets a queueing failure keep climbing; every rung above it
            # would measure the queue, not the system.
            raise ValueError(
                "throughput_collapse_ratio must not be below 0.5 (section 7), got "
                f"{self.throughput_collapse_ratio}"
            )
        if self.throughput_collapse_ratio >= 1.0:
            # The other end is a refusal too, and it fails in the flattering direction:
            # at 1.0 any rung that merely matches the best lower COMPLETE rung is graded
            # a collapse, so the ladder terminates at the plateau every saturating system
            # produces and reports the last rung before the plateau as the boundary.
            raise ValueError(
                "throughput_collapse_ratio must be below 1.0 (section 7), got "
                f"{self.throughput_collapse_ratio}: at one or more, the flat top of a "
                "healthy throughput curve is graded as a queueing collapse"
            )


@dataclass(frozen=True)
class RepetitionResult:
    """One repetition at one rung: the reduced window plus how the window ended.

    ``outcome`` and ``reason`` carry verdicts the reduction itself cannot see -- telemetry
    that stopped mid-window is INVALID even if the surviving records reduce cleanly, and
    section 6 bars counting an invalidated window toward the three. ``post_search`` marks
    the section 5 confirmation repetition: additional to the three, never instead of them,
    and taken after the search so the stopping rule cannot have selected it.
    """

    concurrency: int
    repetition: int
    summary: WindowSummary
    outcome: RungOutcome | None = None
    reason: str = ""
    post_search: bool = False


@dataclass(frozen=True)
class RungResult:
    """The graded verdict for one operating point, plus the evidence grading consumed.

    ``sustainable`` is true only when the rung is COMPLETE: every counted repetition passed
    every gate for its full window. ``throughput_tok_s`` is the median raw output rate of
    the counted repetitions -- raw throughput, not goodput, because the section 7 collapse
    test must see a rung whose gates failed as a throughput point, not as an absence.
    """

    concurrency: int
    outcome: RungOutcome
    reasons: tuple[str, ...] = ()
    sustainable: bool = False
    throughput_tok_s: float | None = None
    counted_repetitions: int = 0
    invalid_repetitions: int = 0
    passed_confirmations: int = 0
    zero_completions: bool = False


@dataclass(frozen=True)
class LadderResult:
    """What the ladder as a whole licenses, per sections 3, 5 and 7.

    ``rungs`` holds only the rungs grading reached: grading stops at the first collapse or
    zero-completion boundary, because every rung above a collapsed one measures a queue,
    not a system. ``is_lower_bound`` marks a ladder exhausted without failure -- a censored
    observation that must be reported as "at least N" with its censoring cause named, never
    as a measured maximum. ``sustainable_publishable`` is the final gate on a Sustainable
    boundary figure: confirmed by a post-search repetition, and not harness-limited.
    """

    rungs: Mapping[int, RungResult]
    terminated_at: int | None
    monotone: bool
    bisection_permitted: bool
    is_lower_bound: bool
    censoring_cause: str | None
    max_sustainable_concurrency: int | None
    confirmed: bool
    sustainable_publishable: bool
    cache_caveat: str | None


def _repetition_failure(rep: RepetitionResult) -> tuple[str | None, bool]:
    """The conservative verdict for one counted repetition.

    Returns a human-readable failure reason -- ``None`` when the window passed -- and
    whether the window saw zero completions under offered load, the one failure section 7
    singles out as a ladder-terminating boundary rather than a slow operating point.
    """
    summary = rep.summary
    if summary.n_issued > 0 and summary.n_completed == 0:
        # Zero completions *inside the declared window* -- which is the throughput
        # statement, and is still true of a server so slow that every request landed after
        # close. Those late finishers give the window latency samples, so the reason must
        # not claim latency was unmeasurable; what it may claim is that the window
        # delivered nothing, and a rung delivering nothing is a boundary, not a slow point.
        straddled = (
            f"; {summary.n_latency_samples} request(s) completed only after window close"
            if summary.n_latency_samples
            else "; latency statistics are (U)"
        )
        return (
            f"repetition {rep.repetition}: zero completions inside the declared window "
            f"under offered load{straddled}, so the rung cannot be reported as a "
            "slow-but-valid point -- it is a boundary where service fell over (section 7)",
            True,
        )
    if rep.outcome in (RungOutcome.FAILED, RungOutcome.ABORTED):
        detail = f" ({rep.reason})" if rep.reason else ""
        return (
            f"repetition {rep.repetition}: recorded {rep.outcome.value}{detail}; a trusted "
            "negative record is a boundary, never pass-by-omission (section 7)",
            False,
        )
    if summary.slo_pass is not True:
        return (
            f"repetition {rep.repetition}: the declared gates did not hold for the full "
            "steady-state window; one failing window fails the rung, because capacity is "
            "defined by the worst served user, not by best-of-N or majority vote (section 5)",
            False,
        )
    return (None, False)


def grade_rung(
    concurrency: int,
    repetitions: Sequence[RepetitionResult],
    policy: LadderPolicy,
) -> RungResult:
    """Grade one operating point from its repetitions, per sections 5 to 7.

    The verdict is order-free and conservative: an INVALID window is discarded rather than
    counted or failed; fewer than ``policy.repetitions`` surviving pre-search windows makes
    the rung INVALID, because two good windows are not evidence of stability; and a single
    failing window -- gate, recorded failure, or post-search confirmation -- makes it
    FAILED. First-answer-wins over disagreeing probes is exactly what section 5 forbids.
    """
    counted: list[RepetitionResult] = []
    confirmations: list[RepetitionResult] = []
    n_invalid = 0
    n_stray = 0
    for rep in repetitions:
        if rep.concurrency != concurrency:
            # A window filed under a rung it was not measured at is a bookkeeping defect
            # in the harness, not evidence about either rung.
            n_stray += 1
        elif rep.outcome is RungOutcome.INVALID:
            n_invalid += 1
        elif rep.post_search:
            confirmations.append(rep)
        else:
            counted.append(rep)

    reasons: list[str] = []
    zero_completions = False
    failed = False
    for rep in counted:
        reason, zero = _repetition_failure(rep)
        zero_completions = zero_completions or zero
        if reason is not None:
            failed = True
            reasons.append(reason)

    passed_confirmations = 0
    for rep in confirmations:
        reason, zero = _repetition_failure(rep)
        zero_completions = zero_completions or zero
        if reason is None:
            passed_confirmations += 1
        else:
            failed = True
            reasons.append(
                f"confirmation {reason}; the boundary rung is the one the search selected "
                "because it passed, so only a repetition taken after the stopping rule is "
                "out of play shows the pass is a property of the system (section 5)"
            )

    if n_stray:
        outcome = RungOutcome.INVALID
        reasons.append(
            f"{n_stray} window(s) carried a concurrency other than this rung's; evidence "
            "filed under the wrong rung claims no operating point"
        )
    elif len(counted) < policy.repetitions:
        outcome = RungOutcome.INVALID
        excluded = ""
        if n_invalid:
            excluded = f"; {n_invalid} invalidated window(s) were excluded"
        reasons.append(
            f"only {len(counted)} valid repetition(s){excluded}; section 6 requires at "
            f"least {policy.repetitions} (three) independent repetitions at every "
            "reported operating point, and a window invalidated under section 7 is not "
            "one of them"
        )
    elif failed:
        outcome = RungOutcome.FAILED
    else:
        outcome = RungOutcome.COMPLETE

    # Median across the windows, not best or worst: one lucky window must not raise the
    # reference the collapse test holds against every higher rung, and one unlucky window
    # must not lower it enough to manufacture a collapse that never happened.
    throughputs = [
        rep.summary.output_tok_s for rep in counted if rep.summary.output_tok_s is not None
    ]
    throughput = float(statistics.median(throughputs)) if throughputs else None

    return RungResult(
        concurrency=concurrency,
        outcome=outcome,
        reasons=tuple(reasons),
        sustainable=outcome is RungOutcome.COMPLETE,
        throughput_tok_s=throughput,
        counted_repetitions=len(counted),
        # Carried even when the rung graded fine on its three surviving windows: a bundle
        # that shows three passes and says nothing about the two windows thrown away is
        # indistinguishable from one where nothing was thrown away.
        invalid_repetitions=n_invalid,
        passed_confirmations=passed_confirmations,
        zero_completions=zero_completions,
    )


def grade_ladder(
    rungs: Mapping[int, Sequence[RepetitionResult]],
    policy: LadderPolicy,
    censoring_cause: str | None = None,
) -> LadderResult:
    """Grade the ladder as a whole: collapse, termination, monotonicity, censoring.

    Rungs are graded low to high. Grading stops at the first throughput collapse or
    zero-completion boundary and rungs above it are left out of the report entirely, per
    section 7. A ladder graded to the top with no failure is a censored observation --
    section 5 makes it a lower bound with a named cause, because "the experiment set the
    limit" and "the engine set the limit" are different claims with different fixes. The
    boundary rung is publishable only once a post-search confirmation repetition passed,
    and a harness-limited ladder may not publish a Sustainable boundary at all.
    """
    graded: dict[int, RungResult] = {}
    terminated_at: int | None = None
    best_complete_tok_s: float | None = None
    for level in sorted(rungs):
        result = grade_rung(level, rungs[level], policy)
        reasons = list(result.reasons)
        outcome = result.outcome
        # Raw throughput, never goodput: goodput is undefined at a rung whose gates
        # failed, so a gate failure would otherwise be indistinguishable from a collapse
        # and the ladder would terminate before the measured-tier ceiling was found.
        collapsed = False
        if (
            outcome is not RungOutcome.INVALID
            and result.throughput_tok_s is not None
            and best_complete_tok_s is not None
            and result.throughput_tok_s < policy.throughput_collapse_ratio * best_complete_tok_s
        ):
            collapsed = True
            outcome = RungOutcome.FAILED
            reasons.append(
                "throughput collapse: median output tok/s "
                f"{result.throughput_tok_s:g} is below {policy.throughput_collapse_ratio:g} "
                f"times {best_complete_tok_s:g}, the best lower COMPLETE rung, while "
                "offered concurrency rose; every rung above this one measures a queue, "
                "not a system (section 7)"
            )
        graded[level] = replace(
            result,
            outcome=outcome,
            reasons=tuple(reasons),
            sustainable=outcome is RungOutcome.COMPLETE,
        )
        if collapsed or graded[level].zero_completions:
            terminated_at = level
            break
        if outcome is RungOutcome.COMPLETE and graded[level].throughput_tok_s is not None:
            current = graded[level].throughput_tok_s
            if best_complete_tok_s is None or current > best_complete_tok_s:
                best_complete_tok_s = current

    levels = sorted(graded)
    failure_seen = False
    monotone = True
    for level in levels:
        if graded[level].outcome is RungOutcome.COMPLETE:
            if failure_seen:
                # A pass above a failure contradicts the assumption bisection stands on;
                # smoothing it away would let a binary search return an arbitrary point
                # wearing a boundary label.
                monotone = False
        else:
            failure_seen = True

    # INVALID is not a failure observation. A ladder whose top rung produced no usable
    # window has not found a boundary, and counting it as one publishes the highest rung
    # that happened to survive instrumentation as if the engine had drawn the line there.
    found_failure = any(
        graded[level].outcome in (RungOutcome.FAILED, RungOutcome.ABORTED) for level in levels
    )
    is_lower_bound = bool(levels) and not found_failure

    # Declared before the run and therefore true regardless of how grading came out: a
    # harness that could not offer more load did not probe the server at either end.
    harness_limited = censoring_cause is not None and "harness-limited" in censoring_cause

    cause: str | None = None
    if is_lower_bound:
        if censoring_cause is not None:
            cause = censoring_cause
        elif any(graded[level].outcome is RungOutcome.INVALID for level in levels):
            cause = (
                "instrumentation-invalid: the highest rung reached produced no valid "
                "window, so no failure was observed and none may be inferred; the fix is "
                "repaired telemetry at that rung, not more rungs (section 5)"
            )
        else:
            cause = (
                "server-not-saturated: the ladder was exhausted without failure while the "
                "server was still healthy; the fix is more rungs (section 5)"
            )

    # Nothing above the first non-COMPLETE rung can be a sustainable boundary, even when it
    # passed: section 5 resolves disagreement conservatively, so a pass sitting above a
    # failure is the contradiction to explain, not the figure to publish.
    ceiling = next(
        (level for level in levels if graded[level].outcome is not RungOutcome.COMPLETE), None
    )
    sustainable_levels = [
        level
        for level in levels
        if graded[level].sustainable and (ceiling is None or level < ceiling)
    ]
    max_sustainable = sustainable_levels[-1] if sustainable_levels else None
    confirmed = max_sustainable is not None and graded[max_sustainable].passed_confirmations > 0
    sustainable_publishable = confirmed and not harness_limited and monotone

    cache_caveat: str | None = None
    if policy.cache_policy.strip().lower() in ("unknown", "null"):
        cache_caveat = (
            "cache policy unknown, recorded null with a (U) statement (section 3): "
            "repeated identical prompts can create cache hits unavailable in production "
            "and inflate both throughput and roofline efficiency"
        )

    return LadderResult(
        rungs=graded,
        terminated_at=terminated_at,
        monotone=monotone,
        bisection_permitted=monotone,
        is_lower_bound=is_lower_bound,
        censoring_cause=cause,
        max_sustainable_concurrency=max_sustainable,
        confirmed=confirmed,
        sustainable_publishable=sustainable_publishable,
        cache_caveat=cache_caveat,
    )
