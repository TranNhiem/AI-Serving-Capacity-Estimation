"""Ladder grading: the rules that decide what a rung is allowed to claim.

Chapter 7 sections 5 to 7 are the most prescriptive part of the protocol, and almost every
rule there exists because the obvious implementation is wrong in a direction that flatters the
result. Grading is kept as pure logic over already-reduced windows so those rules can be
tested without a server -- the private harness embedded the same decisions inside its sweep
loop, where none of them were reachable by a test.
"""

import pytest

from ascep.bench.ladder import (
    LadderPolicy,
    RepetitionResult,
    RungOutcome,
    grade_ladder,
    grade_rung,
)
from ascep.bench.metrics import SloGates, WindowSummary

GATES = SloGates(ttft_p95_max_s=2.0, itl_p95_max_s=0.1, e2e_p95_max_s=30.0, error_rate_max_pct=1.0)


def _summary(*, tok_s=1000.0, completed=500, slo_pass=True, error_pct=0.0):
    return WindowSummary(
        n_issued=completed,
        n_completed=completed,
        peak_in_flight=completed,
        n_latency_samples=completed,
        excluded_error_count=0,
        excluded_invalid_count=0,
        excluded_warmup_count=0,
        error_rate_pct=error_pct,
        ttft_p50_s=0.3,
        ttft_p95_s=0.9,
        ttft_p99_s=1.4,
        itl_p50_s=0.02,
        itl_p95_s=0.05,
        itl_p99_s=0.08,
        itl_population="pooled-gaps",
        e2e_p50_s=4.0,
        e2e_p95_s=9.0,
        e2e_p99_s=12.0,
        output_tok_s=tok_s,
        requests_per_s=completed / 60.0,
        goodput_tok_s=tok_s if slo_pass else None,
        slo_pass=slo_pass,
        low_confidence=frozenset(),
        reasons={},
    )


def _reps(concurrency, n=3, **kw):
    return [
        RepetitionResult(concurrency=concurrency, repetition=i, summary=_summary(**kw))
        for i in range(n)
    ]


POLICY = LadderPolicy(
    gates=GATES, repetitions=3, throughput_collapse_ratio=0.5, cache_policy="unique-prefixes"
)


# --- section 6: three independent repetitions, and only valid ones count ---------------


def test_a_rung_with_fewer_than_three_repetitions_cannot_be_complete():
    """ "At least three independent repetitions MUST be executed at each reported operating
    point." Two good windows are not evidence of stability, they are two good windows."""
    rung = grade_rung(64, _reps(64, n=2), POLICY)
    assert rung.outcome is RungOutcome.INVALID
    assert any("three" in r or "3" in r for r in rung.reasons)


def test_an_invalid_window_does_not_count_toward_the_three():
    """ "A window marked invalid under section 7 is not one of the three; counting it as a
    repetition manufactures stability from a failed attempt."""
    reps = _reps(64, n=2)
    reps.append(
        RepetitionResult(
            concurrency=64,
            repetition=2,
            summary=_summary(),
            outcome=RungOutcome.INVALID,
            reason="telemetry stopped mid-window",
        )
    )
    rung = grade_rung(64, reps, POLICY)
    assert rung.outcome is RungOutcome.INVALID


def test_three_clean_passing_repetitions_make_a_complete_and_sustainable_rung():
    rung = grade_rung(64, _reps(64), POLICY)
    assert rung.outcome is RungOutcome.COMPLETE
    assert rung.sustainable is True


def test_a_complete_rung_whose_gates_failed_is_not_sustainable():
    """COMPLETE licenses (M) only; Sustainable needs every gate passed."""
    rung = grade_rung(256, _reps(256, slo_pass=False), POLICY)
    assert rung.outcome is RungOutcome.FAILED
    assert rung.sustainable is False


# --- section 5: disagreement is resolved conservatively, never by vote -----------------


def test_two_passes_and_one_failure_makes_the_rung_fail():
    """ "The conservative reading is normative, not best-of-N or majority vote: capacity is
    defined by the worst served user." A majority vote sells the failing windows as
    Sustainable."""
    reps = _reps(128, n=2) + [
        RepetitionResult(concurrency=128, repetition=2, summary=_summary(slo_pass=False))
    ]
    rung = grade_rung(128, reps, POLICY)
    assert rung.outcome is RungOutcome.FAILED
    assert rung.sustainable is False


def test_first_answer_does_not_win():
    """The same disagreement in the other order must grade identically."""
    a = [RepetitionResult(concurrency=128, repetition=0, summary=_summary(slo_pass=False))]
    a += _reps(128, n=2)
    b = _reps(128, n=2) + [
        RepetitionResult(concurrency=128, repetition=2, summary=_summary(slo_pass=False))
    ]
    assert grade_rung(128, a, POLICY).outcome is grade_rung(128, b, POLICY).outcome


# --- section 7: collapse, and what it is measured on -----------------------------------


def test_throughput_collapse_fails_the_rung_and_terminates_the_ladder():
    """ "Every rung above a collapsed one measures a queue, not a system."""
    rungs = {
        32: _reps(32, tok_s=1000.0),
        64: _reps(64, tok_s=1800.0),
        128: _reps(128, tok_s=700.0),  # below 0.5 x 1800 while concurrency rose
        256: _reps(256, tok_s=650.0),
    }
    result = grade_ladder(rungs, POLICY)
    assert result.rungs[128].outcome is RungOutcome.FAILED
    assert any("collapse" in r for r in result.rungs[128].reasons)
    assert result.terminated_at == 128
    assert 256 not in result.rungs, "rungs above a collapse are not graded as operating points"


def test_collapse_is_measured_on_throughput_not_goodput():
    """Deliberate, per section 7: goodput is undefined at a rung whose gates failed, so
    testing collapse on goodput would make every gate failure look like a collapse and
    terminate the ladder before the measured-tier ceiling is found.
    """
    rungs = {
        32: _reps(32, tok_s=1000.0),
        64: _reps(64, tok_s=1800.0),
        # Gates failed, so goodput is None -- but raw throughput went UP. Not a collapse.
        128: _reps(128, tok_s=2100.0, slo_pass=False),
    }
    result = grade_ladder(rungs, POLICY)
    assert result.rungs[128].outcome is RungOutcome.FAILED, "gates failed, so the rung failed"
    assert not any("collapse" in r for r in result.rungs[128].reasons)
    assert result.terminated_at is None


def test_the_collapse_ratio_is_compared_against_the_best_lower_complete_rung():
    """Not against the immediately preceding rung: a single weak rung in between would
    raise the bar for every rung above it and hide the collapse."""
    rungs = {
        32: _reps(32, tok_s=2000.0),
        64: _reps(64, tok_s=1100.0),
        128: _reps(128, tok_s=900.0),  # 0.82 of 1100, but only 0.45 of the best lower rung
    }
    result = grade_ladder(rungs, POLICY)
    assert result.rungs[128].outcome is RungOutcome.FAILED
    assert any("collapse" in r for r in result.rungs[128].reasons)


def test_a_collapse_ratio_below_one_half_is_refused():
    """ "The ratio MUST NOT be below 0.5" -- a laxer ratio lets a queueing failure keep
    climbing and every rung above it measures the queue."""
    with pytest.raises(ValueError, match="0.5"):
        LadderPolicy(gates=GATES, throughput_collapse_ratio=0.3, cache_policy="disabled")


def test_zero_completions_fails_the_rung_with_unmeasured_latency():
    """ "Zero completion yields no latency statistic, so it MUST NOT be reported as a
    slow-but-valid point; the rung is a boundary where service fell over."""
    reps = [
        RepetitionResult(
            concurrency=512,
            repetition=i,
            summary=WindowSummary(
                n_issued=400,
                n_completed=0,
                peak_in_flight=400,
                n_latency_samples=0,
                excluded_error_count=400,
                excluded_invalid_count=0,
                excluded_warmup_count=0,
                error_rate_pct=100.0,
                ttft_p50_s=None,
                ttft_p95_s=None,
                ttft_p99_s=None,
                itl_p50_s=None,
                itl_p95_s=None,
                itl_p99_s=None,
                itl_population=None,
                e2e_p50_s=None,
                e2e_p95_s=None,
                e2e_p99_s=None,
                output_tok_s=None,
                requests_per_s=0.0,
                goodput_tok_s=None,
                slo_pass=False,
                low_confidence=frozenset(),
                reasons={},
            ),
        )
        for i in range(3)
    ]
    result = grade_ladder({64: _reps(64), 512: reps}, POLICY)
    assert result.rungs[512].outcome is RungOutcome.FAILED
    assert result.terminated_at == 512


# --- section 5: censoring, the bound the experiment set rather than the engine ---------


def test_a_ladder_that_never_failed_reports_a_lower_bound_not_a_maximum():
    """ "A censored observation reported as a discovered boundary understates the system and
    hides that the experiment, not the engine, set the limit."""
    rungs = {32: _reps(32), 64: _reps(64), 128: _reps(128)}
    result = grade_ladder(rungs, POLICY)
    assert result.is_lower_bound is True
    assert result.max_sustainable_concurrency == 128
    assert result.censoring_cause is not None, "the cause must be named; the fixes differ"


def test_a_harness_limited_ladder_may_not_publish_a_sustainable_boundary():
    """ "A harness-limited run MUST NOT enter the Sustainable tier as a boundary figure at any
    concurrency above the offered load." The server was never probed at all."""
    rungs = {32: _reps(32), 64: _reps(64)}
    result = grade_ladder(rungs, POLICY, censoring_cause="harness-limited")
    assert result.is_lower_bound is True
    assert result.sustainable_publishable is False
    assert "harness-limited" in (result.censoring_cause or "")


def test_a_ladder_that_found_a_failure_is_not_censored():
    rungs = {32: _reps(32), 64: _reps(64), 128: _reps(128, slo_pass=False)}
    result = grade_ladder(rungs, POLICY)
    assert result.is_lower_bound is False
    assert result.max_sustainable_concurrency == 64


# --- section 5: monotonicity and the confirmation repetition ---------------------------


def test_non_monotone_pass_fail_is_reported_rather_than_smoothed():
    """ "A binary search over a non-monotone function returns an arbitrary point wearing a
    boundary label," so the report must say the assumption did not hold."""
    rungs = {
        32: _reps(32),
        64: _reps(64, slo_pass=False),
        128: _reps(128),  # passes above a failure: not monotone
    }
    result = grade_ladder(rungs, POLICY)
    assert result.monotone is False
    assert result.bisection_permitted is False


def test_a_monotone_ladder_permits_bisection():
    rungs = {32: _reps(32), 64: _reps(64), 128: _reps(128, slo_pass=False)}
    result = grade_ladder(rungs, POLICY)
    assert result.monotone is True
    assert result.bisection_permitted is True


def test_the_boundary_rung_needs_a_confirmation_repetition_taken_after_the_search():
    """ "The boundary is the one rung the search selected *because* it passed, so it is the
    rung most exposed to a favourable window." The fourth repetition is additional to the
    three, never instead of them.
    """
    rungs = {32: _reps(32), 64: _reps(64), 128: _reps(128, slo_pass=False)}
    unconfirmed = grade_ladder(rungs, POLICY)
    assert unconfirmed.max_sustainable_concurrency == 64
    assert unconfirmed.confirmed is False
    assert unconfirmed.sustainable_publishable is False

    confirmation = RepetitionResult(
        concurrency=64, repetition=3, summary=_summary(), post_search=True
    )
    confirmed = grade_ladder({**rungs, 64: _reps(64) + [confirmation]}, POLICY)
    assert confirmed.confirmed is True
    assert confirmed.sustainable_publishable is True


def test_a_failing_confirmation_repetition_fails_the_boundary_rung():
    """That confirmation repetition MUST pass; if it does not, the conservative rule above
    applies and the rung is recorded as failing."""
    confirmation = RepetitionResult(
        concurrency=64, repetition=3, summary=_summary(slo_pass=False), post_search=True
    )
    rungs = {32: _reps(32), 64: _reps(64) + [confirmation], 128: _reps(128, slo_pass=False)}
    result = grade_ladder(rungs, POLICY)
    assert result.rungs[64].outcome is RungOutcome.FAILED
    assert result.max_sustainable_concurrency == 32


# --- section 3: cache policy is declared, never assumed --------------------------------


def test_an_unknown_cache_policy_is_recorded_as_unmeasured_not_ignored():
    """ "If cache policy is unknown, record null with a (U) statement." Repeated identical
    prompts create cache hits unavailable in production and inflate throughput."""
    policy = LadderPolicy(gates=GATES, cache_policy="unknown")
    result = grade_ladder({32: _reps(32), 64: _reps(64)}, policy)
    assert result.cache_caveat is not None
    assert "(U)" in result.cache_caveat


def test_a_declared_cache_policy_carries_no_caveat():
    result = grade_ladder({32: _reps(32), 64: _reps(64)}, POLICY)
    assert result.cache_caveat is None


def test_an_undeclared_cache_policy_is_refused_outright():
    """Silence is the failure mode: a harness that never asked will look identical to one
    that disabled the cache, and only one of them measured production behaviour."""
    with pytest.raises(ValueError, match="cache"):
        LadderPolicy(gates=GATES, cache_policy="")


# --- section 5: what INVALID and non-monotone ladders are not allowed to claim ----------


def _invalid_reps(concurrency, n=3):
    return [
        RepetitionResult(
            concurrency=concurrency,
            repetition=i,
            summary=_summary(),
            outcome=RungOutcome.INVALID,
            reason="telemetry stopped mid-window",
        )
        for i in range(n)
    ]


def test_a_ladder_whose_top_rung_is_invalid_is_censored_not_a_boundary():
    """An unusable window is not a negative result, so no boundary was observed.

    Treating INVALID as a failure ends the ladder with `is_lower_bound` false and no
    censoring cause, which publishes the highest rung that happened to survive
    instrumentation as though the engine had drawn the line there.
    """
    result = grade_ladder({32: _reps(32), 64: _reps(64), 128: _invalid_reps(128)}, POLICY)
    assert result.rungs[128].outcome is RungOutcome.INVALID
    assert result.is_lower_bound is True
    assert result.censoring_cause is not None
    assert "instrumentation" in result.censoring_cause


def test_a_pass_above_a_failure_is_not_the_sustainable_boundary():
    """Section 5 resolves disagreement conservatively, and this is a disagreement.

    A rung that passed above one that failed is the contradiction the report has to
    explain. Taking the higher figure because it is the highest COMPLETE rung is the same
    best-of-N reasoning the per-rung rule already forbids, applied one level up.
    """
    rungs = {32: _reps(32), 64: _reps(64, slo_pass=False), 128: _reps(128)}
    result = grade_ladder(rungs, POLICY)
    assert result.monotone is False
    assert result.rungs[128].outcome is RungOutcome.COMPLETE
    assert result.max_sustainable_concurrency == 32


def test_a_confirmed_boundary_under_a_contradicted_ladder_is_still_unpublishable():
    """Confirmation shows the rung is real; it says nothing about the rungs above it."""
    confirmation = RepetitionResult(
        concurrency=32, repetition=3, summary=_summary(), post_search=True
    )
    rungs = {
        32: _reps(32) + [confirmation],
        64: _reps(64, slo_pass=False),
        128: _reps(128),
    }
    result = grade_ladder(rungs, POLICY)
    assert result.confirmed is True
    assert result.sustainable_publishable is False


def test_discarded_windows_are_reported_even_when_the_rung_passes():
    """C8 wants the bundle re-analyzable, and a discarded window is part of the evidence.

    Three passes with the discards omitted look exactly like three passes out of three.
    """
    rung = grade_rung(64, _reps(64) + _invalid_reps(64, n=2), POLICY)
    assert rung.outcome is RungOutcome.COMPLETE
    assert rung.counted_repetitions == 3
    assert rung.invalid_repetitions == 2
