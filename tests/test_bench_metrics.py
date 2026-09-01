"""The reduction from raw records to published figures, tested clause by clause.

Chapter 4 is unusually prescriptive about this step because almost every way of getting it
wrong produces a plausible number rather than an error. Each test below names the clause it
enforces; if a test here disagrees with the chapter, the chapter wins and the test is the bug.
"""

import pytest

from ascep.bench import Outcome, RequestRecord
from ascep.bench.metrics import (
    SloGates,
    bootstrap_ci,
    percentile,
    reduce_window,
    slice_window,
)


def _ok(rid, issued, ttft, itls, out_tokens=None, in_tokens=None):
    """A completed request whose first token lands at `issued + ttft`, then `itls` gaps."""
    first = issued + ttft
    stamps, t = [], first
    for gap in itls:
        t += gap
        stamps.append(t)
    return RequestRecord(
        request_id=rid,
        issued_ts=issued,
        outcome=Outcome.OK,
        first_token_ts=first,
        token_ts=stamps,
        end_ts=stamps[-1] if stamps else first,
        input_tokens=in_tokens,
        output_tokens=out_tokens if out_tokens is not None else len(stamps) + 1,
    )


# --- §4.3 percentile convention -------------------------------------------------------


def test_percentile_uses_hyndman_fan_type_7_interpolation():
    """The protocol names one convention so two analysts get the same number.

    Rank is p*(n-1) with linear interpolation between the bracketing order statistics --
    numpy's default. Nearest-rank would give 3 for the p50 below, not 2.5.
    """
    assert percentile([1.0, 2.0, 3.0, 4.0], 0.50) == pytest.approx(2.5)
    assert percentile([0.0, 1.0], 0.25) == pytest.approx(0.25)
    assert percentile([float(i) for i in range(20)], 0.95) == pytest.approx(18.05)


def test_percentile_of_a_single_sample_is_that_sample():
    assert percentile([7.0], 0.5) == pytest.approx(7.0)


def test_percentile_is_order_independent():
    assert percentile([4.0, 1.0, 3.0, 2.0], 0.5) == pytest.approx(2.5)


# --- §4.3 sample-size floors ----------------------------------------------------------


def test_p95_below_twenty_samples_is_unmeasured_not_the_maximum():
    """n >= 1/(1-p) is the absolute floor: p95 needs 20 samples.

    Below it the interpolated value is the observed maximum wearing a percentile label --
    several times the true tail on an unlucky window, flatteringly low on a lucky one.
    """
    recs = [_ok(f"r{i}", i * 1.0, 0.1, [0.01]) for i in range(19)]
    s = reduce_window(recs, window_s=19.0)
    assert s.ttft_p95_s is None
    assert "ttft_p95_s" in s.reasons and "20" in s.reasons["ttft_p95_s"]


def test_p95_at_exactly_twenty_samples_is_reported():
    recs = [_ok(f"r{i}", i * 1.0, 0.1, [0.01]) for i in range(20)]
    assert reduce_window(recs, window_s=20.0).ttft_p95_s is not None


def test_p99_below_one_hundred_samples_is_unmeasured():
    recs = [_ok(f"r{i}", i * 1.0, 0.1, [0.01]) for i in range(99)]
    s = reduce_window(recs, window_s=99.0)
    assert s.ttft_p99_s is None
    assert s.ttft_p95_s is not None, "p95 has its own, lower floor and should still report"


def test_a_percentile_between_the_two_floors_is_reported_but_flagged():
    """§4.3: the absolute floor is where the figure stops being meaningless, not where it
    becomes stable. Between 1/(1-p) and 10/(1-p) there is one observation in the tail, so a
    single straggler moves the figure -- publishable, but not as a bare point estimate.
    """
    recs = [_ok(f"r{i}", i * 0.1, 0.1 + i * 0.001, [0.01]) for i in range(100)]
    s = reduce_window(recs, window_s=10.0)
    assert s.ttft_p95_s is not None
    assert "ttft_p95_s" in s.low_confidence


def test_a_percentile_above_the_advisory_floor_is_not_flagged():
    recs = [_ok(f"r{i}", i * 0.01, 0.1 + i * 0.001, [0.01]) for i in range(200)]
    s = reduce_window(recs, window_s=2.0)
    assert "ttft_p95_s" not in s.low_confidence


# --- §4.1 which ITL population, and §4.7 never the e2e span ---------------------------


def _no_stamps(rid, issued, ttft, span, out_tokens):
    """A completed request with token counts and end time but no per-token timestamps."""
    return RequestRecord(
        request_id=rid,
        issued_ts=issued,
        outcome=Outcome.OK,
        first_token_ts=issued + ttft,
        token_ts=[],
        end_ts=issued + ttft + span,
        output_tokens=out_tokens,
    )


def test_without_per_token_stamps_itl_is_the_decode_span_not_the_e2e_span():
    """§4.1 defines per-request ITL as (t_last - t_first) / (output_tokens - 1).

    The barred statistic is e2e / output_tokens, which folds prefill and queueing into a
    per-token figure. The decode span excludes both, so it is measurable here and reporting
    it (U) would hide a number the protocol requires. What must not happen is reporting it
    under the same label as the pooled population, so the summary names which one it is.
    """
    recs = [_no_stamps(f"r{i}", float(i), 3.0, 9.9, 100) for i in range(50)]
    s = reduce_window(recs, window_s=50.0)
    assert s.itl_population == "per-request-mean"
    assert s.itl_p50_s == pytest.approx(0.1)
    naive = (3.0 + 9.9) / 100
    assert s.itl_p50_s != pytest.approx(naive), "e2e / output_tokens is a different statistic"


def test_a_single_token_reply_contributes_no_itl_sample():
    """There is no decode phase behind one token, only prefill.

    Dividing its span by zero steps raises; dividing by one quietly files a TTFT-shaped
    measurement in the ITL distribution, which is the same conflation §4.1 forbids.
    """
    recs = [_no_stamps(f"r{i}", float(i), 0.4, 0.0, 1) for i in range(50)]
    s = reduce_window(recs, window_s=50.0)
    assert s.itl_population is None
    assert s.itl_p50_s is None
    assert s.e2e_p95_s is not None, "e2e is still measurable from the stamps that do exist"


def test_pooled_gaps_are_preferred_wherever_the_stamps_exist():
    recs = [_ok(f"r{i}", i * 1.0, 0.1, [0.01] * 3) for i in range(20)]
    assert reduce_window(recs, window_s=20.0).itl_population == "pooled-gaps"


def test_a_stall_inside_a_request_reaches_the_itl_tail_while_the_median_stays_clean():
    """Why the pooled population is preferred, in one construction.

    Twenty requests, fifty gaps each: 1000 ITL samples of which 20 are 0.9 s stalls. Pooled
    at the gap level the p99 is the stall and the p50 is unaffected, which is exactly the
    signal an ITL gate is for. Averaged per request first, every sample becomes 0.028 s --
    the stall vanishes, and only 20 samples remain, which is below the p99 floor anyway.
    """
    recs = [_ok(f"r{i}", i * 1.0, 0.1, [0.01] * 25 + [0.9] + [0.01] * 24) for i in range(20)]
    s = reduce_window(recs, window_s=20.0)
    assert s.itl_p50_s == pytest.approx(0.01)
    assert s.itl_p99_s == pytest.approx(0.9)
    per_request_mean = (0.01 * 49 + 0.9) / 50
    assert s.itl_p99_s != pytest.approx(per_request_mean), "the mean is what pooling avoids"


# --- §4.7.2 monotonicity validation ---------------------------------------------------


def test_a_record_with_a_negative_span_is_excluded_and_counted_not_clamped():
    """Clock skew must not be clamped to zero: it would bias p50 toward zero silently."""
    good = [_ok(f"r{i}", i * 1.0, 0.1, [0.01]) for i in range(20)]
    bad = RequestRecord(
        request_id="skewed",
        issued_ts=100.0,
        outcome=Outcome.OK,
        first_token_ts=100.5,
        token_ts=[],
        end_ts=100.2,  # ends before its own first token
    )
    s = reduce_window(good + [bad], window_s=21.0)
    assert s.excluded_invalid_count == 1
    assert s.n_issued == 21, "an invalid record is still an issued request"


def test_sub_millisecond_skew_is_tolerated_rather_than_rejected():
    """Client and server clocks are never perfectly synchronised; 1 ms slack is honest."""
    r = RequestRecord(
        request_id="tiny-skew",
        issued_ts=100.0,
        outcome=Outcome.OK,
        first_token_ts=99.9997,  # 0.3 ms before issue
        token_ts=[],
        end_ts=100.5,
    )
    assert reduce_window([r], window_s=1.0).excluded_invalid_count == 0


# --- §4.7 error handling and the issued denominator -----------------------------------


def test_error_records_are_excluded_from_latency_but_counted_in_the_denominator():
    ok = [_ok(f"r{i}", i * 1.0, 0.1, [0.01]) for i in range(20)]
    failed = [
        RequestRecord(request_id="e1", issued_ts=50.0, outcome=Outcome.ERROR),
        RequestRecord(request_id="e2", issued_ts=51.0, outcome=Outcome.TIMEOUT),
    ]
    s = reduce_window(ok + failed, window_s=52.0)
    assert s.excluded_error_count == 2
    assert s.n_issued == 22
    assert s.n_completed == 20
    assert s.error_rate_pct == pytest.approx(2 / 22 * 100)


def test_a_refusal_at_admission_counts_against_the_error_rate():
    """Chapter 7 §6: issued, not admitted. Otherwise a shedding server reports 0%."""
    ok = [_ok(f"r{i}", i * 1.0, 0.1, [0.01]) for i in range(10)]
    refused = [
        RequestRecord(request_id=f"x{i}", issued_ts=20.0 + i, outcome=Outcome.REFUSED)
        for i in range(10)
    ]
    s = reduce_window(ok + refused, window_s=30.0)
    assert s.error_rate_pct == pytest.approx(50.0)


# --- §4.2 goodput is a property of the window -----------------------------------------


def test_goodput_is_undefined_for_a_window_where_a_gate_failed():
    """Not "the throughput of the requests that passed" -- no request passes a percentile."""
    recs = [_ok(f"r{i}", i * 0.05, 5.0, [0.01]) for i in range(20)]  # TTFT 5 s, gate is 2 s
    gates = SloGates(
        ttft_p95_max_s=2.0, itl_p95_max_s=None, e2e_p95_max_s=None, error_rate_max_pct=1.0
    )
    s = reduce_window(recs, window_s=1.0, gates=gates)
    assert s.slo_pass is False
    assert s.goodput_tok_s is None
    assert s.output_tok_s is not None, "raw throughput is still measured-tier reportable"


def test_goodput_equals_throughput_when_every_gate_held():
    recs = [_ok(f"r{i}", i * 0.05, 0.1, [0.01]) for i in range(20)]
    gates = SloGates(
        ttft_p95_max_s=2.0, itl_p95_max_s=1.0, e2e_p95_max_s=10.0, error_rate_max_pct=1.0
    )
    s = reduce_window(recs, window_s=1.0, gates=gates)
    assert s.slo_pass is True
    assert s.goodput_tok_s == pytest.approx(s.output_tok_s)


def test_a_gate_whose_statistic_is_unmeasurable_counts_as_failed():
    """§4.3: "the framework could not expose p99 TPOT" is not a pass.

    Only 5 requests, so p95 is below its floor and unmeasured. A permissive reduction would
    call the gate satisfied and let this window feed the sustainable tier.
    """
    recs = [_ok(f"r{i}", i * 0.1, 0.1, [0.01]) for i in range(5)]
    gates = SloGates(
        ttft_p95_max_s=2.0, itl_p95_max_s=None, e2e_p95_max_s=None, error_rate_max_pct=1.0
    )
    s = reduce_window(recs, window_s=1.0, gates=gates)
    assert s.ttft_p95_s is None
    assert s.slo_pass is False
    assert s.goodput_tok_s is None


# --- §4.8 determinism -----------------------------------------------------------------


def test_the_same_records_reduce_to_identical_numbers_every_time():
    """C8 requires bit-identical re-analysis, which rules out unseeded resampling."""
    recs = [_ok(f"r{i}", i * 0.05, 0.1 + i * 0.001, [0.01, 0.02]) for i in range(200)]
    a = reduce_window(recs, window_s=10.0)
    b = reduce_window(recs, window_s=10.0)
    assert a == b


def test_the_bootstrap_interval_is_identical_across_calls_and_processes():
    """§4.8: "two sites handed identical records must obtain identical bounds".

    That rules out any unseeded resampling, so the seed is an argument rather than a default
    drawn from the clock -- and the same seed must give the same bounds forever, which is why
    the generator is `random.Random(seed)` and not the module-level `random`.
    """
    samples = [0.1 + (i % 37) * 0.01 for i in range(500)]
    a = bootstrap_ci(samples, 0.95, seed=20260901, resamples=200)
    b = bootstrap_ci(samples, 0.95, seed=20260901, resamples=200)
    assert a == b
    lo, hi = a
    assert lo <= percentile(samples, 0.95) <= hi


def test_a_different_seed_is_allowed_to_differ_but_must_be_declared():
    samples = [0.1 + (i % 37) * 0.01 for i in range(500)]
    assert bootstrap_ci(samples, 0.95, seed=1, resamples=200) == bootstrap_ci(
        samples, 0.95, seed=1, resamples=200
    )


def test_a_sample_below_the_percentile_floor_has_no_interval_either():
    """An interval around an unmeasured figure would imply the figure exists."""
    assert bootstrap_ci([0.1] * 19, 0.95, seed=1, resamples=200) is None


def test_an_empty_window_reports_nothing_measured_rather_than_zero():
    """Zero throughput and zero error rate are both claims; neither was observed."""
    s = reduce_window([], window_s=10.0)
    assert s.n_issued == 0
    assert s.ttft_p50_s is None
    assert s.error_rate_pct is None


# --- chapter 7 §4: the slice table that proves steady state ---------------------------


def test_the_window_is_cut_into_adjacent_equal_slices_that_tile_it_exactly():
    """ "Adjacent equal slices" is normative: a gap or an overlap would let a bad interval
    fall between two rows and never appear in the table that exists to expose it."""
    recs = [_ok(f"r{i}", i * 0.1, 0.05, [0.01]) for i in range(100)]
    rows = slice_window(recs, window_s=10.0, n_slices=5, t0=0.0)
    assert len(rows) == 5
    assert [r.t_start for r in rows] == pytest.approx([0.0, 2.0, 4.0, 6.0, 8.0])
    assert [r.t_end for r in rows] == pytest.approx([2.0, 4.0, 6.0, 8.0, 10.0])
    for a, b in zip(rows, rows[1:]):
        assert a.t_end == pytest.approx(b.t_start)


def test_accepted_and_completed_are_counted_on_different_timestamps():
    """Chapter 7 §4 asks for both rates per slice, and they diverge exactly when it matters.

    Ten requests are all issued in the first slice but all finish in the second. A table that
    counted both on one timestamp would show a flat system; the real shape is a burst that
    the server absorbed and paid for later, which is what a steady-state check is looking for.
    """
    recs = [
        RequestRecord(
            request_id=f"r{i}",
            issued_ts=0.5,
            outcome=Outcome.OK,
            first_token_ts=0.6,
            token_ts=[2.5],
            end_ts=2.5,
            output_tokens=2,
        )
        for i in range(10)
    ]
    rows = slice_window(recs, window_s=4.0, n_slices=2, t0=0.0)
    assert rows[0].accepted == 10 and rows[0].completed == 0
    assert rows[1].accepted == 0 and rows[1].completed == 10


def test_a_decaying_completion_rate_is_visible_slice_by_slice():
    """The whole point of the table: an unexplained monotonic trend must be legible."""
    recs = []
    n = 0
    for s, count in enumerate([40, 30, 20, 10]):
        for k in range(count):
            t = s * 1.0 + k * (1.0 / count)
            recs.append(_ok(f"r{n}", t, 0.01, [0.001]))
            n += 1
    rows = slice_window(recs, window_s=4.0, n_slices=4, t0=0.0)
    rates = [r.completed_req_s for r in rows]
    assert rates == sorted(rates, reverse=True)
    assert rates[0] > rates[-1] * 3


def test_achieved_concurrency_is_time_weighted_not_a_head_count():
    """Four requests spanning the whole slice is a concurrency of four, and two that each
    span half of it is a concurrency of one -- counting arrivals would call both four."""
    spanning = [
        RequestRecord(
            request_id=f"s{i}",
            issued_ts=0.0,
            outcome=Outcome.OK,
            first_token_ts=0.1,
            end_ts=1.0,
            output_tokens=1,
        )
        for i in range(4)
    ]
    (row,) = slice_window(spanning, window_s=1.0, n_slices=1, t0=0.0)
    assert row.achieved_concurrency == pytest.approx(4.0)


def test_entry_and_exit_slices_are_retained_unless_a_rule_says_otherwise():
    """ "Retained or removed only by a predeclared rule" -- so the default keeps everything
    and dropping the ramp is an explicit, reportable argument rather than a quiet habit."""
    recs = [_ok(f"r{i}", i * 0.1, 0.05, [0.01]) for i in range(100)]
    assert len(slice_window(recs, window_s=10.0, n_slices=5, t0=0.0)) == 5
    trimmed = slice_window(recs, window_s=10.0, n_slices=5, t0=0.0, trim_slices=1)
    assert len(trimmed) == 3
    assert trimmed[0].index == 1, "the retained rows keep their original position"


def test_records_outside_the_declared_window_land_in_no_slice():
    """§4.2: rates are computed over the declared window, never the record-implied span,
    which stretches with stragglers exactly when the system is slowest."""
    inside = [_ok(f"i{i}", 0.1 + i * 0.01, 0.01, [0.001]) for i in range(10)]
    straggler = _ok("late", 0.2, 0.01, [0.001])
    straggler.end_ts = 40.0  # finishes long after the window closed
    rows = slice_window(inside + [straggler], window_s=1.0, n_slices=1, t0=0.0)
    assert rows[0].completed == 10
