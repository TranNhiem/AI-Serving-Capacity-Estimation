"""Tests for the per-rung dispersion block in capacity reports.

Section 7 requires at least three independent repetitions and their dispersion; every
report the harness emitted before this block published exactly one window per rung. On a
GB200 media ladder the three counted windows at concurrency 32 measured ttft_p95 of
8.769, 7.972 and 9.412 s and the report published 7.972 -- the fastest -- with nothing
to say the rung had spanned 18 percent.
"""

from __future__ import annotations

import json
from pathlib import Path

import jsonschema

from ascep.bench.ladder import RepetitionResult
from ascep.bench.metrics import WindowSummary
from ascep.bench.run import _counted, _dispersion, _measured

SCHEMA_PATH = Path(__file__).resolve().parents[1] / "schemas" / "run.schema.json"


def _summary(**overrides):
    """A WindowSummary with every figure measured, overridable field by field."""
    fields = {
        "n_issued": 100,
        "n_completed": 100,
        "n_latency_samples": 100,
        "excluded_error_count": 0,
        "excluded_invalid_count": 0,
        "excluded_warmup_count": 0,
        "error_rate_pct": 0.0,
        "ttft_p50_s": 1.0,
        "ttft_p95_s": 2.0,
        "ttft_p99_s": 3.0,
        "itl_p50_s": 0.05,
        "itl_p95_s": 0.1,
        "itl_p99_s": 0.2,
        "itl_population": "pooled-gaps",
        "e2e_p50_s": 4.0,
        "e2e_p95_s": 5.0,
        "e2e_p99_s": 6.0,
        "output_tok_s": 500.0,
        "requests_per_s": 10.0,
        "goodput_tok_s": 500.0,
        "slo_pass": True,
    }
    fields.update(overrides)
    return WindowSummary(**fields)


def _rep(repetition, post_search=False, **overrides):
    return RepetitionResult(
        concurrency=32,
        repetition=repetition,
        summary=_summary(**overrides),
        post_search=post_search,
    )


def _row_schema():
    with SCHEMA_PATH.open(encoding="utf-8") as handle:
        schema = json.load(handle)
    return schema["properties"]["results"]["items"]


def _row():
    """The smallest rung row the schema accepts, with the keys the harness always writes."""
    return {
        "input_tokens": 100,
        "output_tokens": 200,
        "concurrency": 32,
        "ttft_p50_s": 1.0,
        "ttft_p95_s": 2.0,
        "ttft_p99_s": 3.0,
        "itl_p50_s": 0.05,
        "itl_p95_s": 0.1,
        "itl_population": "pooled-gaps",
        # A pooled-gaps row owes the coalescing factor or the schema rejects it: without
        # this key the fixture would fail validation for a reason that has nothing to do
        # with dispersion, and the three schema tests below would report the wrong defect.
        "tokens_per_stream_chunk": 1.0,
        "e2e_p95_s": 5.0,
        "e2e_p99_s": 6.0,
        "output_tok_s": 500.0,
        "requests_per_s": 10.0,
        "gpu_util_pct": None,
        "gpu_util_pct_u_reason": "(U) a load generator cannot see the GPU.",
        "gpu_mem_util_pct": None,
        "gpu_mem_util_pct_u_reason": "(U) a load generator cannot see the GPU.",
        "error_rate_pct": 0.0,
        "slo_pass": True,
        "outcome": "complete",
        "provenance": "M",
    }


def test_dispersion_reports_lower_median_minimum_and_maximum_of_counted_repetitions():
    """A rung whose windows measure 9.412, 7.972 and 8.769 s publishes one of them and
    nothing else; a block that averaged, or picked any window but the extremes and the
    lower median, would leave the reader unable to see the rung spanned 16.42 percent."""
    reps = [
        _rep(1, ttft_p95_s=9.412),
        _rep(2, ttft_p95_s=7.972),
        _rep(3, ttft_p95_s=8.769),
    ]
    block, reason = _dispersion(_counted(reps))
    assert reason is None
    assert block["repetitions_counted"] == 3
    assert block["ttft_p95_s"] == {
        "min": 7.972,
        "median": 8.769,
        "max": 9.412,
        "n": 3,
        "spread_pct": 16.42,
    }


def test_median_is_the_lower_median_not_the_average_of_the_middle_two():
    """An even count of repetitions is where the lower-median convention bites: averaging
    the middle two would publish 2.5, a value no window measured, and the block would
    drift into meaning something different from the row picker's median."""
    reps = [_rep(i, ttft_p95_s=float(i)) for i in (1, 2, 3, 4)]
    block, reason = _dispersion(_counted(reps))
    assert reason is None
    assert block["repetitions_counted"] == 4
    assert block["ttft_p95_s"]["median"] == 2.0


def test_confirmation_repetition_does_not_widen_the_spread_it_was_never_counted_in():
    """The section 5 confirmation window is additional evidence about a boundary, never
    one of the three; letting its 100 s outlier set the max would publish a spread the
    rung's own grade excluded, and no reader could reconcile the block with the verdict."""
    reps = [
        _rep(1, ttft_p95_s=8.0),
        _rep(2, ttft_p95_s=9.0),
        _rep(3, ttft_p95_s=10.0),
        _rep(4, post_search=True, ttft_p95_s=100.0),
    ]
    block, reason = _dispersion(_counted(reps))
    assert reason is None
    assert block["repetitions_counted"] == 3
    assert block["ttft_p95_s"]["max"] == 10.0


def test_figure_null_in_one_repetition_reports_n_of_the_survivors():
    """A window whose reduction could not compute a tail left it null; reading that null
    as zero would invent a 0.0 endpoint no window measured and triple the published
    spread, while claiming n of 3 would hide that the spread stands on two windows."""
    reps = [
        _rep(1, itl_p95_s=None),
        _rep(2, itl_p95_s=0.1),
        _rep(3, itl_p95_s=0.3),
    ]
    block, reason = _dispersion(_counted(reps))
    assert reason is None
    entry = block["itl_p95_s"]
    assert entry["n"] == 2
    assert entry["min"] == 0.1
    assert entry["median"] == 0.1
    assert entry["max"] == 0.3
    assert entry["spread_pct"] == 200.0


def test_figure_null_in_every_repetition_is_null_with_a_reason_not_absent_not_zero():
    """A figure no window produced must come back null with a (U) reason: a zero would
    read as a measured spread of nothing, and an absent key would read as a rung that
    never looked when the truth is that it looked and the reduction found nothing."""
    reps = [_rep(i, itl_p95_s=None) for i in (1, 2, 3)]
    block, reason = _dispersion(_counted(reps))
    assert reason is None
    assert "itl_p95_s" in block
    assert block["itl_p95_s"] is None
    assert block["itl_p95_s_u_reason"].startswith("(U) ")


def test_single_counted_repetition_carries_null_dispersion_and_a_u_reason():
    """One window has no spread; a block of identical min, median and max would publish
    perfect stability as a finding when nothing was measured twice, and omitting the key
    would make an unmeasured rung indistinguishable from one the harness never filled."""
    reps = [_rep(1)]
    dispersion, reason = _dispersion(_counted(reps))
    assert dispersion is None
    assert reason is not None
    assert "1" in reason
    row = {"dispersion_u_reason": "TODO"}
    _measured(row, "dispersion", dispersion, reason)
    assert row["dispersion"] is None
    assert row["dispersion_u_reason"].startswith("(U) ")
    assert "1" in row["dispersion_u_reason"]


def test_zero_median_gives_null_spread_pct_with_a_reason_rather_than_raising():
    """error_rate_pct is zero-median on every healthy ladder, so this branch is the normal
    case: dividing by it would either raise or publish inf, and a relative spread against
    a zero median is a division by zero dressed as a statistic."""
    reps = [_rep(i, error_rate_pct=0.0) for i in (1, 2, 3)]
    block, reason = _dispersion(_counted(reps))
    assert reason is None
    entry = block["error_rate_pct"]
    assert entry["min"] == 0.0
    assert entry["median"] == 0.0
    assert entry["max"] == 0.0
    assert entry["n"] == 3
    assert entry["spread_pct"] is None
    assert entry["spread_pct_u_reason"].startswith("(U) ")


def test_report_carrying_the_dispersion_block_validates_against_the_schema():
    """A block the schema rejects would fail validation on every fresh report, turning a
    reporting gap into a harness defect the operator did not create."""
    row = _row()
    block, reason = _dispersion(_counted([_rep(1), _rep(2), _rep(3)]))
    assert reason is None
    row["dispersion"] = block
    jsonschema.validate(instance=row, schema=_row_schema())


def test_report_with_null_dispersion_and_a_reason_validates_against_the_schema():
    """A one-repetition rung must be able to say so; a schema that rejected the explicit
    null would force the harness to choose between invalid output and a fabricated block
    of identical min, median and max."""
    row = _row()
    row["dispersion"] = None
    row["dispersion_u_reason"] = (
        "(U) this rung had 1 counted repetition(s); a spread needs two windows"
    )
    jsonschema.validate(instance=row, schema=_row_schema())


def test_report_without_dispersion_still_validates_against_the_schema():
    """The property is additive and optional: reports written before this change are
    evidence, and a schema tightening they predate must not invalidate them."""
    jsonschema.validate(instance=_row(), schema=_row_schema())
