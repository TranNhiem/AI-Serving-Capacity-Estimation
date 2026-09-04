"""What the draft report ``ascep bench`` writes actually claims.

Split out of ``test_cli_bench``, which grades the command's contract. These grade the
document it leaves behind: which floor the ladder observed and how it is named, which rows
are tagged (U) because a load generator cannot know them, and the rule that the Measured
tier is the engine ceiling with the gates ignored while the Sustainable tier is not. A draft
that is wrong in these ways still validates, still verifies against its bundle, and is
still read by whoever cites it.
"""

from __future__ import annotations

import json
import pathlib

import pytest
from bench_cli_support import (
    _config,
    _report,
    _run_offline,
    _write,
)

from ascep import conformance
from ascep.bench import ladder
from ascep.bench import report as bench_report
from ascep.cli import main

pytest.importorskip("httpx", reason="ascep bench needs the [run] extra")

# --- the draft names the floor it observed and tags the rows it left empty -------------


def _offline_report(tmp_path: pathlib.Path, monkeypatch) -> dict:
    """Run the whole offline ladder and return the draft it wrote."""
    path = _write(tmp_path, _config(tmp_path))
    _run_offline(monkeypatch)
    assert main(["bench", path]) == 0, "the offline ladder did not complete"
    return _report(tmp_path)


def _hand_built_ladder(*rungs):
    """A LadderResult whose only content is its rung verdicts.

    Nothing else varies, because the constraint label must follow from the rungs alone and
    not from the narrative -- censoring, monotonicity, a cache caveat -- the ladder attaches
    around them.
    """
    return ladder.LadderResult(
        rungs={rung.concurrency: rung for rung in rungs},
        terminated_at=None,
        monotone=True,
        bisection_permitted=True,
        is_lower_bound=False,
        censoring_cause=None,
        max_sustainable_concurrency=None,
        confirmed=False,
        sustainable_publishable=False,
        cache_caveat=None,
    )


def test_a_gated_failure_above_the_tier_labels_the_constraint_slo():
    """A tier stating a concurrency with a null constraint is a C5 error the run itself could
    have answered: the failed rung above is the observed floor, and chapter 5 puts the slo
    label there. Withholding it made every draft this harness emits non-conforming."""
    result = _hand_built_ladder(
        ladder.RungResult(concurrency=4, outcome=ladder.RungOutcome.COMPLETE),
        ladder.RungResult(concurrency=8, outcome=ladder.RungOutcome.FAILED),
    )
    assert bench_report._boundary_constraint(result, 4) == "slo"


def test_a_failure_that_delivered_nothing_labels_the_constraint_throughput():
    """Zero completions is a throughput collapse, not a missed latency gate. Grading it slo
    would send the author renegotiating latency promises against a system that had stopped
    delivering at all."""
    result = _hand_built_ladder(
        ladder.RungResult(concurrency=4, outcome=ladder.RungOutcome.COMPLETE),
        ladder.RungResult(concurrency=8, outcome=ladder.RungOutcome.FAILED, zero_completions=True),
    )
    assert bench_report._boundary_constraint(result, 4) == "throughput"


def test_the_lowest_failing_rung_above_the_tier_is_the_boundary_that_labels_it():
    """Taking any failing rung instead of the lowest lets the collapse at 16 shadow the missed
    gate at 8: the report would print throughput where the climb actually broke on slo, and
    the operator would buy bandwidth to fix a latency promise."""
    result = _hand_built_ladder(
        ladder.RungResult(concurrency=4, outcome=ladder.RungOutcome.COMPLETE),
        ladder.RungResult(concurrency=8, outcome=ladder.RungOutcome.FAILED),
        ladder.RungResult(concurrency=16, outcome=ladder.RungOutcome.FAILED, zero_completions=True),
    )
    assert bench_report._boundary_constraint(result, 4) == "slo"


def test_no_failing_rung_above_the_tier_means_no_constraint_is_named():
    """A ladder exhausted without failure measured "at least this much"; a label would print
    that lower bound as a maximum. An ABORTED rung is failure evidence by cause rather than
    evidence that a floor binds, and a rung that failed BELOW the tier was climbed past, so
    neither may conjure a label either."""
    exhausted = _hand_built_ladder(
        ladder.RungResult(concurrency=4, outcome=ladder.RungOutcome.COMPLETE),
        ladder.RungResult(concurrency=8, outcome=ladder.RungOutcome.COMPLETE),
    )
    assert bench_report._boundary_constraint(exhausted, 4) is None
    aborted_above = _hand_built_ladder(
        ladder.RungResult(concurrency=4, outcome=ladder.RungOutcome.COMPLETE),
        ladder.RungResult(concurrency=8, outcome=ladder.RungOutcome.ABORTED),
    )
    assert bench_report._boundary_constraint(aborted_above, 4) is None
    failed_below = _hand_built_ladder(
        ladder.RungResult(concurrency=2, outcome=ladder.RungOutcome.FAILED),
        ladder.RungResult(concurrency=4, outcome=ladder.RungOutcome.COMPLETE),
    )
    assert bench_report._boundary_constraint(failed_below, 4) is None


def test_an_exhausted_ladder_leaves_the_constraint_unknown_beside_a_filled_tier(
    tmp_path, monkeypatch
):
    """The offline ladder completes every declared rung, so its figure is a lower bound.
    Labelling it with a constraint would print "at least this much" as "this much at most";
    the C5 error on the null is the correct grade and must not be silenced."""
    tier = _offline_report(tmp_path, monkeypatch)["capacity_tiers"]["measured"]
    assert tier["max_concurrent_users"] is not None
    assert tier["binding_constraint"] is None
    assert tier["binding_constraint_u_reason"].startswith("(U)")


def test_every_row_bench_leaves_empty_carries_provenance_u(tmp_path, monkeypatch):
    """A null provenance on an empty row is a C1 error nothing can justify, because the schema
    defines no provenance_u_reason to put a sibling reason in. "U" is the only tag that says
    the row states nothing, so a null here means bench emitted an unanswerable C1 again."""
    report = _offline_report(tmp_path, monkeypatch)
    tiers = report["capacity_tiers"]
    assert tiers["theoretical"]["provenance"] == "U"
    assert tiers["recommended"]["provenance"] == "U"
    assert report["sizing_result"]["provenance"] == "U"


def test_a_campaign_at_one_context_length_declares_single_point(tmp_path, monkeypatch):
    """An unlabelled single point reads as a curve. The flag does not raise the grade -- it
    states the limit the grade already reflects -- but leaving the default stands the draft up
    as a multi-context measurement it never was."""
    assert _offline_report(tmp_path, monkeypatch)["run"]["single_point"] is True


def test_rung_means_that_differ_only_by_sampling_noise_are_one_context_length():
    """The six figures below are the per-rung context means a real GB200 ladder measured at a
    single declared 1,500-token shape. Counted as a set they are six context lengths, and the
    draft published single_point false: a context curve nobody measured, with C4's finding
    silenced for exactly the campaign it was written for. Since context_tokens is always a
    mean, that made the flag unreachable for every real run of bench.
    """
    noise = [
        {"context_tokens": length}
        for length in (2043.65, 2043.94, 2045.28, 2045.46, 2045.48, 2046.50)
    ]
    assert bench_report._distinct_context_lengths(noise) == 1

    # And the tolerance must not swallow a curve: these are three shapes a report could
    # legitimately interpolate over, and collapsing them would suppress the opposite error.
    curve = [{"context_tokens": length} for length in (1500.0, 4000.0, 16000.0)]
    assert bench_report._distinct_context_lengths(curve) == 3


def test_the_emitted_draft_carries_no_null_it_cannot_justify(tmp_path, monkeypatch):
    """Every C1 finding a draft carries is the harness's doing, not the run's: bench chooses
    what to null and what to say about it, so an unjustified null here is unfixable by the
    operator who ran it. This is the regression test for the whole class."""
    verdict = conformance.check(_offline_report(tmp_path, monkeypatch))
    c1 = [
        f"{finding.path}: {finding.message}" for finding in verdict.findings if finding.rule == "C1"
    ]
    assert not c1, "bench emitted a null C1 cannot accept:\n  " + "\n  ".join(c1)


def test_a_decidable_boundary_reaches_both_filled_tiers(tmp_path, monkeypatch):
    """The rule and the report are separate failures: `_boundary_constraint` can be right
    while the tier it should label is still emitted null, which is the C5 error the operator
    cannot fix. The offline ladder never fails a rung, so the boundary is stubbed here --
    what is under test is the wiring, not the rule the tests above cover directly."""
    monkeypatch.setattr(bench_report, "_boundary_constraint", lambda result, concurrency: "slo")
    tiers = _offline_report(tmp_path, monkeypatch)["capacity_tiers"]
    for name in ("measured", "sustainable"):
        assert tiers[name]["binding_constraint"] == "slo", f"{name} dropped the label"
        assert "binding_constraint_u_reason" not in tiers[name], (
            f"{name} states a constraint and keeps the reason it was unknown; a stale (U) "
            "tells a reviewer to discount a figure the run actually pinned down"
        )


def test_the_draft_note_bench_writes_is_the_one_the_checker_recognises(tmp_path, monkeypatch):
    """`ascep conformance --raise` finds the paragraph it may replace by an exact prefix
    match. If bench ever builds that paragraph inline again, the match silently stops firing
    and every graded report keeps a note insisting it was never graded."""
    note = _offline_report(tmp_path, monkeypatch)["conformance_note"]
    assert note.startswith(conformance.DRAFT_NOTE)


def test_a_bench_draft_can_be_graded_up_by_the_command_its_note_names(tmp_path, monkeypatch):
    """The note tells the reader that `ascep conformance` is the command that may raise the
    claim. Unfulfilled, that sentence sends every operator to a command that prints a grade
    and throws it away, and the published file claims the harness floor forever.

    The boundary is stubbed for the same reason as the wiring test above: the offline ladder
    never fails a rung, so its C5 errors stand and it grades `non-conforming` honestly. A
    real ladder that found its boundary is the case this promise is made to."""
    monkeypatch.setattr(bench_report, "_boundary_constraint", lambda result, concurrency: "slo")
    path = _write(tmp_path, _config(tmp_path))
    _run_offline(monkeypatch)
    assert main(["bench", path]) == 0
    report_path = tmp_path / "report.json"
    assert main(["conformance", str(report_path), "--raise"]) == 0
    raised = json.loads(report_path.read_text(encoding="utf-8"))
    assert raised["conformance"] == "partial"
    assert raised["conformance_note"].startswith(conformance.GRADED_NOTE)


# --- the measured tier is the engine ceiling, gates ignored (chapter 5 §5.5) ----------


def _offline_report_with(tmp_path: pathlib.Path, monkeypatch, *, offline=None, **overrides):
    """Run the offline ladder under an altered config and return the draft it wrote."""
    path = _write(tmp_path, _config(tmp_path, **overrides))
    _run_offline(monkeypatch, **(offline or {}))
    assert main(["bench", path]) == 0, "the offline ladder did not complete"
    return _report(tmp_path)


def _only_the_top_rung_fails(tmp_path, monkeypatch):
    """A report from the [1, 2, 4] ladder whose rungs 1 and 2 pass and whose rung 4 misses TTFT.

    The failure is driven through a declared gate rather than by patching the grader, because
    a test that fakes the grading cannot catch a bug that lives in which graded rungs the
    tier is selected from. The latencies stay far inside the window: a rung slow enough to
    outlast ``window_s`` would leave no in-window records and grade INVALID, which claims no
    operating point and would exercise a different branch than the one under test.
    """
    return _offline_report_with(
        tmp_path,
        monkeypatch,
        offline={"ttft_by_rung": {4: 0.05}},
        **{"slo_gates.ttft_p95_max_s": 0.01},
    )


def test_a_rung_that_failed_its_gates_still_sets_the_engine_ceiling(tmp_path, monkeypatch):
    """Selecting on COMPLETE alone published the highest *passing* rung as the ceiling.

    That collapses measured onto sustainable and tells the reader the engine stops where the
    SLO stops, erasing the one distinction the two tiers exist to draw: here the report would
    claim 2 streams when the engine plainly carried 4.
    """
    tiers = _only_the_top_rung_fails(tmp_path, monkeypatch)["capacity_tiers"]
    assert tiers["measured"]["max_concurrent_users"] == 4
    assert tiers["sustainable"]["max_concurrent_users"] == 2
    assert tiers["measured"]["provenance"] == "M"


def test_the_ceiling_on_a_failed_top_rung_names_what_stopped_it(tmp_path, monkeypatch):
    """`_boundary_constraint` searches only *above* a rung for the floor that bound it.

    On a ladder that failed at its top rung there is nothing above, so the headline tier came
    out saying "the engine stops at 4 streams" while declining to say what stopped it -- the
    one fact a reader sizes against. The rung missed a latency gate, so the floor is slo.
    """
    measured = _only_the_top_rung_fails(tmp_path, monkeypatch)["capacity_tiers"]["measured"]
    # Anchored to the failed rung on purpose: the old selection put this tier on rung 2,
    # which does have a failing rung above it, so the label came out right while the figure
    # it labelled was the wrong one.
    assert measured["max_concurrent_users"] == 4
    assert measured["binding_constraint"] == "slo"
    assert measured.get("binding_constraint_u_reason") is None, (
        "a stale (U) beside a named constraint tells a reviewer to discount a figure the "
        "run actually pinned down"
    )


def test_the_two_tiers_do_not_agree_across_a_failure_inside_the_envelope(tmp_path, monkeypatch):
    """C7 is the checker's name for the erasure, and it must not fire on the harness's own draft.

    A rung inside this workload's context envelope failed its gate, so the engine ceiling and
    what users can rely on are different numbers by definition. The harness publishing them as
    equal is precisely "a results row at or below the workload's average context failed its
    SLO gate, yet the sustainable tier still equals the measured tier".
    """
    report = _only_the_top_rung_fails(tmp_path, monkeypatch)
    c7 = [
        f"{finding.path}: {finding.message}"
        for finding in conformance.check(report).findings
        if finding.rule == "C7"
    ]
    assert not c7, "bench emitted the C7 it is meant to avoid:\n  " + "\n  ".join(c7)


def test_a_ladder_with_no_passing_rung_still_publishes_the_ceiling_it_measured(
    tmp_path, monkeypatch
):
    """A gate nothing can meet used to produce an all-null measured tier.

    The draft then said "no rung completed its declared repetitions", telling the reader
    nothing was measured when the harness had measured exactly where the SLO stops. It is the
    sustainable tier that has nothing to say here, and it must justify saying it.
    """
    report = _offline_report_with(tmp_path, monkeypatch, **{"slo_gates.ttft_p95_max_s": 0.0005})
    measured = report["capacity_tiers"]["measured"]
    assert measured["max_concurrent_users"] is not None
    assert measured["binding_constraint"] == "slo"
    sustainable = report["capacity_tiers"]["sustainable"]
    assert sustainable["max_concurrent_users"] is None
    assert sustainable["max_concurrent_users_u_reason"].startswith("(U)")


def test_a_failed_top_rung_names_the_floor_it_observed():
    """A rung that missed a gate and one that completed nothing are different floors.

    Conflating them would tell an operator to buy throughput when the deployment actually
    breached a latency promise it could otherwise have met, or the reverse.
    """
    missed_gate = _hand_built_ladder(
        ladder.RungResult(concurrency=4, outcome=ladder.RungOutcome.COMPLETE),
        ladder.RungResult(concurrency=8, outcome=ladder.RungOutcome.FAILED),
    )
    assert bench_report._observed_constraint(missed_gate, 8) == "slo"
    collapsed = _hand_built_ladder(
        ladder.RungResult(concurrency=4, outcome=ladder.RungOutcome.COMPLETE),
        ladder.RungResult(concurrency=8, outcome=ladder.RungOutcome.FAILED, zero_completions=True),
    )
    assert bench_report._observed_constraint(collapsed, 8) == "throughput"


def test_reading_the_floor_off_a_rung_does_not_shadow_the_search_above_it():
    """The rung-level read is a fallback for the top rung, not a replacement for the old rule.

    A tier sitting on a COMPLETE rung is still labelled by the lowest failing rung above it,
    and a ladder exhausted without failure still names nothing -- printing a constraint beside
    an "at least this much" figure would read as a maximum the run never established.
    """
    failure_above = _hand_built_ladder(
        ladder.RungResult(concurrency=4, outcome=ladder.RungOutcome.COMPLETE),
        ladder.RungResult(concurrency=8, outcome=ladder.RungOutcome.FAILED),
    )
    assert bench_report._observed_constraint(failure_above, 4) == "slo"
    exhausted = _hand_built_ladder(
        ladder.RungResult(concurrency=4, outcome=ladder.RungOutcome.COMPLETE),
        ladder.RungResult(concurrency=8, outcome=ladder.RungOutcome.COMPLETE),
    )
    assert bench_report._observed_constraint(exhausted, 4) is None
