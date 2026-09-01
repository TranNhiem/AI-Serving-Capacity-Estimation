"""The negative corpus: eight reports that are wrong in exactly one way each.

Every other test here asks whether a correct report passes. These ask the harder question --
whether an incorrect one fails, and fails for the stated reason rather than by luck.

The first draft of this corpus is the reason the tests below look the way they do. It mutated
the published moe-26b example, which is honestly ``partial`` and already trips C4, C6, C7 and
C8 before anyone touches it. Against that baseline five of the eight cases asserted a rule the
report was already breaking, so they passed without the mutation doing anything; one asserted a
rule that never fired at all; and one mutation deleted a stale justification, which *removed* a
finding. Seven of the eight were also rejected by the JSON Schema before the grader ever ran,
so they demonstrated schema validation while claiming to demonstrate conformance grading.

Hence three assertions per case, none of which that draft would have survived: the report is
schema-valid, so the grader is what caught it; the findings it adds to the baseline are all of
the stated rule, so the mutation caused them; and one of them names the field that was edited,
so the case shows the defect it advertises rather than a side effect somewhere else.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from ascep.conformance import check
from ascep.validation import validate

REPO_ROOT = Path(__file__).resolve().parents[1]
CORPUS_ROOT = REPO_ROOT / "examples" / "negative"

sys.path.insert(0, str(CORPUS_ROOT))
from build_corpus import CASES, edit  # noqa: E402

BASELINE = json.loads((CORPUS_ROOT / "baseline.json").read_text(encoding="utf-8"))
CASE_IDS = [c[0] for c in CASES]


def _findings(report: dict) -> set[tuple[str, str, str]]:
    return {(f.rule, f.path, f.message) for f in check(report).findings}


BASELINE_FINDINGS = _findings(BASELINE)


def _report(case_id: str) -> dict:
    return json.loads((CORPUS_ROOT / case_id / "report.json").read_text(encoding="utf-8"))


# --- the baseline is the thing that makes attribution possible ------------------------


def test_the_baseline_grades_conforming_with_no_findings_at_all():
    """Not "grades conforming" -- zero findings.

    A baseline that carries warnings still lets a case pass on a finding it did not cause, and
    the warning is invisible in the assertion that swallowed it.
    """
    verdict = check(BASELINE)
    assert not verdict.findings, [(f.rule, f.path) for f in verdict.findings]
    assert verdict.level == "conforming"


def test_the_baseline_is_schema_valid():
    assert validate("capacity-report", BASELINE) == []


def test_the_baseline_says_in_its_own_note_that_it_is_a_fixture():
    """It is a constructed document, not a run.

    Every other report in this repository is a measurement, and a conforming one that is not
    would be the most quotable thing here -- a clean set of numbers with nothing marked
    unmeasured. It has to refuse that reading in its own text, because the file travels
    without the directory it came from.
    """
    note = BASELINE.get("conformance_note", "")
    assert "fixture" in note.lower()
    assert "not a measurement" in note.lower()


# --- each case: one defect, one rule, at the field the README points at ---------------


@pytest.mark.parametrize(("case_id", "rule", "edits"), [c[:3] for c in CASES], ids=CASE_IDS)
def test_one_case_adds_findings_of_exactly_the_rule_it_claims(case_id, rule, edits):
    report = _report(case_id)

    # Schema-valid, or the grader is not what caught this and the case is mislabelled.
    assert validate("capacity-report", report) == [], "the schema rejects this before C-rules"

    added = _findings(report) - BASELINE_FINDINGS
    assert added, "the mutation changed nothing the checker noticed"
    assert {r for r, _, _ in added} == {rule}, sorted({r for r, _, _ in added})

    # The first edit is the defect; any others are the _u_reason that keeps the case honest.
    path = edits[0][0]
    assert any(path in found_at for _, found_at, _ in added), sorted(f for _, f, _ in added)


@pytest.mark.parametrize(("case_id", "rule", "edits"), [c[:3] for c in CASES], ids=CASE_IDS)
def test_no_case_removes_a_finding_the_baseline_did_not_have(case_id, rule, edits):
    """The failure mode that sank the first draft: one mutation deleted a stale justification
    and took a finding away with it, so the case was published as evidence of a rule firing
    while it was evidence of one going quiet."""
    assert not BASELINE_FINDINGS - _findings(_report(case_id))


@pytest.mark.parametrize(("case_id", "rule", "edits"), [c[:3] for c in CASES], ids=CASE_IDS)
def test_the_published_report_is_the_one_the_builder_produces(case_id, rule, edits):
    """Committed artifacts drift from the script that made them, and a corpus that has drifted
    is documentation of a mutation nobody can regenerate."""
    expected = json.loads(json.dumps(BASELINE))
    for path, value in edits:
        edit(expected, path, value)
    assert _report(case_id) == expected


@pytest.mark.parametrize(("case_id", "rule", "edits"), [c[:3] for c in CASES], ids=CASE_IDS)
def test_the_case_readme_names_the_rule_and_the_field(case_id, rule, edits):
    """A reader arrives at the README, not at this file. If it names the wrong rule the corpus
    is teaching the wrong lesson, and nothing else here would notice."""
    readme = (CORPUS_ROOT / case_id / "README.md").read_text(encoding="utf-8")
    assert rule in readme
    assert edits[0][0] in readme
    assert "TODO" not in readme, "the explanation was never written"


def test_every_rule_from_c1_to_c8_has_a_case_or_a_stated_reason_it_cannot():
    """The gap this catches is a corpus that quietly covers six rules and looks like eight.

    A rule with no reachable schema-valid mutation is a real finding about where enforcement
    actually lives, and it belongs in the corpus README where a reader will see it -- not in a
    missing directory.
    """
    covered = {c[1] for c in CASES}
    index = (CORPUS_ROOT / "README.md").read_text(encoding="utf-8")
    for n in range(1, 9):
        rule = f"C{n}"
        assert rule in covered or rule in index, f"{rule} has neither a case nor an explanation"


def test_the_corpus_index_lists_every_case_directory():
    """A case that exists on disk and not in the index is a case nobody reads, and the next
    person to add one copies the last one -- index entry included or not."""
    index = (CORPUS_ROOT / "README.md").read_text(encoding="utf-8")
    on_disk = {p.name for p in CORPUS_ROOT.iterdir() if p.is_dir() and (p / "report.json").exists()}
    assert on_disk == set(CASE_IDS), "a case directory exists that the builder does not produce"
    for case_id in CASE_IDS:
        assert case_id in index, f"{case_id} is not listed in the corpus README"
