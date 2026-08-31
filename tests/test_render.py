"""Acceptance tests for `ascep.render`, the module that turns a report dict into Markdown.

The renderer is the protocol's public face: the JSON is for machines, but the Markdown is what
a reviewer actually reads. That makes render bugs a specific and nasty class — not crashes,
but *quietly wrong documents*. A dropped section heading, a `None` where an em-dash should be,
or a conformance note that never makes it onto the page all produce output that looks
finished while saying less than the data does. These tests exist to stop the renderer from
publishing a report that is thinner than the JSON it came from.

They are written against the real example report wherever possible, because a renderer that
only works on fabricated fixtures proves nothing about the artefact people will actually
render first.
"""

from __future__ import annotations

import copy
import json
import pathlib

import pytest

from ascep import render

ROOT = pathlib.Path(__file__).parent.parent
REPORT_PATH = ROOT / "examples" / "moe-26b-h100-tp2" / "report.json"

TITLE_PREFIX = "# ASCEP Capacity Report — "


@pytest.fixture
def report() -> dict:
    """A fresh copy per test — several tests here mutate it."""
    return json.loads(REPORT_PATH.read_text(encoding="utf-8"))


@pytest.fixture
def markdown(report) -> str:
    return render.render(report)


def _lines(markdown: str) -> list[str]:
    return markdown.splitlines()


def _table_blocks(lines: list[str]) -> list[list[str]]:
    """Group the document into its contiguous runs of table lines."""
    blocks: list[list[str]] = []
    current: list[str] = []
    for line in lines:
        if line.startswith("|"):
            current.append(line)
        elif current:
            blocks.append(current)
            current = []
    if current:
        blocks.append(current)
    return blocks


# --- overall document structure -------------------------------------------------------


def test_the_document_opens_with_the_standard_title(markdown):
    assert markdown.startswith(TITLE_PREFIX)


def test_all_eight_sections_are_present_exactly_once_and_in_order(markdown):
    """The section skeleton is the protocol's contract with readers. A renderer that drops
    or reorders a section publishes a report that looks complete and is not — the worst
    kind of defect, because nobody diffs a rendered page against the schema by eye."""
    lines = _lines(markdown)
    positions = []
    for n in range(1, 9):
        prefix = f"## {n}."
        matches = [i for i, line in enumerate(lines) if line.startswith(prefix)]
        assert len(matches) == 1, f"expected exactly one {prefix!r} heading, got {matches}"
        positions.append(matches[0])
    assert positions == sorted(positions), f"section headings out of order: {positions}"


# --- headline and conformance note ----------------------------------------------------


def test_there_is_exactly_one_headline_naming_users_and_gpus(report, markdown):
    """The headline is the one line most readers will quote; it must name the recommended
    tier's user count and the GPU count it rests on, taken from the report itself."""
    headline_lines = [line for line in _lines(markdown) if line.startswith("**Headline:**")]
    assert len(headline_lines) == 1, f"expected one headline, got {len(headline_lines)}"
    headline = headline_lines[0]

    recommended = report["capacity_tiers"]["recommended"]
    users = recommended["max_concurrent_users"]
    gpu_count = (
        recommended.get("n_gpus")
        or report["serving"].get("gpu_count")
        or report["hardware"].get("gpu_count")
    )
    assert render._format_number(users) in headline
    assert render._format_number(gpu_count) in headline


def test_the_conformance_note_is_rendered_with_its_real_text(report, markdown):
    """The note is what distinguishes a careful `partial` from a careless one; it is the
    whole reason the level is reviewable. The schema requires it, so a renderer that
    accepted the field and then dropped it from the page would silently defeat the level
    it is describing."""
    note_lines = [line for line in _lines(markdown) if line.startswith("**Conformance note.**")]
    assert len(note_lines) == 1, f"expected one conformance note, got {len(note_lines)}"
    note = report["conformance_note"]
    assert note, "fixture report has no conformance note; the test cannot exercise this path"
    assert render._clean(note) in note_lines[0]


def test_a_missing_conformance_note_renders_the_missing_fallback_not_blank(report):
    """An empty `**Conformance note.**` line reads as an editorial slip; the (U) fallback
    reads as what it is — a declared gap."""
    report.pop("conformance_note", None)
    markdown = render.render(report)
    note_lines = [line for line in _lines(markdown) if line.startswith("**Conformance note.**")]
    assert len(note_lines) == 1
    assert render._MISSING in note_lines[0]


# --- provenance tags ------------------------------------------------------------------


def test_every_provenance_tag_used_in_the_report_appears_in_the_output(markdown):
    """Tags are how a reader weights each number; output losing a tag class has silently
    re-graded the report's evidence."""
    for tag in ("(M)", "(I)", "(T)", "(U)"):
        assert tag in markdown, f"provenance tag {tag} never rendered"


def test_the_tag_legend_table_is_present(markdown):
    assert "| tag | meaning |" in markdown
    for tag in ("**(M)**", "**(I)**", "**(T)**", "**(U)**"):
        assert f"| {tag} |" in markdown, f"legend row for {tag} missing"


# --- declined and missing values ------------------------------------------------------


def test_a_declined_tier_renders_its_reason_not_a_bare_dash(report, markdown):
    """The theoretical tier is declined with a (U) reason in the real report. That reason
    must survive onto the page: a renderer that prints a bare em-dash turns a documented
    gap into an undocumented one, and the reader can no longer tell ' declined on purpose'
    from 'field fell over'."""
    theoretical = report["capacity_tiers"]["theoretical"]
    assert theoretical["max_concurrent_users"] is None, "fixture no longer exercises this path"
    reason_keys = [k for k in theoretical if k.endswith("_u_reason")]
    assert reason_keys, "declined tier carries no reason; fixture drifted"
    for key in reason_keys:
        reason = render._clean(theoretical[key])
        if reason.startswith("(U)"):
            reason = reason[3:].strip()
        assert reason, f"{key} is empty"
        assert f"— *(U) {reason}*" in markdown


def test_a_null_without_a_reason_renders_the_missing_fallback(report):
    """Delete the justification next to a null and the renderer must say so in words, not
    render nothing."""
    before = render.render(report)
    tier = report["capacity_tiers"]["theoretical"]
    tier.pop("max_concurrent_users_u_reason")
    after = render.render(report)
    assert after.count(render._MISSING) == before.count(render._MISSING) + 1


# --- render_file ----------------------------------------------------------------------


def test_render_file_returns_the_same_string_as_render(report):
    assert render.render_file(REPORT_PATH) == render.render(report)


def test_render_file_writes_exactly_what_it_returns(report, tmp_path):
    out = tmp_path / "report.md"
    returned = render.render_file(REPORT_PATH, out=out)
    assert out.read_text(encoding="utf-8") == returned
    assert returned.startswith(TITLE_PREFIX)


# --- robustness -----------------------------------------------------------------------


def test_render_does_not_mutate_its_input(report):
    before = copy.deepcopy(report)
    render.render(report)
    assert report == before


def test_rendering_an_empty_report_does_not_crash():
    """The renderer is the tool a contributor points at a half-finished report to see what
    is still missing; crashing on incompleteness makes it useless exactly when it is most
    needed."""
    markdown = render.render({})
    assert isinstance(markdown, str)
    assert markdown.startswith(TITLE_PREFIX)


def test_rendering_a_nearly_empty_report_does_not_crash():
    markdown = render.render({"ascep_version": "0.1.0"})
    assert isinstance(markdown, str)
    assert markdown.startswith(TITLE_PREFIX)


# --- markdown hygiene -----------------------------------------------------------------


def test_no_python_none_leaks_into_the_rendered_output(markdown):
    """A literal `None` in a value slot is the visible symptom of a value that reached a cell
    without going through the (U) path — someone forgot the missing-value branch.

    Matched per *cell*, not per line. An author is entitled to begin a sentence with the
    English word "None" inside a prose cell, and the real report does: one assumption reads
    "None for these figures, but they MUST NOT be extrapolated past 8 GPUs". A substring
    search over whole lines flags that as a defect, which is the failure mode this whole file
    exists to avoid — a check that cries wolf on honest output gets switched off, and then it
    is not checking anything.
    """
    offenders = [
        cell
        for line in _lines(markdown)
        if line.startswith("|")
        for cell in line.strip("|").split("|")
        if cell.strip().strip("`*_ ") == "None"
    ]
    assert not offenders, offenders


def test_every_table_block_is_well_formed(markdown):
    """A table with a ragged row renders on GitHub as a wall of pipes and none of the data
    is readable; checking pipe counts per block is cheap insurance. Escaped `\\|` inside a
    cell is content, not a column boundary, so it is stripped before counting."""
    lines = _lines(markdown)
    blocks = _table_blocks(lines)
    assert blocks, "expected at least one Markdown table in the rendered report"
    for block in blocks:
        assert len(block) >= 2, f"table block with no separator row: {block[0]!r}"
        separator_chars = set(block[1].replace("\\|", ""))
        assert separator_chars <= set("|-: "), f"second table line is not a separator: {block[1]!r}"
        widths = {line.replace("\\|", "").count("|") for line in block}
        assert len(widths) == 1, f"ragged table (pipe counts {widths}): {block[0]!r}"
