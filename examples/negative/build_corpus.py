"""Regenerate the negative corpus: eight reports, each wrong in exactly one way.

Every case is ``baseline.json`` plus one edit and nothing else, so the finding the checker
reports can be attributed to that edit and to no other cause. Two properties make that
attribution hold, and both are enforced by ``tests/test_negative_corpus.py`` rather than here.

The baseline grades ``conforming`` with zero findings. An earlier draft of this corpus mutated
the published moe-26b example, which is honestly ``partial`` and already trips C4, C6, C7 and
C8. Five of the eight cases then "passed" on findings the baseline had before the mutation, and
one mutation removed a finding instead of adding one. A corpus built on a report that is
already broken cannot show a reader which break is the one being demonstrated.

Every case stays valid against the JSON Schemas. A defect the schema rejects never reaches the
grader, so publishing it as a C-rule example teaches the wrong lesson about which layer caught
what. Where a rule cannot be reached that way the corpus README says so, rather than shipping a
schema-invalid report that appears to prove otherwise.

The eight edits below were not chosen by reading the checker. They came out of an exhaustive
search over every single-field mutation of the baseline -- null, null-with-reason, deletion,
boolean flip, an order of magnitude either way -- keeping only those that stay schema-valid and
add findings of exactly one rule. Guessing which edit breaks C3 produces a corpus that passes
for the wrong reason; enumerating the space proves reachability instead.

Run: ``python examples/negative/build_corpus.py``
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
CORPUS_ROOT = REPO_ROOT / "examples" / "negative"
BASELINE = CORPUS_ROOT / "baseline.json"

#: Sentinel for an edit that removes a key rather than changing its value. The two are
#: different claims -- an absent key reads as "not applicable", a null as "unknown" -- and a
#: corpus about exactly that distinction cannot express one of them with the other.
DELETE = "__DELETE__"

#: The reason attached where a case nulls a field honestly. Several cases are meant to be
#: honest: they satisfy C1 and still fail, which is the whole point of those entries.
UNMEASURED = "(U) this fact was not captured by the harness that produced the run"


def _split(path: str) -> list[Any]:
    """Dotted path to a key list, with all-digit components read as list indices."""
    return [int(p) if p.lstrip("-").isdigit() else p for p in path.split(".")]


def edit(report: dict, path: str, value: Any) -> None:
    """Apply one edit in place. ``value`` may be :data:`DELETE` to remove the key entirely."""
    steps = _split(path)
    node = report
    for step in steps[:-1]:
        node = node[step]
    if value == DELETE:
        del node[steps[-1]]
    else:
        node[steps[-1]] = value


# (case_id, rule, edits, headline). ``edits`` is a list because nulling a field honestly takes
# two -- the value and its _u_reason sibling -- and a case that nulled the value alone would
# trip C1 as well and stop demonstrating the rule it is named for. The first edit's path is
# what the tests require the finding to name, so a mutation that trips the right rule in the
# wrong place fails there rather than shipping as a demonstration of something it does not show.
CASES: list[tuple[str, str, list[tuple[str, Any]], str]] = [
    (
        "c1",
        "C1",
        [("hardware.node_exclusivity", None)],
        "A null with nothing to say for itself",
    ),
    (
        "c2",
        "C2",
        [("model.weight_bytes_tag", DELETE)],
        "A number with no provenance",
    ),
    (
        "c3",
        "C3",
        [("serving.gpu_count", 8)],
        "A topology that does not multiply out",
    ),
    (
        "c4",
        "C4",
        [
            ("scaling.2.context_tokens", None),
            ("scaling.2.context_tokens_u_reason", UNMEASURED),
        ],
        "A throughput figure with no context length",
    ),
    (
        "c5",
        "C5",
        [
            ("capacity_tiers.measured.binding_constraint", None),
            ("capacity_tiers.measured.binding_constraint_u_reason", UNMEASURED),
        ],
        "A capacity figure that does not say which floor bound it",
    ),
    (
        "c6",
        "C6",
        [("capacity_tiers.measured.max_concurrent_users", 1.6384)],
        "A measured tier below the sustainable tier derived from it",
    ),
    (
        "c7",
        "C7",
        [("run.slo_gates.declared_before_run", False)],
        "SLO gates chosen after the results were in",
    ),
    (
        "c8",
        "C8",
        [
            ("reproduction.raw_records_path", None),
            ("reproduction.raw_records_path_u_reason", UNMEASURED),
        ],
        "A report whose per-request records were never published",
    ),
]


README = """# {headline}

**Rule broken:** {rule}
**The one edit:** {change}

{explanation}

## Reproduce

```bash
ascep conformance examples/negative/{case_id}/report.json
```

Every other byte of this report is identical to `examples/negative/baseline.json`, which grades
`conforming` with no findings. Diff the two to see the edit on its own:

```bash
diff <(jq -S . examples/negative/baseline.json) \\
     <(jq -S . examples/negative/{case_id}/report.json)
```
"""


def _describe(edits: list[tuple[str, Any]]) -> str:
    """Render the edit list the way the README states it, one clause per changed field."""
    parts = []
    for path, value in edits:
        if value == DELETE:
            parts.append(f"`{path}` is deleted")
        elif value is None:
            parts.append(f"`{path}` is set to `null`")
        elif path.endswith("_u_reason"):
            parts.append(f"`{path}` is added")
        else:
            parts.append(f"`{path}` is set to `{json.dumps(value)}`")
    return ", ".join(parts)


def main() -> None:
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    for case_id, rule, edits, headline in CASES:
        report = copy.deepcopy(baseline)
        for path, value in edits:
            edit(report, path, value)
        out_dir = CORPUS_ROOT / case_id
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

        # The explanation is prose and is edited by hand; the builder rewrites the mechanical
        # frame around it. Regenerating a case must not silently discard what someone wrote
        # about why the rule exists.
        body = out_dir / "explanation.md"
        explanation = body.read_text(encoding="utf-8").strip() if body.exists() else "TODO"
        (out_dir / "README.md").write_text(
            README.format(
                headline=headline,
                rule=rule,
                change=_describe(edits),
                explanation=explanation,
                case_id=case_id,
            ),
            encoding="utf-8",
        )
        print(f"wrote {out_dir.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
