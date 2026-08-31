"""The committed example report must be generated, valid, and honest.

``test_example_reproduces.py`` checks the arithmetic — that every (I) follows from its (M).
This file checks the *artifact*: that ``report.json`` is what ``build_report.py`` actually
emits, that it satisfies the schema, and that rule C1 holds in both directions.

The C1 walk here is deliberately generic rather than example-specific. It is the first piece
of the ``ascep conformance`` checker, and it is written so that any report added under
``examples/`` is picked up with no registration.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib

import pytest

from ascep.validation import validate

ROOT = pathlib.Path(__file__).parent.parent
REPORTS = sorted((ROOT / "examples").glob("*/report.json"))

# Keys that name a thing rather than assert a measurement. A null here is a missing label,
# which the schema already rejects; it is not an unmeasured quantity needing a (U) entry.
STRUCTURAL = {"conformance_note", "efficiency_over_one_explanation"}


def _load(path: pathlib.Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _build(example_dir: pathlib.Path) -> dict:
    spec = importlib.util.spec_from_file_location(
        f"build_{example_dir.name}", example_dir / "build_report.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.build()


def _nulls(node, path="") -> list[tuple[str, dict]]:
    """Every null-valued key, as (dotted path, its parent object)."""
    found = []
    if isinstance(node, dict):
        for k, v in node.items():
            p = f"{path}.{k}".lstrip(".")
            if v is None and k not in STRUCTURAL:
                found.append((p, node))
            found.extend(_nulls(v, p))
    elif isinstance(node, list):
        for i, v in enumerate(node):
            found.extend(_nulls(v, f"{path}[{i}]"))
    return found


assert REPORTS, "no example reports found; examples/*/report.json is the point of examples/"


@pytest.fixture(params=REPORTS, ids=lambda p: p.parent.name)
def report_path(request):
    return request.param


def test_report_validates(report_path):
    errors = validate("capacity-report", _load(report_path))
    assert not errors, "\n".join(errors)


def test_report_is_regenerable(report_path):
    """The committed file must equal what its builder emits.

    Without this, a report can be hand-edited into agreeing with a conclusion it no longer
    supports — the single easiest way to launder a number through this protocol.
    """
    assert _load(report_path) == _build(report_path.parent), (
        f"{report_path.parent.name}/report.json is stale or hand-edited. "
        f"Run: python examples/{report_path.parent.name}/build_report.py"
    )


def test_every_null_is_justified(report_path):
    """C1: unknown values are recorded as null with a (U) entry — never guessed, never omitted.

    A null is justified by a sibling ``<field>_u_reason`` or by a matching entry in the
    ``unmeasured_assumptions`` register. Reports that null a field and say nothing are the
    failure this rule exists to catch.
    """
    report = _load(report_path)
    registered = {e["field"] for e in report["unmeasured_assumptions"]}
    unjustified = [
        path
        for path, parent in _nulls(report)
        if not parent.get(f"{path.rsplit('.', 1)[-1]}_u_reason")
        and path not in registered
        and path.split(".", 1)[-1] not in registered
    ]
    assert not unjustified, "null with no (U) justification:\n  " + "\n  ".join(unjustified)


def test_u_reasons_are_tagged_and_not_stale(report_path):
    """Every justification carries the (U) tag (C2) and justifies a field that is really null.

    Deliberately *not* checking that reasons are long. "(U) not recorded" is a complete and
    honest answer to why a field is missing; the blast radius belongs in
    ``unmeasured_assumptions``, which has dedicated ``impact_if_wrong`` and ``cost_to_measure``
    fields for it. A length threshold here would only reward padding, and a padded reason
    reads as diligence while saying less than a short true one.

    The stale check is the one that matters: a reason left behind after its field was measured
    tells a reviewer to discount a number that is actually solid.
    """
    report = _load(report_path)

    def walk(node, path=""):
        bad = []
        if isinstance(node, dict):
            for k, v in node.items():
                p = f"{path}.{k}".lstrip(".")
                if k.endswith("_u_reason"):
                    if not isinstance(v, str) or not v.startswith("(U)"):
                        bad.append(f"{p}: does not carry the (U) tag")
                    elif v.strip() == "(U)":
                        bad.append(f"{p}: tag with no reason after it")
                    base = k[: -len("_u_reason")]
                    if base not in node:
                        bad.append(f"{p}: justifies {base}, which is not declared here")
                    elif node[base] is not None:
                        bad.append(f"{p}: justifies {base}, which is not null")
                bad.extend(walk(v, p))
        elif isinstance(node, list):
            for i, v in enumerate(node):
                bad.extend(walk(v, f"{path}[{i}]"))
        return bad

    problems = walk(report)
    assert not problems, "\n  ".join(problems)


def test_no_orphan_entries_in_the_unmeasured_register(report_path):
    """The other direction: the register must not list fields the report went on to measure."""
    report = _load(report_path)
    nulled = {p for p, _ in _nulls(report)}
    # Register paths are written against the source declaration (e.g. 'model.n_kv_heads');
    # accept any null whose dotted path ends the same way.
    orphans = [
        e["field"]
        for e in report["unmeasured_assumptions"]
        if not any(p == e["field"] or p.endswith("." + e["field"].split(".")[-1]) for p in nulled)
    ]
    assert not orphans, f"registered as unmeasured but not null in the report: {orphans}"


def test_partial_reports_say_why(report_path):
    report = _load(report_path)
    if report["conformance"] == "partial":
        assert report["unmeasured_assumptions"], (
            "a partial report with an empty unmeasured register is claiming to be partial "
            "for no stated reason"
        )


def test_no_tier_is_quoted_without_its_binding_constraint(report_path):
    """C5: a capacity figure without its binding constraint does not say what to buy."""
    for name, row in _load(report_path)["capacity_tiers"].items():
        if row["max_concurrent_users"] is not None:
            assert row["binding_constraint"], f"{name} states a capacity but no constraint"


def test_roofline_efficiency_is_not_silently_above_one(report_path):
    eff = _load(report_path)["roofline_comparison"]["roofline_efficiency"]
    if eff is not None:
        assert eff < 1.0, (
            f"roofline efficiency {eff} >= 1.0 means the measurement beat the hardware's "
            f"theoretical ceiling, which indicates a measurement error, not a fast server"
        )
