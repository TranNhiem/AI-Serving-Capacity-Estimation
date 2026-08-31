"""Completeness and consistency grader for the ASCEP conformance rules C1-C8.

The JSON Schemas (ascep.validation) validate structure and vocabulary; this module grades
what a schema cannot express: that every null is justified, that every number carries a
provenance tag, that topology and context bind every capacity figure, and that the four
capacity tiers tell a coherent story. :func:`check` aggregates every violation into a
:class:`Verdict` instead of raising on the first one, so a declaration can be fixed in a
single pass rather than a dozen.
"""

from __future__ import annotations

import numbers
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

_VALID_PROVENANCE = frozenset({"M", "I", "T", "U"})
_TIERS = ("theoretical", "measured", "sustainable", "recommended")
_TIER_NUMBER_FIELDS = (
    "max_concurrent_users",
    "max_tokens_per_s",
    "max_requests_per_s",
    "daily_requests",
)
_SCALING_NUMBER_FIELDS = ("output_tok_s", "scaling_efficiency")
_TOPOLOGY_FIELDS = ("tensor_parallel", "pipeline_parallel", "gpu_count")
_REPRODUCTION_FIELDS = (
    "run_configs_path",
    "raw_records_path",
    "engine_logs_path",
    "environment_capture_path",
    "container_digest",
)
_LEVEL_STRENGTH = {"non-conforming": 0, "partial": 1, "conforming": 2}
_U_REASON_SUFFIX = "_u_reason"

# The marker `ascep init` leaves in every unfilled string. Every other rule in this module
# tests `is None`, so before this existed a leftover "TODO" read as a declaration: a report
# with `reproduction.raw_records_path: "TODO"` passed C8 while pointing at nothing. Duplicated
# from `ascep.init.TODO` rather than imported, to keep this module free of any import that
# could later reach outside the stdlib; tests/test_init.py asserts the two agree.
_PLACEHOLDER = "TODO"


@dataclass(frozen=True)
class Finding:
    """One rule violation: which rule, how bad, where, and what to do about it."""

    rule: str
    severity: str
    path: str
    message: str


@dataclass(frozen=True)
class Verdict:
    """The computed conformance level plus every finding that produced it."""

    claimed: str
    level: str
    findings: tuple[Finding, ...]

    @property
    def errors(self) -> tuple[Finding, ...]:
        """Findings severe enough to cap the level below conforming."""
        return tuple(f for f in self.findings if f.severity == "error")

    @property
    def warnings(self) -> tuple[Finding, ...]:
        """Findings that downgrade to partial without invalidating the report."""
        return tuple(f for f in self.findings if f.severity == "warning")

    @property
    def overstated(self) -> bool:
        """Whether the submitted label claims more than the computed level."""
        return _LEVEL_STRENGTH.get(self.claimed, 0) > _LEVEL_STRENGTH[self.level]


def check(report: dict) -> Verdict:
    """Grade ``report`` against C1-C8 and return the computed verdict."""
    if not isinstance(report, dict):
        findings = (
            Finding(
                rule="C1",
                severity="error",
                path="report",
                message="Submit the report as a JSON object; it cannot be graded otherwise.",
            ),
        )
        return Verdict(claimed="", level="non-conforming", findings=findings)
    findings: list[Finding] = []
    _check_c1(report, findings)
    _check_c2(report, findings)
    _check_c3(report, findings)
    _check_c4(report, findings)
    _check_c5(report, findings)
    _check_c6(report, findings)
    _check_c7(report, findings)
    _check_c8(report, findings)
    ordered = tuple(sorted(findings, key=lambda f: (f.rule, f.path)))
    claimed = report.get("conformance")
    return Verdict(
        claimed=claimed if isinstance(claimed, str) else "",
        level=_compute_level(ordered),
        findings=ordered,
    )


def _compute_level(findings: tuple[Finding, ...]) -> str:
    errors = [f for f in findings if f.severity == "error"]
    if any(f.rule in ("C1", "C2", "C3", "C4", "C5") for f in errors):
        return "non-conforming"
    if errors or findings:
        return "partial"
    return "conforming"


def _is_number(value: Any) -> bool:
    # bool is an int subclass, but a gate flag is not a numeric claim and must be excluded.
    return isinstance(value, numbers.Real) and not isinstance(value, bool)


def _tokens(value: Any) -> str:
    """Render a context length the way a reader writes one: ``2,000``, not ``2000.0``.

    JSON has no integer type distinct from float, so a context declared as 2000 arrives as
    2000.0 and lands verbatim in a finding message. These strings are the protocol's public
    face; a stray ``.0`` reads like a tool that does not know what it is measuring.
    """
    if not _is_number(value):
        return str(value)
    return f"{int(value):,}" if float(value).is_integer() else f"{value:,.2f}"


def _get(node: Any, key: str) -> Any:
    return node.get(key) if isinstance(node, dict) else None


def _has_number(row: dict, fields: tuple[str, ...]) -> bool:
    return any(_is_number(row.get(field)) for field in fields)


def _tier_rows(report: dict) -> Iterator[tuple[str, dict]]:
    tiers = _get(report, "capacity_tiers")
    if isinstance(tiers, dict):
        for name, row in tiers.items():
            if isinstance(row, dict):
                yield name, row


def _scaling_rows(report: dict) -> Iterator[tuple[int, dict]]:
    scaling = _get(report, "scaling")
    if isinstance(scaling, list):
        for index, row in enumerate(scaling):
            if isinstance(row, dict):
                yield index, row


# ----------------------------------------------------------------------- C1 completeness


def _check_c1(report: dict, findings: list[Finding]) -> None:
    assumption_fields = set()
    assumptions = _get(report, "unmeasured_assumptions")
    if isinstance(assumptions, list):
        for entry in assumptions:
            if isinstance(entry, dict) and isinstance(entry.get("field"), str):
                assumption_fields.add(entry["field"])
    _walk_nulls(report, "", assumption_fields, findings)
    _walk_placeholders(report, "", findings)
    _check_schema(report, findings)


def _walk_placeholders(node: Any, path: str, findings: list[Finding]) -> None:
    """Flag scaffolding text that was never replaced.

    A null is an honest unknown and C1 only asks that it be justified. A leftover ``TODO`` is
    something else: it occupies the slot, so every ``is None`` test in this module reads it as
    a declaration, and the report claims a value it does not have. Empty strings are the same
    failure typed differently.

    Deliberately not a general junk detector -- "n/a" and "unknown" are not caught, because
    guessing at what a human meant is a losing game. This catches the one string the toolchain
    itself produces, which is the one a contributor is most likely to leave behind.
    """
    if isinstance(node, list):
        for index, item in enumerate(node):
            _walk_placeholders(item, f"{path}.{index}", findings)
        return
    if isinstance(node, dict):
        for key, value in node.items():
            _walk_placeholders(value, f"{path}.{key}" if path else key, findings)
        return
    if not isinstance(node, str):
        return

    if _PLACEHOLDER in node:
        findings.append(
            Finding(
                rule="C1",
                severity="error",
                path=path,
                message=(
                    f"Replace the '{_PLACEHOLDER}' left by `ascep init`: fill this in, or set "
                    "it to null with a (U) reason. Scaffolding text occupies the slot, so "
                    "every other check reads it as a value that was declared."
                ),
            )
        )
    elif not node.strip():
        findings.append(
            Finding(
                rule="C1",
                severity="error",
                path=path,
                message=(
                    "An empty string is not a declaration. Give a value, or null with a "
                    "(U) reason saying why it is unknown."
                ),
            )
        )


def _walk_nulls(node: Any, path: str, assumption_fields: set, findings: list[Finding]) -> None:
    if isinstance(node, list):
        for index, item in enumerate(node):
            _walk_nulls(item, f"{path}.{index}", assumption_fields, findings)
        return
    if not isinstance(node, dict):
        return
    for key, value in node.items():
        child = f"{path}.{key}" if path else key
        if key.endswith(_U_REASON_SUFFIX):
            base = key[: -len(_U_REASON_SUFFIX)]
            if base in node and node[base] is not None:
                findings.append(
                    Finding(
                        rule="C1",
                        severity="error",
                        path=child,
                        message=(
                            f"Remove this stale justification: '{base}' is not null, so "
                            "the (U) note tells reviewers to discount a solid number."
                        ),
                    )
                )
            # Justification strings are prose about a claim, not claims themselves.
            continue
        if value is None:
            if not _null_is_justified(node, key, child, assumption_fields):
                findings.append(
                    Finding(
                        rule="C1",
                        severity="error",
                        path=child,
                        message=(
                            f"Justify this null with a sibling '{key}_u_reason' starting "
                            "with '(U)' or an unmeasured_assumptions entry; unknown values "
                            "must be recorded, never silently omitted."
                        ),
                    )
                )
        else:
            _walk_nulls(value, child, assumption_fields, findings)


def _null_is_justified(node: dict, key: str, path: str, assumption_fields: set) -> bool:
    reason = node.get(key + _U_REASON_SUFFIX)
    if isinstance(reason, str) and reason.startswith("(U)"):
        return True
    # Entries may name the full dotted path or, for rows inside arrays, the leaf field.
    leaf = path.rsplit(".", 1)[-1]
    return any(field in (path, leaf) for field in assumption_fields)


def _check_schema(report: dict, findings: list[Finding]) -> None:
    try:
        from ascep import validation
    except ImportError:
        _warn_schema_skipped(findings)
        return
    try:
        messages = validation.validate("capacity-report", report)
    except (ImportError, FileNotFoundError):
        # jsonschema is imported lazily inside ascep.validation, and the schemas may be
        # missing from an unpacked checkout; the checker must still grade what it can.
        _warn_schema_skipped(findings)
        return
    for message in messages:
        head, _, _ = message.partition(": ")
        findings.append(
            Finding(
                rule="C1",
                severity="error",
                path=head if head.startswith("$") else "report",
                message=f"Fix the schema violation: {message}.",
            )
        )


def _warn_schema_skipped(findings: list[Finding]) -> None:
    findings.append(
        Finding(
            rule="C1",
            severity="warning",
            path="report",
            message=(
                "Install jsonschema and ship the schemas/ directory: structural "
                "validation was skipped, so this verdict covers C1-C8 semantics only."
            ),
        )
    )


# ---------------------------------------------------------------------- C2 provenance


def _check_c2(report: dict, findings: list[Finding]) -> None:
    _walk_provenance(report, "", findings)
    tag_hint = (
        "Tag these numbers with exactly one of (M), (I), (T) or (U); an untagged "
        "numeric claim is non-conforming."
    )
    for name, row in _tier_rows(report):
        if row.get("provenance") is None and _has_number(row, _TIER_NUMBER_FIELDS):
            findings.append(Finding("C2", "error", f"capacity_tiers.{name}.provenance", tag_hint))
    for index, row in _scaling_rows(report):
        if row.get("provenance") is None and _has_number(row, _SCALING_NUMBER_FIELDS):
            findings.append(Finding("C2", "error", f"scaling.{index}.provenance", tag_hint))


def _walk_provenance(node: Any, path: str, findings: list[Finding]) -> None:
    if isinstance(node, list):
        for index, item in enumerate(node):
            _walk_provenance(item, f"{path}.{index}", findings)
        return
    if not isinstance(node, dict):
        return
    for key, value in node.items():
        child = f"{path}.{key}" if path else key
        if key == "provenance":
            if value is not None and (not isinstance(value, str) or value not in _VALID_PROVENANCE):
                findings.append(
                    Finding(
                        rule="C2",
                        severity="error",
                        path=child,
                        message=(
                            f"Replace provenance {value!r} with exactly one of 'M', 'I', "
                            "'T' or 'U'; anything else is not a claim a reviewer can act on."
                        ),
                    )
                )
        else:
            _walk_provenance(value, child, findings)


# ----------------------------------------------------------------------- C3 topology


def _check_c3(report: dict, findings: list[Finding]) -> None:
    serving = _get(report, "serving")
    if isinstance(serving, dict):
        for field in _TOPOLOGY_FIELDS:
            if serving.get(field) is None:
                findings.append(
                    Finding(
                        rule="C3",
                        severity="error",
                        path=f"serving.{field}",
                        message=(
                            f"Record serving.{field}: every capacity figure must be bound "
                            "to the topology it was measured at, or be rebuilt when the "
                            "topology changes."
                        ),
                    )
                )
        tp = serving.get("tensor_parallel")
        pp = serving.get("pipeline_parallel")
        gpus = serving.get("gpu_count")
        if all(_is_number(v) for v in (tp, pp, gpus)) and gpus != tp * pp:
            findings.append(
                Finding(
                    rule="C3",
                    severity="error",
                    path="serving.gpu_count",
                    message=(
                        f"Reconcile gpu_count ({gpus}) with tensor_parallel * "
                        f"pipeline_parallel ({tp} * {pp}); the mismatch is a typo that "
                        "silently rescales every per-GPU figure."
                    ),
                )
            )
    for index, row in _scaling_rows(report):
        for field in _TOPOLOGY_FIELDS:
            if row.get(field) is None:
                findings.append(
                    Finding(
                        rule="C3",
                        severity="error",
                        path=f"scaling.{index}.{field}",
                        message=(
                            f"Record {field} for this scaling point; a scaling ratio "
                            "without its topology cannot be compared against anything."
                        ),
                    )
                )
    for name, row in _tier_rows(report):
        if row.get("max_concurrent_users") is not None and row.get("n_gpus") is None:
            findings.append(
                Finding(
                    rule="C3",
                    severity="error",
                    path=f"capacity_tiers.{name}.n_gpus",
                    message=(
                        "Record n_gpus for this tier; a per-tier capacity figure without "
                        "its GPU count cannot be priced."
                    ),
                )
            )


# ------------------------------------------------------------------------ C4 context


def _check_c4(report: dict, findings: list[Finding]) -> None:
    run = _get(report, "run")
    results = run.get("results") if isinstance(run, dict) else None
    if isinstance(results, list):
        contexts = set()
        for index, row in enumerate(results):
            if not isinstance(row, dict):
                continue
            context = row.get("context_tokens")
            if context is None and row.get("input_tokens") is None:
                findings.append(
                    Finding(
                        rule="C4",
                        severity="error",
                        path=f"run.results.{index}",
                        message=(
                            "Record context_tokens (or at least input/output tokens) for "
                            "this point; a throughput figure without its context length "
                            "cannot be compared or projected."
                        ),
                    )
                )
            if _is_number(context):
                contexts.add(float(context))
        if len(contexts) < 3:
            findings.append(
                Finding(
                    rule="C4",
                    severity="warning",
                    path="run.results",
                    message=(
                        f"Only {len(contexts)} distinct context length(s) measured; the "
                        "protocol wants a curve of at least three, so label this a "
                        "single-point measurement."
                    ),
                )
            )
    for index, row in _scaling_rows(report):
        if row.get("context_tokens") is None:
            findings.append(
                Finding(
                    rule="C4",
                    severity="warning",
                    path=f"scaling.{index}.context_tokens",
                    message=(
                        "State the context length of this scaling point; throughput "
                        "ratios are only comparable at equal context."
                    ),
                )
            )


# ----------------------------------------------------------------------- C5 constraint


def _check_c5(report: dict, findings: list[Finding]) -> None:
    for name, row in _tier_rows(report):
        if row.get("max_concurrent_users") is not None and row.get("binding_constraint") is None:
            findings.append(
                Finding(
                    rule="C5",
                    severity="error",
                    path=f"capacity_tiers.{name}.binding_constraint",
                    message=(
                        "Name the floor that binds ('weights', 'kv', 'throughput' or "
                        "'slo'); a capacity number without its constraint does not say "
                        "what to buy."
                    ),
                )
            )
    sizing = _get(report, "sizing_result")
    if (
        isinstance(sizing, dict)
        and sizing.get("gpus_required") is not None
        and sizing.get("binding_constraint") is None
    ):
        findings.append(
            Finding(
                rule="C5",
                severity="error",
                path="sizing_result.binding_constraint",
                message=(
                    "Name the constraint behind gpus_required; otherwise the number "
                    "cannot be acted on."
                ),
            )
        )


# --------------------------------------------------------------------------- C6 tiers


def _check_c6(report: dict, findings: list[Finding]) -> None:
    tiers = _get(report, "capacity_tiers")
    if not isinstance(tiers, dict):
        return
    users: dict[str, float] = {}
    for name in _TIERS:
        row = tiers.get(name)
        if not isinstance(row, dict):
            findings.append(
                Finding(
                    rule="C6",
                    severity="error",
                    path=f"capacity_tiers.{name}",
                    message=(
                        f"Add a '{name}' tier, even if every figure in it is null with a "
                        "(U) reason; reporting fewer than four tiers invites readers to "
                        "assume the most favourable one."
                    ),
                )
            )
            continue
        value = row.get("max_concurrent_users")
        if value is None:
            findings.append(
                Finding(
                    rule="C6",
                    severity="warning",
                    path=f"capacity_tiers.{name}.max_concurrent_users",
                    message=(
                        "Stating this tier as unmeasured is legitimate; keep its (U) "
                        "reason and close the gap when the measurement exists."
                    ),
                )
            )
        elif _is_number(value):
            users[name] = float(value)
    for low, high in (
        ("recommended", "sustainable"),
        ("sustainable", "measured"),
        ("measured", "theoretical"),
    ):
        if low in users and high in users and users[low] > users[high]:
            findings.append(
                Finding(
                    rule="C6",
                    severity="error",
                    path=f"capacity_tiers.{high}.max_concurrent_users",
                    message=(
                        f"Fix the tier ordering: {high} ({users[high]}) must be >= {low} "
                        f"({users[low]}); later tiers only ever remove capacity."
                    ),
                )
            )
    roofline = _get(report, "roofline_comparison")
    if isinstance(roofline, dict):
        efficiency = roofline.get("roofline_efficiency")
        if _is_number(efficiency) and efficiency >= 1.0:
            findings.append(
                Finding(
                    rule="C6",
                    severity="error",
                    path="roofline_comparison.roofline_efficiency",
                    message=(
                        "Re-measure: a roofline efficiency at or above 1.0 means the "
                        "measurement beat the hardware ceiling, which indicates a "
                        "measurement error rather than a fast server."
                    ),
                )
            )


# -------------------------------------------------------------------------- C7 gates


def _check_c7(report: dict, findings: list[Finding]) -> None:
    run = _get(report, "run")
    if not isinstance(run, dict):
        return
    gates = run.get("slo_gates")
    if isinstance(gates, dict):
        if gates.get("declared_before_run") is not True:
            findings.append(
                Finding(
                    rule="C7",
                    severity="error",
                    path="run.slo_gates.declared_before_run",
                    message=(
                        "Set declared_before_run to true and fix gate thresholds in the "
                        "run config before measuring; thresholds chosen after seeing "
                        "results are not gates."
                    ),
                )
            )
        thresholds = [
            value
            for key, value in gates.items()
            if key != "declared_before_run" and not key.endswith(_U_REASON_SUFFIX)
        ]
        if all(value is None for value in thresholds):
            findings.append(
                Finding(
                    rule="C7",
                    severity="warning",
                    path="run.slo_gates",
                    message=(
                        "Declare at least one gate threshold; a run with no gates at all "
                        "cannot produce a meaningful sustainable tier."
                    ),
                )
            )
    results = run.get("results")
    failures = [
        row
        for row in (results if isinstance(results, list) else [])
        if isinstance(row, dict) and row.get("slo_pass") is False
    ]
    if not failures:
        return
    tiers = _get(report, "capacity_tiers")
    measured = _get(_get(tiers, "measured"), "max_concurrent_users")
    sustainable = _get(_get(tiers, "sustainable"), "max_concurrent_users")
    if not (_is_number(measured) and _is_number(sustainable) and measured == sustainable):
        return

    # A gate failure only disqualifies the sustainable tier where it lies INSIDE the envelope
    # the capacity claim covers (chapter 5.5). A TTFT failure at 8k context does not invalidate
    # a figure computed at 2k. Grading it as an error anyway would reject honest reports and
    # teach people to ignore the checker, which costs more than the rule buys.
    envelope = _get(_get(report, "workload"), "avg_context_tokens")
    inside = [
        row
        for row in failures
        if not _is_number(envelope)
        or not _is_number(row.get("context_tokens"))
        or row["context_tokens"] <= envelope
    ]
    if inside:
        findings.append(
            Finding(
                rule="C7",
                severity="error",
                path="capacity_tiers.sustainable.max_concurrent_users",
                message=(
                    "Exclude the failing operating point: a results row at or below the "
                    "workload's average context failed its SLO gate, yet the sustainable "
                    "tier still equals the measured tier."
                ),
            )
        )
        return
    contexts = ", ".join(_tokens(row["context_tokens"]) for row in failures)
    findings.append(
        Finding(
            rule="C7",
            severity="warning",
            path="capacity_tiers.sustainable.max_concurrent_users",
            message=(
                f"State the context envelope with this figure: the gate fails at {contexts} "
                f"tokens, above the workload's {_tokens(envelope)}, so the sustainable tier is "
                "valid only at or below that context and must not be quoted as a general number."
            ),
        )
    )


# ---------------------------------------------------------------------- C8 reproduction


def _check_c8(report: dict, findings: list[Finding]) -> None:
    reproduction = _get(report, "reproduction")
    if not isinstance(reproduction, dict):
        findings.append(
            Finding(
                rule="C8",
                severity="warning",
                path="reproduction",
                message=(
                    "Publish the reproduction bundle (run configs, raw records, engine "
                    "logs, environment capture, container digest); without it the report "
                    "is capped at partial."
                ),
            )
        )
        return
    for field in _REPRODUCTION_FIELDS:
        if reproduction.get(field) is None:
            findings.append(
                Finding(
                    rule="C8",
                    severity="warning",
                    path=f"reproduction.{field}",
                    message=(
                        f"Publish reproduction.{field} with the report; without it the "
                        "figures cannot be reproduced and the report is capped at partial."
                    ),
                )
            )
