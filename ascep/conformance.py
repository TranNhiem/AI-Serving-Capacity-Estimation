"""Completeness and consistency grader for the ASCEP conformance rules C1-C8.

The JSON Schemas (ascep.validation) validate structure and vocabulary; this module grades
what a schema cannot express: that every null is justified, that every number carries a
provenance tag, that topology and context bind every capacity figure, and that the four
capacity tiers tell a coherent story. :func:`check` aggregates every violation into a
:class:`Verdict` instead of raising on the first one, so a declaration can be fixed in a
single pass rather than a dozen.
"""

from __future__ import annotations

import math
import numbers
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from ascep.capacity import Provenance, Tier

# Derived, never retyped. A grader holding its own copy of the vocabulary it grades against
# will pass a report that the schemas reject, or reject one they accept, the first time the
# two drift — and the drift is invisible because each file reads correctly on its own.
_VALID_PROVENANCE = frozenset(p.value for p in Provenance)
_TIERS = tuple(t.value for t in Tier)
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

# The date-time placeholder from the same generator. A `format: date-time` field cannot hold
# "TODO" without failing validation for a reason that is the tool's doing, so `ascep init`
# writes the epoch there instead -- which parses, validates, and survives `grep TODO`. Without
# this line an untouched skeleton claims a real generation timestamp, which is the one
# fabricated value the placeholder rule exists to make impossible.
_EPOCH_PLACEHOLDER = "1970-01-01T00:00:00Z"


def _is_placeholder_text(value: str) -> bool:
    """Whether a string is generated scaffolding rather than something a person wrote.

    Case-folded and stripped, because "todo" and " TODO " are the same omission typed less
    carefully. Anchored to the whole value, and to the one prefix `ascep init` writes into a
    `_u_reason`, so a real path or a real sentence that happens to contain the word is left
    alone -- an author who has to fight this rule stops reading it.
    """
    text = value.strip()
    return (
        text.upper() == _PLACEHOLDER
        or text == _EPOCH_PLACEHOLDER
        or text.startswith(f"(U) {_PLACEHOLDER}:")
    )


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
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        return False
    # NaN and the infinities are not numeric claims either, and admitting them disables this
    # whole module: every comparison against NaN is False, so the C6 ordering, the C6 roofline
    # ceiling, the C7 gate check and the C4 curve count all pass on a report that declares
    # nothing. `json.load` parses the bare tokens NaN and Infinity by default, so this is a
    # file a real toolchain can emit, not a hand-crafted attack. Rejecting them here makes them
    # invisible to those rules; `_walk_placeholders` then reports them under C1 by name, so
    # they surface as the declaration failure they are rather than vanishing.
    return math.isfinite(float(value))


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


def _input_modalities(report: dict) -> list:
    """The model layer's declared input modalities, normalised to a list.

    Section 9's obligations gate on image or video appearing in this one list, and two
    readers deriving it independently would drift on exactly the malformed cases --
    absent key, string instead of list -- that a grader must survive without raising.
    """
    modalities = _get(_get(report, "model"), "input_modalities")
    return modalities if isinstance(modalities, list) else []


def _report_declares_media(report: dict) -> bool:
    """Whether section 9's modality gate fires: the model declares image or video input."""
    modalities = _input_modalities(report)
    return "image" in modalities or "video" in modalities


# ----------------------------------------------------------------------- C1 completeness


def _check_c1(report: dict, findings: list[Finding]) -> None:
    assumption_fields = set()
    assumptions = _get(report, "unmeasured_assumptions")
    if isinstance(assumptions, list):
        for entry in assumptions:
            if isinstance(entry, dict) and isinstance(entry.get("field"), str):
                assumption_fields.add(entry["field"])
    _walk_nulls(report, "", assumption_fields, findings)
    _check_c1_cpu_cores_note(report, findings)
    _walk_placeholders(report, "", findings)
    _check_schema(report, findings)


def _walk_placeholders(node: Any, path: str, findings: list[Finding]) -> None:
    """Flag scaffolding text that was never replaced.

    A null is an honest unknown and C1 only asks that it be justified. A leftover ``TODO`` is
    something else: it occupies the slot, so every ``is None`` test in this module reads it as
    a declaration, and the report claims a value it does not have. Empty strings are the same
    failure typed differently, and so are NaN and the infinities.

    Deliberately not a general junk detector -- "n/a" and "unknown" are not caught, because
    guessing at what a human meant is a losing game. This catches the strings the toolchain
    itself produces, which are the ones a contributor is most likely to leave behind, plus the
    non-finite floats, which need no guessing: they have a definite meaning and it is not a
    measurement.

    Matching is exact rather than by substring, for the same reason. A field reading
    ``s3://logs/TODO-migration/2026.gz`` is a real path that happens to contain the word, and
    rejecting it would make this rule something authors work around instead of with.
    """
    if isinstance(node, list):
        for index, item in enumerate(node):
            _walk_placeholders(item, f"{path}.{index}", findings)
        return
    if isinstance(node, dict):
        for key, value in node.items():
            # A stale justification beside a filled-in field is already reported by the null
            # walk, and with the opposite instruction: delete it, not fill it in. Two C1 errors
            # at one path telling the author contradictory things is worse than one.
            if key.endswith(_U_REASON_SUFFIX):
                if node.get(key[: -len(_U_REASON_SUFFIX)]) is not None:
                    continue
            _walk_placeholders(value, f"{path}.{key}" if path else key, findings)
        return
    if _is_non_finite(node):
        findings.append(
            Finding(
                rule="C1",
                severity="error",
                path=path,
                message=(
                    f"Replace {node!r} with a number or with null plus a (U) reason. A "
                    "non-finite value is what a division by zero or an empty average leaves "
                    "behind, and every threshold comparison against it silently succeeds."
                ),
            )
        )
        return
    if not isinstance(node, str):
        return

    if _is_placeholder_text(node):
        findings.append(
            Finding(
                rule="C1",
                severity="error",
                path=path,
                message=(
                    f"Replace the scaffolding `ascep init` left here ({node.strip()!r}): fill "
                    "this in, or set it to null with a (U) reason. A placeholder occupies the "
                    "slot, so every other check reads it as a value that was declared."
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


def _is_assumptions_entry(path: str) -> bool:
    """Whether ``path`` addresses an entry of the top-level ``unmeasured_assumptions``.

    Scoped to exactly ``unmeasured_assumptions.<index>`` so the C1 exemption below cannot
    leak: a key named ``value_used`` inside a nested array, a copied structure at another
    depth, or anywhere else in a report is graded like any other null.
    """
    head, _, index = path.partition(".")
    return head == "unmeasured_assumptions" and index.isdigit()


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
            # A null value_used in a section-7 entry is the honest state, not a gap: it says
            # no substitute was plugged in, and the entry around it already names the field,
            # the blast radius and the closure cost -- more than a (U) sentence could. Both
            # remedies C1 prescribes are worthless here. The sibling key is schema-illegal,
            # and the section-7 remedy only works spelled as the bare leaf 'value_used',
            # which makes the register declare itself an unmeasured assumption and clears
            # its own null on the way past. Either way the block a reviewer reads first
            # gains an entry recording nothing about the deployment -- and the third option,
            # the one needing no ceremony at all, is to invent a value.
            if key == "value_used" and _is_assumptions_entry(path):
                continue
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


def _is_non_finite(value: Any) -> bool:
    """A real number that is NaN or an infinity -- the one numeric shape that is not a value."""
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        return False
    return not math.isfinite(float(value))


#: Characters a bare "(U)" tag can be padded with while still saying nothing. A reason has to
#: carry words; the tag on its own is the author agreeing with the rule rather than answering it.
_TAG_PADDING = " \t.:;-—,"


def _null_is_justified(node: dict, key: str, path: str, assumption_fields: set) -> bool:
    # A 'notes' entry beside the null is deliberately not consulted: notes explain why a
    # declared value is what it is. Letting one satisfy C1 would give a missing value a way
    # to look justified while never saying the (U) that marks it as unknown.
    reason = node.get(key + _U_REASON_SUFFIX)
    # The tag alone is not the justification, the sentence after it is. Accepting a bare "(U)"
    # turned C1 into a formality: pasting four characters beside every null cleared the whole
    # rule, and the reviewer who needed to know WHY a figure is missing got nothing.
    if isinstance(reason, str) and reason.startswith("(U)") and reason[3:].strip(_TAG_PADDING):
        return True
    # Entries may name the full dotted path or, for rows inside arrays, the leaf field.
    leaf = path.rsplit(".", 1)[-1]
    return any(field in (path, leaf) for field in assumption_fields)


def _check_c1_cpu_cores_note(report: dict, findings: list[Finding]) -> None:
    """Require a media report to say what its declared cpu_cores actually is.

    The null walk cannot carry this obligation, because the value is never null in
    practice: under a media workload the host CPU decodes and patchifies every image, so
    cpu_cores is a capacity input, and a bare integer is the compliant-looking shape of the
    omission. Without a note saying whether the number is the machine's core count or an
    allocation's, the report publishes a ceiling no reader can attribute to the GPU or the
    host.
    """
    if not _report_declares_media(report):
        return
    hardware = _get(report, "hardware")
    if not isinstance(hardware, dict) or hardware.get("cpu_cores") is None:
        # A null cpu_cores is the null walk's business, which already demands its (U)
        # reason; firing here too would report one omission as two findings.
        return
    notes = hardware.get("notes")
    note = notes.get("cpu_cores") if isinstance(notes, dict) else None
    if isinstance(note, str) and note.strip():
        return
    findings.append(
        Finding(
            rule="C1",
            severity="error",
            path="hardware.cpu_cores",
            message=(
                "Add a note for hardware.cpu_cores: put a 'cpu_cores' key in the hardware "
                "layer's 'notes' object saying whether this is the machine's core count or "
                "an allocation's. This report declares image or video input, so the host CPU "
                "decodes and patchifies every image and cpu_cores is a capacity input, not "
                "an inventory fact; without the note the report has published a CPU-bound "
                "ceiling nobody can attribute."
            ),
        )
    )


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


#: Numbers whose provenance travels in a sibling ``*_tag`` rather than a ``provenance`` key,
#: as (layer, number, tag). The schemas require the number and leave the tag optional, so
#: nothing but this table stops the three highest-lever figures in a report -- the weight
#: bytes the whole weights floor rests on, the user count the demand is scaled from, and the
#: mean context the KV floor divides by -- from being published with no provenance at all.
_TAGGED_NUMBERS = (
    ("model", "weight_bytes_on_disk", "weight_bytes_tag"),
    ("workload", "daily_active_users", "daily_users_tag"),
    ("workload", "avg_context_tokens", "avg_context_tokens_tag"),
)


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
    for layer, number, tag in _TAGGED_NUMBERS:
        section = _get(report, layer)
        if not isinstance(section, dict) or section.get(number) is None:
            # A null number is C1's business: it needs a reason, not a provenance tag. Asking
            # for both would make every honestly-unmeasured field fail two rules for one gap.
            continue
        if section.get(tag) is None:
            findings.append(
                Finding(
                    "C2",
                    "error",
                    f"{layer}.{tag}",
                    f"{layer}.{number} is published as a number with no provenance. {tag_hint}",
                )
            )


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
            # Stays a warning either way: a single point is a real limit on what the report can
            # claim, not a defect in it. The two messages differ because the actions differ --
            # one asks for a declaration, the other confirms the declaration was read and tells
            # the author what it costs them, so a labelled report stops looking like a to-do.
            declared = isinstance(run, dict) and run.get("single_point") is True
            message = (
                f"Only {len(contexts)} distinct context length(s) measured. Declared "
                "run.single_point, so this is honest — but the figures hold at that context "
                "only and MUST NOT be quoted as a curve or projected to another shape."
                if declared
                else (
                    f"Only {len(contexts)} distinct context length(s) measured; the protocol "
                    "wants a curve of at least three. Measure more, or set run.single_point "
                    "to true -- C4 requires the limit be stated, and an unlabelled point "
                    "reads as a curve."
                )
            )
            findings.append(
                Finding(rule="C4", severity="warning", path="run.results", message=message)
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
    _check_c4_media_and_reasoning(report, findings)


# ---------------------------------------------------------------- C4 media and reasoning


def _check_c4_media_and_reasoning(report: dict, findings: list[Finding]) -> None:
    """Grade the media-shape and reasoning-mode declarations section 9.8 adds to C4.

    Each rule keys off a field the schema makes required and non-nullable, so no check
    here can be skipped by declaring ignorance: a (U) reason satisfies C1 and changes
    nothing below. The isinstance guards are for the malformed report only -- C4 must
    not raise on input the schema check has already rejected, and a missing workload
    or run object simply means there is nothing for these checks to bind to.
    """
    model = _get(report, "model")
    serving = _get(report, "serving")
    workload = _get(report, "workload")
    run = _get(report, "run")
    if not isinstance(workload, dict):
        return
    modalities = _input_modalities(report)
    reasoning_modes = _get(model, "reasoning_modes")
    if not isinstance(reasoning_modes, list):
        reasoning_modes = []
    images = workload.get("images_per_request")
    videos = workload.get("videos_per_request")

    if "image" in modalities and images is None:
        findings.append(
            Finding(
                rule="C4",
                severity="error",
                path="workload.images_per_request",
                message=(
                    "Record images_per_request: this model declares image input, and "
                    "a vision model benchmarked without saying how much media each "
                    "request carried has an unknown KV floor. 0 (measured, no media) "
                    "and null (not reported) are different claims that size a "
                    "cluster differently."
                ),
            )
        )
    if "video" in modalities:
        if videos is None:
            findings.append(
                Finding(
                    rule="C4",
                    severity="error",
                    path="workload.videos_per_request",
                    message=(
                        "Record videos_per_request: this model declares video input, "
                        "and a vision model benchmarked without saying how much "
                        "media each request carried has an unknown KV floor. 0 "
                        "(measured, no media) and null (not reported) are different "
                        "claims that size a cluster differently."
                    ),
                )
            )
        elif (
            _is_number(videos) and videos > 0 and workload.get("video_seconds_per_request") is None
        ):
            findings.append(
                Finding(
                    rule="C4",
                    severity="error",
                    path="workload.video_seconds_per_request",
                    message=(
                        "Record video_seconds_per_request: a clip count without a "
                        "duration does not say how much media each request carried, "
                        "so the KV floor is unknown. 0 (measured, no media) and null "
                        "(not reported) are different claims that size a cluster "
                        "differently."
                    ),
                )
            )
    if "thinking" in reasoning_modes and workload.get("reasoning_mode") is None:
        findings.append(
            Finding(
                rule="C4",
                severity="error",
                path="workload.reasoning_mode",
                message=(
                    "Record reasoning_mode: this model declares a thinking branch, "
                    "which is two capacity profiles that differ by orders of "
                    "magnitude in output length, and a report that does not name the "
                    "one it measured cannot be read as either."
                ),
            )
        )
    reasoning_mode = workload.get("reasoning_mode")
    if reasoning_mode in ("thinking", "mixed"):
        if workload.get("max_output_tokens") is None:
            findings.append(
                Finding(
                    rule="C4",
                    severity="error",
                    path="workload.max_output_tokens",
                    message=(
                        "Record max_output_tokens: reasoning traces expand to fill "
                        "whatever budget they are given, so an output length without "
                        "its cap describes the harness as much as the model and is "
                        "not reproducible."
                    ),
                )
            )
        if isinstance(run, dict) and run.get("truncation_rate") is None:
            findings.append(
                Finding(
                    rule="C4",
                    severity="error",
                    path="run.truncation_rate",
                    message=(
                        "Record truncation_rate: averaging truncated and completed "
                        "requests into one throughput figure counts tokens no user "
                        "received."
                    ),
                )
            )
        if reasoning_mode == "mixed" and workload.get("reasoning_share") is None:
            findings.append(
                Finding(
                    rule="C4",
                    severity="error",
                    path="workload.reasoning_share",
                    message=(
                        "Record reasoning_share: a mixed workload reported as a "
                        "single undifferentiated run hides the orders-of-magnitude "
                        "spread between its two capacity profiles."
                    ),
                )
            )
    carries_media = (_is_number(images) and images > 0) or (_is_number(videos) and videos > 0)
    if carries_media and not isinstance(_get(serving, "media_preprocessing"), dict):
        findings.append(
            Finding(
                rule="C4",
                severity="warning",
                path="serving.media_preprocessing",
                message=(
                    "Publish media_preprocessing: the media token cost is decided by "
                    "the server's sampling rate, frame cap and pixel budget, so a "
                    "media throughput figure without them cannot be reproduced or "
                    "compared against another deployment."
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
    _check_digest(reproduction, findings)


def _check_digest(reproduction: dict, findings: list[Finding]) -> None:
    """Reject a container digest that cannot be one.

    The paths in this section are deliberately unchecked -- they name artifacts on a machine
    this tool cannot see, so "does it resolve" is not a question it can ask. A digest is the
    exception: it is a self-describing string, so ``sha256:0`` is not an unverifiable claim but
    a malformed one, and it is what a placeholder or a truncated copy-paste looks like. Getting
    it wrong defeats the whole bundle, because the digest is what pins the software the other
    four artifacts were produced by.
    """
    digest = reproduction.get("container_digest")
    if not isinstance(digest, str) or not digest.strip():
        return
    algorithm, _, hexpart = digest.partition(":")
    # Any registry algorithm is accepted; only the shape is checked. Hard-coding sha256 would
    # reject an honest sha512 digest, which is a worse error than the one being caught.
    if algorithm and len(hexpart) >= 32 and all(c in "0123456789abcdefABCDEF" for c in hexpart):
        return
    findings.append(
        Finding(
            rule="C8",
            severity="warning",
            path="reproduction.container_digest",
            message=(
                f"Record the full digest, not {digest!r}: the expected shape is "
                "'<algorithm>:<hex>', as `docker inspect --format '{{index .RepoDigests 0}}'` "
                "prints it. A short or unparseable digest pins nothing."
            ),
        )
    )
