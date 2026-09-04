"""Acceptance tests for `ascep.conformance`, the module that grades a report against C1-C12.

Written against the specified contract before the implementation landed, so the tests are a
statement of what the checker must do rather than a description of what it happens to do.

The hardest thing to get right here is not detecting violations — it is *not* detecting ones
that are not there. A checker that cries wolf on an honest partial report trains people to
ignore it, which is worse than having no checker. `test_a_failing_gate_outside_the_envelope_is_
not_a_c7_error` is the guard for the one place that is easy to get wrong.
"""

from __future__ import annotations

import copy
import json
import pathlib

import pytest

from ascep.conformance import Finding, Verdict, check

ROOT = pathlib.Path(__file__).parent.parent
REPORT = ROOT / "examples" / "moe-26b-h100-tp2" / "report.json"


@pytest.fixture
def report() -> dict:
    """A fresh copy per test — every test here mutates it."""
    return json.loads(REPORT.read_text())


def _rules(findings, severity=None) -> set:
    return {f.rule for f in findings if severity is None or f.severity == severity}


# --- the published example ------------------------------------------------------------


def test_the_published_example_grades_partial(report):
    v = check(report)
    assert v.level == "partial"
    assert v.claimed == "partial"
    assert not v.overstated


def test_the_published_example_has_no_c1_to_c5_errors(report):
    """It is partial because of what it could not publish, not because it is malformed."""
    hard = {f for f in check(report).errors if f.rule in {"C1", "C2", "C3", "C4", "C5"}}
    assert not hard, [(f.rule, f.path, f.message) for f in hard]


def test_missing_reproduction_material_is_warned_not_failed(report):
    """C8 downgrades to partial; it does not invalidate a report that is otherwise sound."""
    v = check(report)
    assert "C8" in _rules(v.warnings)
    assert "C8" not in _rules(v.errors)


def test_a_failing_gate_outside_the_envelope_is_not_a_c7_error(report):
    """The 8,192-token point fails its TTFT gate, and the sustainable tier still equals the
    measured tier — correctly, because capacity was computed at a 2,000-token context and the
    failing point sits outside that envelope. A blanket "any failing row invalidates
    sustainable" rule would reject an honest report, so the rule is envelope-aware: an error
    only when the failure is at or below the context the capacity claim covers."""
    v = check(report)
    failing = [r for r in report["run"]["results"] if r.get("slo_pass") is False]
    assert failing, "fixture no longer exercises this path"
    tiers = report["capacity_tiers"]
    assert tiers["sustainable"]["max_concurrent_users"] == tiers["measured"]["max_concurrent_users"]
    assert "C7" not in _rules(v.errors)
    # but the reader is still told the envelope is bounded
    assert "C7" in _rules(v.warnings)


def test_a_failing_gate_inside_the_envelope_is_a_c7_error(report):
    """Move the failure to a context the capacity claim covers and it must fail."""
    report["run"]["results"][1]["slo_pass"] = False
    report["run"]["results"][1]["context_tokens"] = 1024
    assert "C7" in _rules(check(report).errors)


# --- the verdict object itself --------------------------------------------------------


def test_findings_are_stable_and_partitioned(report):
    v = check(report)
    assert isinstance(v, Verdict)
    assert all(isinstance(f, Finding) for f in v.findings)
    assert list(v.findings) == sorted(v.findings, key=lambda f: (f.rule, f.path))
    assert set(v.errors) | set(v.warnings) == set(v.findings)
    assert not set(v.errors) & set(v.warnings)


def test_every_finding_names_a_rule_a_path_and_an_action(report):
    for f in check(report).findings:
        assert f.rule in {f"C{i}" for i in range(1, 12)}
        assert f.severity in {"error", "warning"}
        assert f.path, f"{f.rule} finding with no path is not actionable"
        assert len(f.message.split()) >= 4, f"{f.rule}: message too terse to act on"


def test_overstating_conformance_is_detected(report):
    report["conformance"] = "conforming"
    v = check(report)
    assert v.claimed == "conforming"
    assert v.level == "partial"
    assert v.overstated


def test_understating_conformance_is_not_flagged(report):
    """Claiming less than you meet is honest, if pessimistic."""
    report["conformance"] = "non-conforming"
    assert not check(report).overstated


# --- C1: complete declaration ---------------------------------------------------------


def test_unjustified_null_is_a_c1_error(report):
    report["serving"]["batching_mode"] = None
    report["serving"].pop("batching_mode_u_reason", None)
    assert "C1" in _rules(check(report).errors)


def test_stale_justification_is_a_c1_error(report):
    """A reason left behind after the field was measured tells a reviewer to discount a
    number that is in fact solid."""
    report["serving"]["tensor_parallel_u_reason"] = "(U) not recorded"
    assert "C1" in _rules(check(report).errors)


def test_a_c1_error_forces_non_conforming(report):
    report["serving"]["batching_mode"] = None
    report["serving"].pop("batching_mode_u_reason", None)
    assert check(report).level == "non-conforming"


# --- C2: provenance tagging -------------------------------------------------------------


def test_a_number_whose_provenance_travels_in_a_sibling_tag_still_needs_one(report):
    """Three figures carry provenance in a ``*_tag`` field rather than a ``provenance`` key,
    and the schemas require the number while leaving the tag optional.

    Nothing but this check stands between that and an untagged claim. It is not hypothetical:
    the published moe-26b example shipped an untagged ``avg_context_tokens`` -- the number the
    KV floor divides by -- and graded partial on unrelated grounds the whole time.
    """
    report["workload"]["avg_context_tokens"] = 2000.0
    report["workload"].pop("avg_context_tokens_tag", None)
    findings = [f for f in check(report).errors if f.rule == "C2"]
    assert [f.path for f in findings] == ["workload.avg_context_tokens_tag"]


def test_an_untagged_number_that_is_null_is_c1s_business_not_c2s(report):
    """An honestly unmeasured field needs a reason, not a provenance tag. Charging it under
    both rules would report two defects for one gap and send the author looking for a second
    thing to fix."""
    report["workload"]["avg_context_tokens"] = None
    report["workload"]["avg_context_tokens_u_reason"] = "(U) no context distribution recorded"
    report["workload"].pop("avg_context_tokens_tag", None)
    assert not [f for f in check(report).errors if f.rule == "C2"]


# --- C3: topology binding -------------------------------------------------------------


def test_topology_that_does_not_multiply_out_is_a_c3_error(report):
    report["serving"]["gpu_count"] = 3
    v = check(report)
    assert "C3" in _rules(v.errors)
    assert v.level == "non-conforming"


# --- C5: binding constraint -----------------------------------------------------------


def test_capacity_without_its_binding_constraint_is_a_c5_error(report):
    report["capacity_tiers"]["measured"]["binding_constraint"] = None
    assert "C5" in _rules(check(report).errors)


def test_sizing_result_without_its_binding_constraint_is_a_c5_error(report):
    report["sizing_result"]["binding_constraint"] = None
    assert "C5" in _rules(check(report).errors)


# --- C6: four tiers -------------------------------------------------------------------


def test_a_missing_tier_is_a_c6_error(report):
    del report["capacity_tiers"]["sustainable"]
    assert "C6" in _rules(check(report).errors)


def test_a_declined_tier_is_only_a_warning(report):
    """The example declines `theoretical` with a (U) reason; that is legitimate and better
    than fabricating a roofline nobody can defend."""
    v = check(report)
    assert report["capacity_tiers"]["theoretical"]["max_concurrent_users"] is None
    assert "C6" not in _rules(v.errors)
    assert "C6" in _rules(v.warnings)


def test_tiers_out_of_order_is_a_c6_error(report):
    report["capacity_tiers"]["recommended"]["max_concurrent_users"] = 10_000
    assert "C6" in _rules(check(report).errors)


def test_roofline_efficiency_at_or_above_one_is_a_c6_error(report):
    report["roofline_comparison"]["roofline_efficiency"] = 1.4
    assert "C6" in _rules(check(report).errors)


# --- C7: gates fixed before the run ---------------------------------------------------


def test_gates_chosen_after_the_run_is_a_c7_error(report):
    report["run"]["slo_gates"]["declared_before_run"] = False
    assert "C7" in _rules(check(report).errors)


# --- robustness -----------------------------------------------------------------------


def test_check_does_not_mutate_its_input(report):
    before = copy.deepcopy(report)
    check(report)
    assert report == before


def test_a_wildly_incomplete_report_does_not_crash():
    v = check({"ascep_version": "0.1.0"})
    assert v.level == "non-conforming"
    assert v.findings


# --- values that are not declarations --------------------------------------------------


def test_a_nan_in_a_measured_tier_is_reported_by_name(report):
    """A NaN neutralises every comparison in this module — the C6 ordering, the C6 roofline
    ceiling, the C7 gate check and the C4 curve count all pass against it. It has to leave the
    numeric rules and reappear under C1, or a report declaring nothing grades conforming."""
    report["capacity_tiers"]["measured"]["max_concurrent_users"] = float("nan")
    v = check(report)
    errors = {f.path for f in v.findings if f.rule == "C1" and f.severity == "error"}
    assert "capacity_tiers.measured.max_concurrent_users" in errors
    assert v.level == "non-conforming"


def test_an_infinity_is_treated_like_a_nan(report):
    """`json.load` parses the bare token Infinity too, so this is a file a real toolchain can
    emit, not a hand-crafted attack."""
    report["capacity_tiers"]["measured"]["max_concurrent_users"] = float("inf")
    v = check(report)
    errors = {f.path for f in v.findings if f.rule == "C1" and f.severity == "error"}
    assert "capacity_tiers.measured.max_concurrent_users" in errors


def test_zero_is_a_declaration_and_is_not_flagged_as_non_finite(report):
    """Guard against over-eager matching. Zero is a legitimate measurement — an error rate of
    0% is the good outcome — and flagging it would make the rule unusable."""
    report["run"]["results"][0]["error_rate_pct"] = 0
    errors = {f.path for f in check(report).findings if f.rule == "C1" and f.severity == "error"}
    assert "run.results.0.error_rate_pct" not in errors


def test_a_bare_u_tag_does_not_justify_a_null_but_a_sentence_does(report):
    """Pasting four characters beside every null once cleared C1 entirely. The sentence after
    the tag is the justification; the tag alone is agreement with the rule, not an answer."""
    row = report["capacity_tiers"]["measured"]
    row["max_concurrent_users"] = None
    path = "capacity_tiers.measured.max_concurrent_users"

    for empty in ("(U)", "(U) ", "(U):", "(U) —"):
        row["max_concurrent_users_u_reason"] = empty
        errors = {f.path for f in check(report).findings if f.rule == "C1"}
        assert path in errors, empty

    row["max_concurrent_users_u_reason"] = "(U) the engine did not report it"
    assert path not in {f.path for f in check(report).findings if f.rule == "C1"}


def test_an_unmeasured_assumptions_entry_still_justifies_a_bare_u_tag(report):
    """The assumption-list fallback predates the sentence rule and carries the same information
    in a different place. Tightening the tag must not reject a report that justifies by naming
    the field, or the fix would cost more honest reports than it catches dishonest ones."""
    path = "capacity_tiers.measured.max_concurrent_users"
    report["capacity_tiers"]["measured"]["max_concurrent_users"] = None
    report["capacity_tiers"]["measured"]["max_concurrent_users_u_reason"] = "(U)"
    report["unmeasured_assumptions"].append(
        {
            "field": path,
            "value_used": "null",
            "impact_if_wrong": "the measured tier is unquotable, so no tier below it is either",
            "cost_to_measure": "one saturation sweep at the deployed shape",
        }
    )
    assert path not in {f.path for f in check(report).findings if f.rule == "C1"}


def test_a_malformed_container_digest_is_a_c8_warning_and_real_ones_are_not(report):
    """A digest is self-describing, so `sha256:0` is malformed rather than unverifiable — the
    one artifact in the bundle this tool can judge without seeing the machine. Rejecting an
    honest sha512 digest would be a worse error than the one being caught."""
    reproduction = report["reproduction"]
    reproduction.pop("container_digest_u_reason", None)
    path = "reproduction.container_digest"

    for bad in ("sha256:0", "n/a", "-", "sha256:"):
        reproduction["container_digest"] = bad
        assert path in {f.path for f in check(report).findings if f.rule == "C8"}, bad

    for good in ("sha256:" + "a1" * 32, "sha512:" + "b7" * 64, "md5:" + "0f" * 16):
        reproduction["container_digest"] = good
        assert path not in {f.path for f in check(report).findings if f.rule == "C8"}, good


def test_a_single_point_campaign_can_say_so_and_the_finding_changes(report):
    """C4 says a single point MUST be labelled, so there has to be somewhere to put the label.
    It stays a warning either way — the limit is real — but an author who declared it should
    not keep reading a message telling them to."""
    report["run"]["results"] = report["run"]["results"][:1]
    unlabelled = next(f for f in check(report).findings if f.path == "run.results")
    assert "single_point" in unlabelled.message
    report["run"]["single_point"] = True
    labelled = next(f for f in check(report).findings if f.path == "run.results")
    assert labelled.severity == "warning"
    assert labelled.message != unlabelled.message
    assert "MUST NOT be quoted as a curve" in labelled.message


def test_a_real_value_containing_the_word_todo_is_not_scaffolding(report):
    """Substring matching made this rule something authors work around instead of with: a
    bucket really can be called TODO-migration, and one false positive teaches people that C1
    errors are noise."""
    report["reproduction"]["raw_records_path"] = "s3://bench/TODO-migration/2026/records.jsonl"
    report["reproduction"].pop("raw_records_path_u_reason", None)
    errors = {f.path for f in check(report).findings if f.rule == "C1"}
    assert "reproduction.raw_records_path" not in errors


def test_a_lowercase_todo_is_still_scaffolding(report):
    """The omission is the same one typed less carefully, and it points at nothing either way."""
    for variant in ("todo", " TODO ", "Todo"):
        report["reproduction"]["raw_records_path"] = variant
        report["reproduction"].pop("raw_records_path_u_reason", None)
        errors = {f.path for f in check(report).findings if f.rule == "C1"}
        assert "reproduction.raw_records_path" in errors, variant


def test_an_epoch_timestamp_is_a_placeholder_not_a_generation_date(report):
    """A parseable sentinel is the worst kind: it validates, it survives `grep TODO`, and it
    reads as a real date. A report generated in 1970 is a report nobody dated."""
    report["report_generated_utc"] = "1970-01-01T00:00:00Z"
    assert "report_generated_utc" in {f.path for f in check(report).findings if f.rule == "C1"}


def test_a_stale_justification_is_reported_once_not_twice(report):
    """Two C1 errors at one path telling the author to delete it and to fill it in is worse
    than one: whichever they act on, the other still fires."""
    report["capacity_tiers"]["measured"]["max_tokens_per_s_u_reason"] = "TODO"
    at_path = [
        f
        for f in check(report).findings
        if f.path == "capacity_tiers.measured.max_tokens_per_s_u_reason"
    ]
    assert len(at_path) == 1
    assert "stale" in at_path[0].message


def test_a_null_image_count_is_not_the_same_claim_as_a_measured_zero(report):
    """The 0-versus-null pair is the most important case in section 9.8. 0 means measured
    and genuinely no media; null means not reported. A checker that accepts null lets an
    unmeasured multimodal workload be published as though it were text-only, and the KV
    floor it implies is wrong by the entire media contribution."""
    report["model"]["input_modalities"] = ["text", "image"]
    report["workload"]["images_per_request"] = None
    findings = {(f.rule, f.severity, f.path) for f in check(report).findings}
    assert ("C4", "error", "workload.images_per_request") in findings
    assert ("C4", "warning", "serving.media_preprocessing") not in findings


def test_a_measured_zero_image_count_is_a_declaration_not_an_omission(report):
    """Zero is the honest answer for a text-only run on a vision-capable model, and
    flagging it would punish the report that did the measurement. The rule must fire on
    the missing declaration, never on the measured absence of media."""
    report["model"]["input_modalities"] = ["text", "image"]
    report["workload"]["images_per_request"] = 0
    findings = {(f.rule, f.severity, f.path) for f in check(report).findings}
    assert ("C4", "error", "workload.images_per_request") not in findings
    assert ("C4", "warning", "serving.media_preprocessing") not in findings


def test_a_video_count_without_a_duration_leaves_the_kv_floor_unknown(report):
    """Two clips can be two seconds or two hours, so the clip count alone does not say how
    much media each request carried. The missing duration is the error; the missing
    preprocessing block is the warning that the media token cost cannot be reproduced."""
    report["model"]["input_modalities"] = ["text", "video"]
    report["workload"]["videos_per_request"] = 2
    report["workload"]["video_seconds_per_request"] = None
    report["serving"].pop("media_preprocessing", None)
    findings = {(f.rule, f.severity, f.path) for f in check(report).findings}
    assert ("C4", "error", "workload.video_seconds_per_request") in findings
    assert ("C4", "warning", "serving.media_preprocessing") in findings
    assert ("C4", "error", "workload.videos_per_request") not in findings
    assert ("C4", "error", "workload.images_per_request") not in findings


def test_a_thinking_branch_without_a_named_mode_cannot_be_read_as_either_profile(report):
    """A model with a thinking branch is two capacity profiles that differ by orders of
    magnitude in output length. A report that does not name the one it measured cannot be
    read as either, so the unnamed mode is the error and nothing else fires."""
    report["model"]["reasoning_modes"] = ["non-thinking", "thinking"]
    report["workload"]["reasoning_mode"] = None
    findings = {(f.rule, f.severity, f.path) for f in check(report).findings}
    assert ("C4", "error", "workload.reasoning_mode") in findings
    assert ("C4", "error", "workload.max_output_tokens") not in findings
    assert ("C4", "error", "workload.reasoning_share") not in findings
    assert ("C4", "error", "run.truncation_rate") not in findings


def test_a_mixed_mode_without_its_share_cap_and_truncation_rate_averages_two_profiles(report):
    """All three fire together because they are one omission, not three: a mixed workload
    without its share, its cap and its truncation rate is a single number averaging two
    capacity profiles that differ by orders of magnitude, and the average describes
    neither."""
    report["workload"]["reasoning_mode"] = "mixed"
    report["workload"]["max_output_tokens"] = None
    report["workload"]["reasoning_share"] = None
    report["run"] = {}
    findings = {(f.rule, f.severity, f.path) for f in check(report).findings}
    assert ("C4", "error", "run.truncation_rate") in findings
    assert ("C4", "error", "workload.max_output_tokens") in findings
    assert ("C4", "error", "workload.reasoning_share") in findings
    assert ("C4", "error", "workload.reasoning_mode") not in findings


def test_media_without_its_preprocessing_is_a_warning_not_an_error(report):
    """The media token cost is decided by the server's sampling rate, frame cap and pixel
    budget, so a media throughput figure without them cannot be reproduced. That limits
    comparability without invalidating the measurement, which is a warning's job."""
    report["model"]["input_modalities"] = ["text"]
    report["workload"]["images_per_request"] = 3
    report["serving"].pop("media_preprocessing", None)
    findings = {(f.rule, f.severity, f.path) for f in check(report).findings}
    assert ("C4", "warning", "serving.media_preprocessing") in findings
    assert ("C4", "error", "workload.images_per_request") not in findings
    assert ("C4", "error", "workload.videos_per_request") not in findings
    assert ("C4", "error", "workload.video_seconds_per_request") not in findings


def test_a_text_only_non_thinking_report_raises_no_media_or_reasoning_findings(report):
    """This is the regression that protects every published example. All ten shipped
    reports are text-only; any of these rules firing spuriously would downgrade reports
    that are already correct, and the failure would look like a protocol change rather
    than a bug."""
    report["model"]["input_modalities"] = ["text"]
    report["model"]["reasoning_modes"] = ["non-thinking"]
    report["workload"]["images_per_request"] = None
    report["workload"]["videos_per_request"] = None
    report["workload"]["video_seconds_per_request"] = None
    report["workload"]["reasoning_mode"] = None
    report["serving"].pop("media_preprocessing", None)
    findings = {(f.rule, f.severity, f.path) for f in check(report).findings}
    for path in (
        "workload.images_per_request",
        "workload.videos_per_request",
        "workload.video_seconds_per_request",
        "workload.reasoning_mode",
        "workload.max_output_tokens",
        "workload.reasoning_share",
        "run.truncation_rate",
        "serving.media_preprocessing",
    ):
        assert ("C4", "error", path) not in findings
        assert ("C4", "warning", path) not in findings


def test_a_malformed_modalities_value_and_a_missing_workload_do_not_crash_c4(report):
    """C4 runs after the schema check but must not depend on it having passed. A grader
    that raises on malformed input turns a bad report into a crash, and a crash tells the
    author nothing about what to fix."""
    report["model"]["input_modalities"] = "image"
    report.pop("workload", None)
    v = check(report)
    assert isinstance(v, Verdict)
    findings = {(f.rule, f.severity, f.path) for f in v.findings}
    for path in (
        "workload.images_per_request",
        "workload.videos_per_request",
        "workload.video_seconds_per_request",
        "workload.reasoning_mode",
        "workload.max_output_tokens",
        "workload.reasoning_share",
        "run.truncation_rate",
        "serving.media_preprocessing",
    ):
        assert ("C4", "error", path) not in findings
        assert ("C4", "warning", path) not in findings


# --- C1: notes explain declared values, never nulls ------------------------------------


def test_a_note_beside_a_null_does_not_justify_the_null(report):
    """notes explain why a declared value is what it is; they are not a (U) reason. A null
    justified by a note alone would let a report carry an unknown with prose that never
    admits the value is missing -- the skeleton would validate."""
    report["serving"]["batching_mode"] = None
    report["serving"].pop("batching_mode_u_reason", None)
    report["serving"]["notes"] = {"batching_mode": "deployment runs whatever the engine picks"}
    errors = {f.path for f in check(report).errors if f.rule == "C1"}
    assert "serving.batching_mode" in errors


def test_a_note_on_a_declared_value_is_not_flagged_as_stale(report):
    """A note beside a solid number must not be reported like a stale (U) reason: the value
    is declared, the note says why, and there is nothing to remove."""
    report["serving"]["notes"] = {"batching_mode": "continuous is the only mode this engine has"}
    errors = {f.path for f in check(report).errors if f.rule == "C1"}
    assert "serving.notes" not in errors
    assert "serving.batching_mode" not in errors


def _declare_image_input(report: dict) -> dict:
    """Turn the text-only example into a schema-valid image *run* with cores declared.

    The published example leaves cpu_cores null with a (U) reason, which is the one shape
    the note rule deliberately stays out of, so a test that only flipped input_modalities
    would pass while measuring nothing. Declaring image also pulls in the model layer's
    vision gates; without them the report is schema-invalid and every C1 finding is the
    schema error, not the rule under test.

    The workload count is set here too, because the note rule gates on media the run
    actually sent rather than media the checkpoint accepts. A helper that declared only the
    modality would build the one report shape the rule is now specified NOT to fire on.
    """
    report["workload"]["images_per_request"] = 2
    report["model"]["input_modalities"] = ["text", "image"]
    report["model"]["image_token_policy"] = "fixed-grid"
    report["model"]["image_tokens_fixed"] = 256
    report["model"]["vision_encoder_params"] = 675_000_000
    report["model"]["vision_encoder_replicated_per_rank"] = True
    report["hardware"]["cpu_cores"] = 12
    report["hardware"].pop("cpu_cores_u_reason", None)
    return report


def test_the_multimodal_fixture_is_schema_valid_so_the_note_rule_is_what_is_measured(report):
    """If declaring image left the report schema-invalid, the tests below would see a C1
    schema error and could pass with the note rule deleted."""
    _declare_image_input(report)
    assert not [f for f in check(report).errors if f.path.startswith("$.")]


def test_a_multimodal_report_without_a_cpu_cores_note_is_a_c1_error(report):
    """Under a media workload the host CPU decodes every image, so a bare integer here
    publishes a ceiling nobody can attribute, and the finding must say that."""
    _declare_image_input(report)
    errors = {f.path: f.message for f in check(report).errors if f.rule == "C1"}
    assert "hardware.cpu_cores" in errors
    assert "capacity input" in errors["hardware.cpu_cores"]
    assert "notes" in errors["hardware.cpu_cores"]


def test_a_cpu_cores_note_satisfies_the_multimodal_obligation(report):
    """The obligation is a note, nothing more: one sentence saying what the number is."""
    _declare_image_input(report)
    report["hardware"]["notes"] = {"cpu_cores": "Cluster QoS grants 12 per GPU; node has 112."}
    errors = {f.path for f in check(report).errors if f.rule == "C1"}
    assert "hardware.cpu_cores" not in errors


def test_a_typoed_note_key_does_not_satisfy_the_cpu_cores_obligation(report):
    """notes keys are free-form strings, so the schema cannot catch cpu_corees; the
    obligation keys on the field name, or the typo would publish the same unattributed
    ceiling the rule exists against."""
    _declare_image_input(report)
    report["hardware"]["notes"] = {"cpu_corees": "cluster QoS allocation, misspelt"}
    errors = {f.path for f in check(report).errors if f.rule == "C1"}
    assert "hardware.cpu_cores" in errors


def test_a_null_cpu_cores_is_reported_once_as_a_missing_value_not_a_missing_note(report):
    """Two findings for one omission trains a reader to skim C1. The null walk owns the
    null; the note rule owns the declared integer nobody attributed."""
    _declare_image_input(report)
    report["hardware"]["cpu_cores"] = None
    paths = [f.path for f in check(report).errors if f.rule == "C1"]
    assert paths.count("hardware.cpu_cores") == 1


def test_a_text_only_report_needs_no_cpu_cores_note(report):
    """Without media the host CPU is not co-limiting the way chapter 9 measures, so
    requiring the note here would be paperwork on reports that are already correct."""
    report["hardware"]["cpu_cores"] = 12
    report["hardware"].pop("cpu_cores_u_reason", None)
    assert report["model"]["input_modalities"] == ["text"]
    errors = {f.path for f in check(report).errors if f.rule == "C1"}
    assert "hardware.cpu_cores" not in errors


def test_a_text_only_run_on_a_multimodal_model_needs_no_cpu_cores_note(report):
    """The gate is what the run sent, not what the checkpoint accepts. Keying it off the
    declared modalities failed every text-only ladder on a multimodal model with a message
    asserting the host decoded images the run never sent, and a rule whose stated reason the
    reader can see is untrue teaches them to route around the grade rather than fix it."""
    _declare_image_input(report)
    report["workload"]["images_per_request"] = 0
    findings = {f.path: f.message for f in check(report).errors if f.rule == "C1"}
    assert "hardware.cpu_cores" not in findings, findings.get("hardware.cpu_cores")


# --- C1: a null value_used in section 7 is the justification, not a gap -----------------


def _assumption(**over) -> dict:
    entry = {
        "field": "workload.peak_to_mean",
        "value_used": None,
        "impact_if_wrong": "peak sizing moves in direct proportion to this figure",
        "cost_to_measure": "two weeks of per-hour traffic logs",
    }
    entry.update(over)
    return entry


def test_a_complete_unmeasured_assumptions_entry_with_a_null_value_used_raises_no_c1(report):
    """A null value_used means no substitute was plugged in -- the field was simply left
    unmeasured. Before the exemption, clearing this finding took either a schema-illegal
    sibling, a section-7 entry naming the register's own field, or an invented value; the
    last needs no ceremony, so the rule pushed authors toward the fabrication the protocol
    exists to prevent."""
    index = len(report["unmeasured_assumptions"])
    report["unmeasured_assumptions"].append(_assumption())
    findings = [
        f
        for f in check(report).findings
        if f.rule == "C1" and f.path.startswith(f"unmeasured_assumptions.{index}")
    ]
    assert not findings, [(f.path, f.message) for f in findings]


def test_an_unmeasured_assumptions_entry_with_a_real_value_used_still_raises_no_c1(report):
    """The exemption must change nothing for the entry that DID plug in a value: it was
    clean before, and a fix that newly flagged it would punish the more informative of the
    two honest shapes a section-7 entry can take."""
    index = len(report["unmeasured_assumptions"])
    report["unmeasured_assumptions"].append(_assumption(value_used=3.0))
    findings = [
        f
        for f in check(report).findings
        if f.rule == "C1" and f.path.startswith(f"unmeasured_assumptions.{index}")
    ]
    assert not findings, [(f.path, f.message) for f in findings]


@pytest.mark.parametrize("name", ["field", "impact_if_wrong", "cost_to_measure"])
def test_a_null_outside_value_used_in_an_unmeasured_assumptions_entry_is_still_reported(
    report, name
):
    """The exemption is scoped to the one key whose null carries no information. The other
    three ARE the justification; a null in any of them is an entry claiming to record an
    omission while omitting the record."""
    index = len(report["unmeasured_assumptions"])
    report["unmeasured_assumptions"].append(_assumption(**{name: None}))
    errors = {f.path for f in check(report).errors if f.rule == "C1"}
    assert f"unmeasured_assumptions.{index}.{name}" in errors


def test_a_value_used_outside_the_assumptions_register_is_graded_like_any_other_null(report):
    """The exemption keys on the path, not the name -- otherwise every future block could
    opt out of C1 by choosing a word. Elsewhere in a report the name buys nothing: the null
    is still charged, and since no other object declares the key, the closed-object rule
    charges it a second time as an undeclared field. Both findings are asserted, because
    the exemption leaking would silence the first while leaving the second, and a test
    watching only the total would still pass."""
    report["run"]["value_used"] = None
    errors = {f.path for f in check(report).errors if f.rule == "C1"}
    assert "run.value_used" in errors
    assert "$.run" in errors


def test_a_block_of_eight_complete_entries_like_bench_emits_raises_no_c1_findings(report):
    """`ascep bench` writes eight section-7 entries on every run, each with a null
    value_used because nothing was plugged in. Before the exemption that made every
    harness-produced report fail its own checker eight times, in the block a reviewer most
    needs to read -- pinned here as a shape, without running a benchmark."""
    report["unmeasured_assumptions"] = [
        _assumption(field=f"workload.assumed_input_{i}") for i in range(8)
    ]
    findings = [
        f
        for f in check(report).findings
        if f.rule == "C1" and f.path.startswith("unmeasured_assumptions")
    ]
    assert not findings, [(f.path, f.message) for f in findings]
