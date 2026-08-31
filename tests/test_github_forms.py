"""Structural tests for the GitHub issue forms and the template chooser.

These files have a uniquely nasty failure mode: GitHub parses them strictly and, when a form
is malformed, does not show an error — it silently drops the template from the chooser. Nobody
who could fix it ever sees the failure, and the first symptom is months of free-text issues
missing the topology, the context length and the capacity tier, which is precisely the
under-specified capacity claim this whole project exists to stop. Reviewing YAML by eye does
not catch it. Parsing it in CI does.

The assertions below are deliberately split in two. The generic ones encode GitHub's own
schema, so they fail if the syntax is wrong. The project-specific ones encode the fields that
make a submission triageable — the capacity tier, the binding constraint, and the acknowledgement
that nothing site-identifying is being pasted into a public issue — so they fail if a
well-intentioned simplification removes the question that makes the form worth having.
"""

from __future__ import annotations

import os
import pathlib
import re
import sys

import pytest

ROOT = pathlib.Path(__file__).parent.parent
FORM_DIR = ROOT / ".github" / "ISSUE_TEMPLATE"

# GitHub's element vocabulary. An unknown `type` is the most common way a form dies silently.
INPUT_TYPES = {"input", "textarea", "dropdown", "checkboxes"}
BODY_TYPES = INPUT_TYPES | {"markdown"}
ID_RE = re.compile(r"^[a-zA-Z0-9_-]+$")


def _yaml():
    """pyyaml lives in the `run` extra, which CI installs; a local dev box may not have it.

    Skipping locally is a convenience. Skipping in CI would reproduce the exact failure this
    file guards against — a check that quietly does nothing — so there it is an error.
    """
    try:
        import yaml
    except ImportError:  # pragma: no cover - exercised only on a bare interpreter
        if os.environ.get("GITHUB_ACTIONS") or os.environ.get("CI"):
            raise AssertionError(
                "pyyaml is missing in CI, so the issue forms went unchecked. "
                "Install the `run` extra rather than letting this test skip."
            ) from None
        pytest.skip("pyyaml not installed; `pip install -e '.[run]'` to run the form checks")
    return yaml


def _load(name: str):
    return _yaml().safe_load((FORM_DIR / name).read_text(encoding="utf-8"))


def _forms():
    return sorted(p.name for p in FORM_DIR.glob("*.yml") if p.name != "config.yml")


def _fields(form) -> dict:
    """Every element that collects an answer, keyed by id."""
    return {e["id"]: e for e in form["body"] if e.get("id")}


def _labels(form) -> str:
    """All prompt text in one blob, lowercased — for asking whether a topic is covered at all."""
    parts = []
    for element in form["body"]:
        attrs = element.get("attributes") or {}
        parts += [str(attrs.get("label", "")), str(attrs.get("description", ""))]
        for option in attrs.get("options") or []:
            parts.append(option["label"] if isinstance(option, dict) else str(option))
    return " ".join(parts).lower()


def test_there_are_issue_forms_at_all():
    """A guard against the directory being renamed or the forms landing in the wrong place.

    GitHub only reads `.github/ISSUE_TEMPLATE/`; a form one directory up is invisible, and
    every other test here would vacuously pass over an empty glob.
    """
    assert FORM_DIR.is_dir(), f"{FORM_DIR} is missing; GitHub reads no other location"
    assert _forms(), "no issue forms found, so the chooser offers nothing but a blank issue"


@pytest.mark.parametrize("name", ["capacity-report.yml", "protocol-defect.yml"])
def test_form_parses_and_matches_githubs_schema(name):
    """The whole point: malformed YAML removes the template from the chooser without a word."""
    form = _load(name)
    assert isinstance(form, dict), f"{name} is not a YAML mapping"
    for key in ("name", "description", "body"):
        assert form.get(key), f"{name} is missing the required top-level `{key}`"
    assert isinstance(form["body"], list) and form["body"], f"{name} has an empty body"

    seen = set()
    for i, element in enumerate(form["body"]):
        where = f"{name} body[{i}]"
        assert isinstance(element, dict), f"{where} is not a mapping"
        kind = element.get("type")
        assert kind in BODY_TYPES, f"{where} has unknown type {kind!r}"

        element_id = element.get("id")
        if element_id is not None:
            assert ID_RE.match(element_id), f"{where} id {element_id!r} has illegal characters"
            assert element_id not in seen, f"{where} reuses id {element_id!r}"
            seen.add(element_id)

        attrs = element.get("attributes") or {}
        if kind == "markdown":
            assert attrs.get("value"), f"{where} is a markdown block with nothing in it"
            # `validations` on markdown is rejected outright by GitHub.
            assert "validations" not in element, f"{where} markdown cannot carry validations"
        else:
            assert attrs.get("label"), f"{where} ({kind}) has no label"

        if kind == "dropdown":
            options = attrs.get("options")
            assert isinstance(options, list) and len(options) > 1, (
                f"{where} is a dropdown with fewer than two options, which is a label"
            )
            assert all(isinstance(o, str) and o for o in options), (
                f"{where} dropdown options must be plain non-empty strings"
            )
        if kind == "checkboxes":
            options = attrs.get("options")
            assert isinstance(options, list) and options, f"{where} has no checkboxes"
            assert all(isinstance(o, dict) and o.get("label") for o in options), (
                f"{where} every checkbox needs a label mapping"
            )

        validations = element.get("validations") or {}
        assert set(validations) <= {"required"}, f"{where} has unknown validations {validations}"
        if validations:
            assert kind in INPUT_TYPES, f"{where} cannot be required; it collects nothing"
            assert element_id, f"{where} is required but has no id, so the answer is unaddressable"


@pytest.mark.parametrize("name", ["capacity-report.yml", "protocol-defect.yml"])
def test_every_form_makes_the_no_secrets_acknowledgement_mandatory(name):
    """A public issue is the easiest place in this project to leak a customer.

    CONTRIBUTING.md forbids hostnames, credentials, internal paths, job IDs, customer names
    and pricing. An unchecked box is a suggestion; `required: true` is the only version that
    makes a submitter read the sentence before pasting an internal benchmark log.
    """
    form = _load(name)
    boxes = [
        option
        for element in form["body"]
        if element.get("type") == "checkboxes"
        for option in (element.get("attributes") or {}).get("options", [])
        if re.search(r"hostname|credential|customer|secret|internal", option["label"], re.I)
    ]
    assert boxes, f"{name} never asks the submitter to confirm the content is publishable"
    assert any(b.get("required") for b in boxes), (
        f"{name} asks about secrets but does not require the answer, so it will be skipped"
    )


def test_capacity_report_form_collects_what_makes_a_number_meaningful():
    """A capacity figure without these is not triageable, and asking later rarely works.

    Each of these corresponds to a conformance rule: the topology to C3, the context length to
    C4, the binding constraint to C5, the tier to C6. A form that omits one produces issues
    that cannot be turned into a report without a round trip the submitter usually abandons.
    """
    form = _load("capacity-report.yml")
    text = _labels(form)
    for topic, needles in {
        "accelerator": ("accelerator", "gpu"),
        "topology / parallelism (C3)": ("tensor", "topolog", "parallel"),
        "model": ("model",),
        "precision": ("precision", "quant"),
        "serving framework": ("framework", "vllm", "engine"),
        "context length (C4)": ("context", "input token", "sequence length"),
        "capacity tier (C6)": ("tier", "sustainable", "theoretical"),
        "binding constraint (C5)": ("binding", "constraint", "bound by"),
        "reproduction bundle (C8)": ("reproduc", "bundle", "artifact"),
    }.items():
        assert any(n in text for n in needles), f"capacity-report.yml never asks about {topic}"


def test_the_tier_and_binding_constraint_are_closed_vocabularies():
    """Free text here defeats the purpose.

    "sustainable" and "theoretical" differ by whatever margin the hardware has left; C6 exists
    because the two get used interchangeably. A dropdown forces the submitter to pick one and
    makes the answer machine-readable; a text box invites "about 2000 tok/s I think".
    """
    form = _load("capacity-report.yml")
    dropdowns = [
        (element["id"], [o.lower() for o in element["attributes"]["options"]])
        for element in form["body"]
        if element.get("type") == "dropdown" and element.get("id")
    ]
    tiers = [opts for _, opts in dropdowns if any("sustainable" in o for o in opts)]
    assert tiers, "the capacity tier is not a dropdown, so C6 is answered in prose or not at all"
    assert all(
        any(word in " ".join(opts) for word in ("theoretical", "measured", "recommended"))
        for opts in tiers
    ), "the tier dropdown does not offer the other tiers, so it cannot distinguish them"

    binding = [
        opts
        for _, opts in dropdowns
        if sum(any(w in o for o in opts) for w in ("kv", "throughput", "weight")) >= 2
    ]
    assert binding, "the binding constraint (C5) is not a dropdown over weights / kv / throughput"


def test_protocol_defect_form_demands_a_concrete_wrong_number():
    """Without one, a spec complaint is a matter of taste and consumes maintainer time as one.

    The form's job is to make that requirement visible before the submitter starts typing,
    not to have a maintainer explain it in a comment afterwards.
    """
    form = _load("protocol-defect.yml")
    text = _labels(form)
    assert any(n in text for n in ("wrong number", "concrete", "specific")), (
        "protocol-defect.yml does not ask for the concrete wrong number the defect permits"
    )
    assert any(n in text for n in ("chapter", "c1", "rule")), (
        "protocol-defect.yml does not ask which rule or chapter is at fault"
    )
    required = [e for e in form["body"] if (e.get("validations") or {}).get("required")]
    assert required, "nothing in protocol-defect.yml is required, so it can be submitted empty"


def test_chooser_config_is_valid_and_points_at_the_spec():
    """The chooser is the one page every issue author sees, and it is easy to leave broken.

    Blank issues stay enabled on purpose: a protocol that asks people to report what it failed
    to anticipate cannot then force every thought into two forms.
    """
    config = _load("config.yml")
    assert isinstance(config, dict)
    assert set(config) <= {"blank_issues_enabled", "contact_links"}, (
        f"config.yml has keys GitHub does not accept: {set(config)}"
    )
    assert config.get("blank_issues_enabled") is True, (
        "blank issues are disabled, so anything the two forms did not anticipate has nowhere to go"
    )
    links = config.get("contact_links") or []
    assert links, "the chooser offers no route to the specification"
    for link in links:
        assert set(link) == {"name", "about", "url"}, f"contact link has wrong keys: {set(link)}"
        assert all(link[k] for k in link), f"contact link has an empty field: {link}"
        assert link["url"].startswith("https://"), f"non-https contact link: {link['url']}"
    assert any("SPEC.md" in link["url"] for link in links), (
        "no contact link points at protocol/SPEC.md, the document that answers most questions"
    )


def test_the_secret_scanner_actually_reaches_the_issue_forms():
    """Not a second scan — a guard that the existing one still covers this directory.

    `tools/check_no_secrets.py` already walks the whole repo, and the placeholder values in
    these forms are a likely carrier for a real hostname: someone illustrates the field with
    the actual cluster they benchmarked on. Duplicating its denylist here would mean writing
    the banned terms into a tracked file, which is the thing the scanner exists to prevent.
    So this asserts coverage instead. If `.github` ever lands in SKIP_DIRS, or `.yml` in
    SKIP_SUFFIXES, the most public files in the repository stop being scanned and nothing
    else says so.
    """
    sys.path.insert(0, str(ROOT / "tools"))
    try:
        import check_no_secrets
    finally:
        sys.path.pop(0)
    scanned = set(check_no_secrets.iter_files(ROOT))
    for path in sorted(FORM_DIR.glob("*.yml")):
        assert path in scanned, f"{path.relative_to(ROOT)} is not covered by the secret scan"
