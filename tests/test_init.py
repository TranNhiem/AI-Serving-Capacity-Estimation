"""`ascep init` must scaffold honestly: complete, unfilled, and wrong about nothing.

A scaffolding command has one failure mode worth guarding against, and it is not crashing.
It is emitting a document that *looks* finished — a plausible number where the tool had no
information — because a report built on that is indistinguishable from a measured one, which
is the exact confusion the whole protocol exists to prevent. So these tests check that the
skeleton is structurally complete, that every value in it is visibly a placeholder, that it
scores as a gap rather than as conforming, and that the two things it genuinely cannot fill
in are reported by name rather than guessed at.
"""

from __future__ import annotations

import json
import pathlib
import re
import subprocess
import sys

import pytest

from ascep import ASCEP_VERSION, conformance, init, validation
from ascep.validation import LAYERS, load_schema

ROOT = pathlib.Path(__file__).parent.parent

pytest.importorskip("jsonschema", reason="schema validation is an optional extra")


def _norm(json_path: str) -> str:
    """`$.run.results[0]` -> `run.results[]`, matching how decisions() names a path."""
    stripped = re.sub(r"^\$\.?", "", json_path)
    return re.sub(r"\[\d+\]", "[]", stripped) or "(root)"


@pytest.mark.parametrize("layer", LAYERS)
def test_every_field_the_schema_requires_is_present(layer):
    """Absence and null are different claims, and only one of them is fillable.

    A missing key gives the user nothing to notice; a null with a `_u_reason` next to it is a
    prompt. This is the whole reason the skeleton is generated from the schema rather than
    written by hand.
    """
    skeleton = init.skeleton(layer)
    for name in load_schema(layer).get("required", []):
        assert name in skeleton, f"{layer}: schema requires {name}, skeleton omits it"


@pytest.mark.parametrize("layer", LAYERS)
def test_every_validation_error_points_at_an_unfilled_field(layer):
    """The gate this module was written against.

    A fresh skeleton does NOT validate, and that is the design: where the schema forbids null
    and no value is knowable, an error saying `None is not of type 'integer'` is a to-do item,
    whereas a satisfying `1` is a claim about someone's cluster that would validate silently.
    So `ascep validate` on a skeleton is the fill-in list.

    What is not acceptable is an error at a path the user cannot act on — a placeholder this
    module chose badly, a `$ref` it failed to resolve, a required field it never emitted.
    Those send a first-time user debugging the tool instead of writing their report. Every
    error must therefore land either on a value that is visibly blank, or on a disjunction
    `decisions()` reported by name (whose own message is an unreadable schema dump, which is
    exactly why it gets announced separately).
    """
    skeleton, notes = init._build(layer)
    announced = {note["path"] for note in notes}
    bad = []
    for error in validation.iter_errors(layer, skeleton):
        if _norm(error.json_path) in announced:
            continue
        if _is_placeholder(error.instance):
            continue
        bad.append(f"{error.json_path}: {error.message[:120]}")
    assert not bad, (
        f"{layer} skeleton fails validation at paths the user cannot act on: " + "; ".join(bad)
    )


def _is_placeholder(value) -> bool:
    """Whether a value is visibly unfilled, so an error about it is a prompt not a puzzle."""
    return value is None or value in (init.TODO, init._EPOCH)


@pytest.mark.parametrize("layer", LAYERS)
def test_every_announced_decision_is_a_real_unmet_disjunction(layer):
    """The converse: `decisions()` must not pad its list.

    A note nobody needs teaches the reader to skim past the two that matter.
    """
    skeleton, notes = init._build(layer)
    for note in notes:
        for option in note["options"]:
            assert option, f"{layer}: empty option list at {note['path']}"
        satisfied = [
            option
            for option in note["options"]
            if all(_lookup(skeleton, note["path"], field) is not None for field in option)
        ]
        assert not satisfied, (
            f"{layer}: {note['path']} announces a choice already made by {satisfied}"
        )


def _lookup(skeleton, path: str, field: str):
    """Value of `field` inside the object at `path`, or None if the path does not resolve."""
    node = skeleton
    if path != "(root)":
        for part in path.split("."):
            if part.endswith("[]"):
                node = node.get(part[:-2])
                node = node[0] if isinstance(node, list) and node else None
            else:
                node = node.get(part) if isinstance(node, dict) else None
            if node is None:
                return None
    return node.get(field) if isinstance(node, dict) else None


@pytest.mark.parametrize("layer", LAYERS)
def test_no_number_in_a_skeleton_is_the_tools_own_invention(layer):
    """The failure this whole module is shaped around.

    A number the tool chose to satisfy a `minimum` validates, reads exactly like a
    declaration, and survives into a published report — `gpu_count: 1` would claim a
    single-GPU deployment that nobody measured. So every number present must trace to the
    schema stating it: a `const` or a `default`, never a constraint the tool worked backwards
    from.
    """
    stated = set()

    def collect(node):
        if isinstance(node, dict):
            for key in ("const", "default"):
                if isinstance(node.get(key), (int, float)) and not isinstance(node[key], bool):
                    stated.add(float(node[key]))
            for value in node.values():
                collect(value)
        elif isinstance(node, list):
            for value in node:
                collect(value)

    for name in LAYERS:
        collect(load_schema(name))

    invented = []

    def walk(node, path=""):
        if isinstance(node, dict):
            for key, value in node.items():
                walk(value, f"{path}.{key}" if path else key)
        elif isinstance(node, list):
            for i, value in enumerate(node):
                walk(value, f"{path}[{i}]")
        elif isinstance(node, (int, float)) and not isinstance(node, bool):
            if float(node) not in stated:
                invented.append(f"{path} = {node}")

    walk(init.skeleton(layer))
    assert not invented, f"{layer} skeleton invents numbers the schema never states: {invented}"


def test_ascep_version_is_the_real_version_because_a_placeholder_cannot_match_its_pattern():
    for layer in LAYERS:
        assert init.skeleton(layer).get("ascep_version") == ASCEP_VERSION


def test_ascep_version_is_still_the_only_patterned_string():
    """`init._KNOWN` handles exactly one patterned field. A second one needs a real answer.

    Without this, adding a `pattern` anywhere would make `init` silently emit a document that
    fails validation for a reason the user cannot act on.
    """
    patterned = []

    def walk(node, path, source):
        if isinstance(node, dict):
            if "pattern" in node:
                patterned.append(f"{source}{path}")
            for key, value in node.items():
                walk(value, f"{path}/{key}", source)
        elif isinstance(node, list):
            for i, value in enumerate(node):
                walk(value, f"{path}/{i}", source)

    for layer in LAYERS:
        walk(load_schema(layer), "", layer)
    unhandled = {p.rsplit("/", 1)[-1] for p in patterned} - set(init._KNOWN)
    assert not unhandled, (
        f"schemas gained patterned fields with no known value: {sorted(unhandled)}; "
        "add them to ascep.init._KNOWN or they will never validate"
    )


def test_a_skeleton_does_not_score_as_conforming():
    """The point of the placeholders. A scaffold that passed the conformance check would make
    the checker worthless — anyone could publish an empty report and cite a green verdict."""
    verdict = conformance.check(init.skeleton("capacity-report"))
    assert verdict.level != "conforming", "an unfilled skeleton must not score as conforming"
    assert verdict.findings, "an unfilled skeleton must produce findings"


def test_every_placeholder_is_greppable():
    """`grep -c TODO report.json` is the progress bar. It only works if nothing hides."""
    text = init.render("capacity-report")
    data = json.loads(text)
    assert init.TODO in text
    # Every null carries a companion reason, so no field is silently blank.
    for key, value in data.items():
        if value is None:
            assert f"{key}_u_reason" in data, f"{key} is null with no stated reason"


def _cli(*argv, cwd=None):
    return subprocess.run(
        [sys.executable, "-m", "ascep.cli", *argv],
        capture_output=True,
        text=True,
        cwd=cwd or ROOT,
    )


def test_cli_writes_a_file_and_names_the_decisions_on_stderr(tmp_path):
    out = tmp_path / "report.json"
    result = _cli("init", "-o", str(out))
    assert result.returncode == 0, result.stderr
    assert json.loads(out.read_text())["ascep_version"] == ASCEP_VERSION
    assert "decide:" in result.stderr
    assert "concurrent_users" in result.stderr


def test_cli_stdout_is_a_pipeable_json_document():
    """Diagnostics on stderr, document on stdout: `ascep init | jq` has to work."""
    result = _cli("init", "--layer", "hardware")
    assert result.returncode == 0, result.stderr
    json.loads(result.stdout)


def test_cli_refuses_to_clobber_without_force(tmp_path):
    out = tmp_path / "report.json"
    out.write_text('{"measured": "for six hours"}')
    result = _cli("init", "-o", str(out))
    assert result.returncode == 2
    assert out.read_text() == '{"measured": "for six hours"}', "refused but overwrote anyway"
    assert "--force" in result.stderr

    forced = _cli("init", "-o", str(out), "--force")
    assert forced.returncode == 0, forced.stderr
    assert json.loads(out.read_text())["ascep_version"] == ASCEP_VERSION


def test_init_needs_no_optional_dependency():
    """`init` is the first command a new user runs, and it must work on the bare install.

    Asserted by import rather than by CI alone: this catches a lazy import inside `validation`
    being hoisted to module scope, which the bare-install job would only catch after a push.
    """
    code = (
        "import sys; before=set(sys.modules); import ascep.init; "
        "new={m.split('.')[0] for m in set(sys.modules)-before}; "
        "extra=sorted(m for m in new if m not in sys.stdlib_module_names "
        "and not m.startswith(('ascep','_'))); "
        "assert not extra, extra; print('ok')"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT)
    assert result.returncode == 0, result.stdout + result.stderr


def test_conformance_and_init_agree_on_the_placeholder_marker():
    """`ascep.conformance` duplicates the marker rather than importing it, to stay free of
    imports that could reach outside the stdlib. That is only safe if they cannot drift."""
    assert conformance._PLACEHOLDER == init.TODO


def test_a_leftover_placeholder_is_an_error_not_a_declaration():
    """The hole `ascep init` would otherwise have opened.

    Every rule in the checker tests `is None`, so a string sitting in a slot reads as
    declared. Filling in the numbers and missing one string must not produce a report that
    claims a reproduction bundle it does not have.
    """
    report = {
        "reproduction": {"raw_records_path": init.TODO},
        "hardware": {"gpu_model": "H100-SXM-80GB"},
    }
    paths = {f.path for f in conformance.check(report).findings if f.severity == "error"}
    assert "reproduction.raw_records_path" in paths
    assert "hardware.gpu_model" not in paths, "a real value must not be flagged"


def test_an_unreplaced_u_reason_is_caught_even_though_c1_skips_reason_prose():
    """`_walk_nulls` skips `*_u_reason` keys on purpose — they are prose about a claim, not
    claims. That exemption is exactly where a generated reason would otherwise hide."""
    reason = init._u_reason("vram_bytes_per_gpu")
    report = {"hardware": {"vram_bytes_per_gpu": None, "vram_bytes_per_gpu_u_reason": reason}}
    findings = conformance.check(report).findings
    assert any(f.path == "hardware.vram_bytes_per_gpu_u_reason" for f in findings)


def test_an_empty_string_is_not_a_declaration():
    report = {"hardware": {"gpu_model": "   "}}
    paths = {f.path for f in conformance.check(report).findings if f.severity == "error"}
    assert "hardware.gpu_model" in paths
