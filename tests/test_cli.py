"""Acceptance tests for `ascep.cli`, the only surface most users will ever touch.

The library modules can all be correct while the command line still betrays the protocol:
an exit code of 0 on a non-conforming report teaches CI to pass anything, a diagnostic
printed to stdout corrupts every pipeline that parses it, and — worst of all — a workload
field dropped on the way into the model produces a confidently wrong GPU count with no
error at all. This suite exists because the most expensive bug the toolkit ever shipped
was of the third kind: `size` under-reported demand exactly fivefold and did not crash, and
only a human reading the output noticed. `test_size_reports_the_true_demand_for_the_
published_workload` is the guard against that class of bug.

Almost every test drives `main(argv)` in-process and captures output with `capsys`, which
keeps the suite fast and makes exit codes assertable. Exactly one test shells out to
`python -m ascep.cli`, because an in-process-only suite passes even when the module entry
point is broken. No file under `examples/` is ever mutated; every mutation happens on a
copy written under `tmp_path`.
"""

from __future__ import annotations

import dataclasses
import json
import pathlib
import subprocess
import sys
import textwrap

from ascep import ASCEP_VERSION, __version__, capacity, conformance
from ascep.cli import _WORKLOAD_KEYS, main

ROOT = pathlib.Path(__file__).parent.parent
REPORT = ROOT / "examples" / "moe-26b-h100-tp2" / "report.json"
WORKLOAD = ROOT / "examples" / "chatbot-10k-dau" / "workload.json"


def _write(tmp_path: pathlib.Path, name: str, payload) -> pathlib.Path:
    path = tmp_path / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _mutated_report(tmp_path: pathlib.Path, mutate) -> pathlib.Path:
    """Load the real published report, apply `mutate` to the dict, and write the copy.

    The CLI only reads files, so a mutation test needs a mutated file; keeping the
    load/dump boilerplate here means the tests below show only the mutation itself.
    """
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    mutate(report)
    return _write(tmp_path, "report.json", report)


def _key_values(stdout: str) -> dict[str, str]:
    """Parse `size` output ("key: value" per line) so failures name the number that moved."""
    return dict(line.split(": ", 1) for line in stdout.splitlines())


def _size_args(workload: pathlib.Path, **overrides) -> list[str]:
    flags = {
        "--kv-tokens": "574798",
        "--throughput-tok-s": "1459",
        "--gpus-per-replica": "2",
        "--headroom": "1.15",
    }
    flags.update(overrides)
    args = ["size", "--workload", str(workload)]
    for flag, value in flags.items():
        args.extend([flag, value])
    return args


# --- version and the entry point ------------------------------------------------------


def test_version_prints_the_package_and_protocol_versions(capsys):
    assert main(["version"]) == 0
    out = capsys.readouterr()
    assert out.out.strip() == f"ascep {__version__} (protocol {ASCEP_VERSION})"
    assert out.err == ""


def test_version_works_through_the_module_entry_point():
    """The one test that runs `python -m ascep.cli` for real. Everything else in this
    module calls main() in-process, and that suite would still pass if the `__main__`
    wiring broke and the installed entry point did nothing at all — which is precisely
    the failure a user would hit first."""
    result = subprocess.run(
        [sys.executable, "-m", "ascep.cli", "version"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert result.stdout.strip() == f"ascep {__version__} (protocol {ASCEP_VERSION})"


def test_no_arguments_prints_help_and_exits_2(capsys):
    """A bare invocation is a usage error, not a success: an exit code of 0 would make
    `ascep` look like it accomplished something in a script that forgot the subcommand."""
    assert main([]) == 2
    assert "usage" in capsys.readouterr().out.lower()


# --- validate -------------------------------------------------------------------------


def test_validate_accepts_the_published_report(capsys):
    assert main(["validate", str(REPORT)]) == 0
    assert capsys.readouterr().out.strip() == "OK"


def test_validate_workload_layer_accepts_the_published_workload(capsys):
    assert main(["validate", str(WORKLOAD), "--layer", "workload"]) == 0
    assert capsys.readouterr().out.strip() == "OK"


def test_validate_reports_a_schema_violation(capsys, tmp_path):
    """Dropping a required key must fail validation with at least one message a human can act
    on — an exit code of 1 with silent stdout would be undiagnosable in CI."""
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    del report["hardware"]
    path = _write(tmp_path, "report.json", report)
    assert main(["validate", str(path)]) == 1
    out = capsys.readouterr().out
    messages = [line for line in out.splitlines() if line.strip()]
    assert messages, "schema violation produced no diagnostic at all"
    assert "OK" not in out


def test_validate_sends_io_errors_to_stderr_not_stdout(capsys):
    """stdout is the machine-parseable channel; a tool that writes errors there corrupts
    any pipeline that parses it. A missing file must be diagnosed on stderr and stdout
    must stay empty."""
    missing = ROOT / "examples" / "no-such-report" / "report.json"
    assert main(["validate", str(missing)]) == 1
    out = capsys.readouterr()
    assert out.out == ""
    assert out.err.startswith("error:")


# --- conformance ----------------------------------------------------------------------


def test_conformance_grades_the_published_report_partial(capsys):
    assert main(["conformance", str(REPORT)]) == 0
    assert capsys.readouterr().out.splitlines()[-1] == "verdict: partial (claimed: partial)"


def test_conformance_strict_fails_on_warnings(capsys):
    """The published report is partial *because* of its warnings, so --strict must reject
    it. --strict is the flag CI runs with; if strict ever silently passed a warned report,
    CI would stop meaning anything while still looking green."""
    assert main(["conformance", "--strict", str(REPORT)]) == 1


def test_conformance_calls_out_an_overstated_claim(capsys, tmp_path):
    """Claiming `conforming` while grading `partial` must be shouted, since otherwise the
    report would be quietly compared against honest ones. The alarm is the OVERSTATED
    banner; the process itself still exits 0 because the *graded* level is not
    non-conforming and --strict was not given."""
    path = _mutated_report(tmp_path, lambda r: r.__setitem__("conformance", "conforming"))
    assert main(["conformance", str(path)]) == 0
    assert "OVERSTATED" in capsys.readouterr().out


def test_conformance_exits_1_on_a_c1_violation(capsys, tmp_path):
    def mutate(report):
        report["serving"]["batching_mode"] = None
        report["serving"].pop("batching_mode_u_reason", None)

    path = _mutated_report(tmp_path, mutate)
    assert main(["conformance", str(path)]) == 1
    verdict = capsys.readouterr().out.splitlines()[-1]
    assert verdict == "verdict: non-conforming (claimed: partial)"


def test_conformance_groups_findings_under_rule_headers(capsys):
    """Findings print grouped under a bare `C8:`-style header, each member line indented
    and bracketed, because that grouping is what a reviewer scans. A finding printed
    without its header would read as free-floating noise."""
    assert main(["conformance", str(REPORT)]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert "C8:" in lines
    members = []
    for line in lines[lines.index("C8:") + 1 :]:
        if line.startswith("  ["):
            members.append(line)
        elif line.strip():
            break  # reached the next group header or the verdict line
    assert members, "C8 group header printed with no findings beneath it"
    assert all(line.startswith("  [") for line in members)


# --- conformance --raise --------------------------------------------------------------
#
# The published report grades `partial`, so claiming `non-conforming` on a copy of it is a
# genuinely understated claim and these tests can run the real checker rather than a stub.


def _understated(tmp_path: pathlib.Path, note: str | None = None) -> pathlib.Path:
    """A copy of the published report claiming less than the checks support."""

    def mutate(report):
        report["conformance"] = "non-conforming"
        if note is not None:
            report["conformance_note"] = note

    return _mutated_report(tmp_path, mutate)


def test_conformance_without_raise_never_touches_the_file(capsys, tmp_path):
    """A checker is run on other people's reports. One that rewrites a file it was asked
    only to read cannot be run on a colleague's artifact, or in CI, or twice."""
    path = _understated(tmp_path)
    before = path.read_bytes()
    assert main(["conformance", str(path)]) == 0
    capsys.readouterr()
    assert path.read_bytes() == before


def test_conformance_raise_writes_the_computed_level_into_the_report(capsys, tmp_path):
    """A grade that exists only in a terminal is not part of the artifact. Without this the
    flagship report circulates forever claiming the floor its harness had to start from."""
    path = _understated(tmp_path)
    assert main(["conformance", str(path), "--raise"]) == 0
    out = capsys.readouterr().out
    assert "verdict: partial (claimed: non-conforming)" in out
    assert json.loads(path.read_text(encoding="utf-8"))["conformance"] == "partial"


def test_conformance_raise_swaps_the_draft_paragraph_for_the_graded_one(capsys, tmp_path):
    """The draft paragraph says the report is ungraded and that `ascep conformance` may
    raise the claim. Left in place beside a raised claim it contradicts the field above it,
    and a reader who catches one note lying stops believing the rest of them."""
    path = _understated(tmp_path, note=conformance.DRAFT_NOTE)
    assert main(["conformance", str(path), "--raise"]) == 0
    capsys.readouterr()
    note = json.loads(path.read_text(encoding="utf-8"))["conformance_note"]
    assert note.startswith(conformance.GRADED_NOTE)
    assert "ungraded draft" not in note
    # The sentence that explains why the report looks thin is true of a graded report too,
    # and dropping it would leave the empty sections looking like an oversight.
    assert "the four report sections it cannot observe" in note


def test_conformance_raise_keeps_a_caveat_appended_after_the_draft_paragraph(tmp_path, capsys):
    """The harness appends the censoring and lower-bound sentences after the draft
    paragraph. A swap that ate them would turn a declared lower bound into a bare maximum
    -- the exact overstatement the caveats exist to prevent."""
    caveat = (
        " The ladder was censored (wall clock), so every concurrency figure in this report "
        "is a lower bound on the hardware, not a finding."
    )
    path = _understated(tmp_path, note=conformance.DRAFT_NOTE + caveat)
    assert main(["conformance", str(path), "--raise"]) == 0
    capsys.readouterr()
    note = json.loads(path.read_text(encoding="utf-8"))["conformance_note"]
    assert note == conformance.GRADED_NOTE + caveat


def test_conformance_raise_leaves_a_note_it_did_not_write_alone(capsys, tmp_path):
    """Prefix match, never a pattern match. A hand-written note is the author's argument
    about their own report, and half-eating it is worse than leaving it untouched."""
    theirs = "We ran this on a shared node, so treat the tail latencies as soft."
    path = _understated(tmp_path, note=theirs)
    assert main(["conformance", str(path), "--raise"]) == 0
    capsys.readouterr()
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["conformance_note"] == theirs
    assert saved["conformance"] == "partial", "the claim is raised even when the note is not"


def test_conformance_raise_on_an_overstated_claim_writes_nothing(capsys, tmp_path):
    """--raise must never become the quiet way to lower a claim to whatever the checks
    happen to support. Lowering is the author's to do, having read the OVERSTATED line."""
    path = _mutated_report(tmp_path, lambda r: r.__setitem__("conformance", "conforming"))
    before = path.read_bytes()
    assert main(["conformance", str(path), "--raise"]) == 0
    out = capsys.readouterr().out
    assert "OVERSTATED" in out
    assert path.read_bytes() == before
    assert "raised:" not in out and "graded:" not in out


def test_conformance_raise_is_idempotent(capsys, tmp_path):
    """Running the checker twice must not produce a diff. A command that rewrites the file
    on every run makes a raised report indistinguishable from an edited one in review."""
    path = _understated(tmp_path, note=conformance.DRAFT_NOTE)
    assert main(["conformance", str(path), "--raise"]) == 0
    capsys.readouterr()
    once = path.read_bytes()
    assert main(["conformance", str(path), "--raise"]) == 0
    assert path.read_bytes() == once
    assert "unchanged:" in capsys.readouterr().out


def test_conformance_raise_writes_the_layout_bench_writes(capsys, tmp_path):
    """A raised report and a fresh draft must diff on their content, not their whitespace,
    or every re-run of the harness looks like a rewrite of the whole file."""
    path = _understated(tmp_path)
    assert main(["conformance", str(path), "--raise"]) == 0
    capsys.readouterr()
    text = path.read_text(encoding="utf-8")
    assert text == json.dumps(json.loads(text), indent=2) + "\n"


# --- render ---------------------------------------------------------------------------


def test_render_writes_markdown_to_stdout(capsys):
    assert main(["render", str(REPORT)]) == 0
    out = capsys.readouterr().out
    assert out.startswith("# ASCEP Capacity Report — ")
    for section in range(1, 9):
        assert f"## {section}." in out


def test_render_to_file_matches_render_to_stdout(capsys, tmp_path):
    """A renderer whose two output paths disagree silently publishes something different
    from what the author reviewed on screen. The `-o` form must write byte-for-byte what
    the stdout form printed, modulo the single newline print() adds, and must print
    nothing of its own."""
    assert main(["render", str(REPORT)]) == 0
    from_stdout = capsys.readouterr().out
    out_file = tmp_path / "r.md"
    assert main(["render", str(REPORT), "-o", str(out_file)]) == 0
    assert capsys.readouterr().out == ""
    written = out_file.read_text(encoding="utf-8")
    assert from_stdout == written + "\n"
    assert written.endswith("\n") and not written.endswith("\n\n")


# --- size: the regression this suite exists for ----------------------------------------


def test_size_reports_the_true_demand_for_the_published_workload(capsys):
    """The regression guard for the most expensive bug this command ever shipped.

    This exact invocation once printed `demand_tok_s: 370.37` — precisely one fifth of the
    true figure — because `requests_per_session` had been left out of a hand-maintained
    allowlist of Workload fields in cli.py, so the turns-per-session multiplier was
    silently dropped between the JSON file and the dataclass. It did not crash. It printed
    a confidently wrong procurement number, which is the single failure mode this protocol
    exists to prevent, and it is invisible to schema validation and type checking alike:
    only redoing the arithmetic catches it.

    The output is parsed into key/value pairs rather than matched as a blob so that a
    future failure names the number that moved instead of pointing at a whole-page diff.
    """
    assert main(_size_args(WORKLOAD)) == 0
    values = _key_values(capsys.readouterr().out)
    assert values["demand_tok_s"] == "1851.85"
    assert values["peak_concurrent_users"] == "555.56"
    assert values["gpus"] == "2"
    assert values["binding_constraint"] == "throughput"


def test_workload_keys_cover_every_field_of_the_workload_dataclass():
    """The structural half of the regression guard above. `_WORKLOAD_KEYS` must be exactly
    the field set of `capacity.Workload`, so that adding a field to the dataclass cannot
    reintroduce the silent-drop bug the way extending a hand-maintained list once failed
    to. Equality, not subset: a key that is not a Workload field would be just as wrong."""
    dataclass_fields = {field.name for field in dataclasses.fields(capacity.Workload)}
    assert set(_WORKLOAD_KEYS) == dataclass_fields


def test_size_treats_a_null_field_as_not_declared(capsys, tmp_path):
    """Null means "not declared", not zero. Coercing null to 0 would change the arithmetic
    without a word of complaint — the same class of silent wrong number as the allowlist
    bug, one coercion away. A file with the key nulled must therefore give exactly the
    answer the same file gives with the key omitted entirely."""
    workload = json.loads(WORKLOAD.read_text(encoding="utf-8"))
    with_null = _write(tmp_path, "null.json", dict(workload, duty_cycle=None))
    without_key = {key: value for key, value in workload.items() if key != "duty_cycle"}
    without_path = _write(tmp_path, "bare.json", without_key)

    assert main(_size_args(with_null)) == 0
    null_output = capsys.readouterr().out
    assert main(_size_args(without_path)) == 0
    bare_output = capsys.readouterr().out
    assert null_output == bare_output
    assert _key_values(null_output)["gpus"] == "2"


def test_size_rejects_a_workload_file_that_is_not_an_object(capsys, tmp_path):
    """A JSON array parses fine but cannot be a workload declaration; treating it as one
    would raise deep inside the model instead of at the boundary where the user can act
    on the message."""
    path = _write(tmp_path, "workload.json", ["not", "an", "object"])
    assert main(_size_args(path)) == 1
    out = capsys.readouterr()
    assert out.out == ""
    assert "object" in out.err


def test_size_refuses_an_unsatisfiable_workload(capsys):
    """With effectively no KV pool and effectively no throughput, no replica count serves
    the published workload. The command must fail loudly; printing any GPU count here
    would be an extrapolation dressed up as a measurement, and somebody would buy it."""
    args = [
        "size",
        "--workload",
        str(WORKLOAD),
        "--kv-tokens",
        "1",
        "--throughput-tok-s",
        "1",
    ]
    assert main(args) == 1
    out = capsys.readouterr()
    assert out.out == ""
    assert out.err.startswith("error:")


# --- the stdlib-only promise ----------------------------------------------------------


#: Injected ahead of the real finders so the named packages raise ImportError even when they
#: are installed. Simulating the bare install is the only way to test it from a dev checkout,
#: where the extras are always present — the one machine where this can never fail naturally.
_BLOCK_EXTRAS = """
import sys
_BLOCKED = {"jsonschema", "referencing", "yaml", "httpx"}
class _Blocker:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in _BLOCKED:
            raise ImportError(name)
        return None
sys.meta_path.insert(0, _Blocker())
"""


def _child(body: str, *argv: str, bare: bool = False) -> subprocess.CompletedProcess:
    """Run `body` in a subprocess, reading its paths from `sys.argv` rather than interpolation.

    The bodies below contain set comprehensions, so `str.format` is not an option and
    `%`-interpolation trips the linter; passing paths as real arguments is both cleaner and
    immune to a path that happens to contain a brace. `bare=True` prepends the import blocker.
    """
    prelude = _BLOCK_EXTRAS if bare else ""
    return subprocess.run(
        [sys.executable, "-c", prelude + textwrap.dedent(body), *argv],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )


def test_version_and_size_import_nothing_outside_the_standard_library():
    """`pyproject.toml` keeps `jsonschema`/`httpx`/`pyyaml` in an optional extra, and these two
    commands must not reach them even transitively. The promise matters because the people most
    likely to reach for `ascep size` are on a locked-down benchmark cluster where installing a
    dependency is a ticket, and a capacity estimate you cannot run is worth nothing.

    An eager `from ascep import validation` added anywhere in the import chain would break it
    silently — the command keeps working on a developer laptop, which is the one machine where
    the extras are always installed. Hence the subprocess: the already-warm imports of this test
    session would otherwise mask the regression.

    Detection is by install location, not by `sys.stdlib_module_names`. That attribute arrived in
    3.10 and `requires-python` is `>=3.9`, so the obvious spelling fails on the oldest interpreter
    we claim to support — and skipping it there would drop the guarantee on exactly the machine
    most likely to have a strange environment. Asking "did this module come out of site-packages"
    also states the promise more directly: the thing being forbidden is reaching a *pip-installed*
    dependency, which is the ticket the cluster user cannot file.
    """
    done = _child(
        """
        import os, sys, sysconfig
        from ascep.cli import main
        assert main(["version"]) == 0
        assert main(["size", "--workload", sys.argv[1],
                     "--kv-tokens", "574798", "--throughput-tok-s", "1459",
                     "--gpus-per-replica", "2"]) == 0
        paths = sysconfig.get_paths()
        sitedirs = tuple(os.path.realpath(p) + os.sep
                         for p in {paths.get("purelib"), paths.get("platlib")} if p)
        def installed(name):
            f = getattr(sys.modules.get(name), "__file__", None)
            return bool(f) and os.path.realpath(f).startswith(sitedirs)
        extra = sorted({m.split(".")[0] for m in list(sys.modules)
                        if installed(m) and not m.startswith(("ascep", "_"))})
        assert not extra, extra
        """,
        str(WORKLOAD),
    )
    assert done.returncode == 0, done.stderr


def test_conformance_and_render_still_work_when_the_extras_are_missing(tmp_path):
    """A weaker promise than the one above, and a different one. `conformance` *uses*
    jsonschema when it is there — the schema check is part of C1 — but must not require it,
    and `render` must not either. Both have to survive a bare `pip install ascep`.
    """
    done = _child(
        """
        import sys
        from ascep.cli import main
        assert main(["conformance", sys.argv[1]]) == 0
        assert main(["render", sys.argv[1], "-o", sys.argv[2]]) == 0
        """,
        str(REPORT),
        str(tmp_path / "out.md"),
        bare=True,
    )
    assert done.returncode == 0, done.stderr
    assert (tmp_path / "out.md").read_text(encoding="utf-8").startswith("# ASCEP Capacity Report")


def test_conformance_says_so_when_it_could_not_run_the_schema_check(tmp_path):
    """Degrading silently is the dangerous half of degrading gracefully: a reader would take
    a clean verdict as evidence the report is schema-valid, when in fact nothing looked. The
    skipped check must appear as a finding.
    """
    done = _child(
        """
        import sys
        from ascep.cli import main
        main(["conformance", sys.argv[1]])
        """,
        str(REPORT),
        bare=True,
    )
    assert done.returncode == 0, done.stderr
    assert "schema" in done.stdout.lower()
    assert "verdict: partial (claimed: partial)" in done.stdout


def test_the_readme_cli_transcript_is_what_the_cli_actually_prints():
    """The quick start shows an `ascep size` invocation with its output commented underneath.
    A reader will copy that command; if the numbers beside it drift, the first thing the
    protocol teaches them is that its own documentation cannot be trusted. So run the command
    from the README verbatim and compare.

    This block is also where a unit error would land: the flags take PER-GPU figures, and an
    earlier draft of the README passed the aggregate TP=2 numbers and omitted
    --gpus-per-replica, which silently doubled the answer. Exactly the mistake C3 exists for,
    made in the file that explains C3.
    """
    readme = (ROOT / "README.md").read_text(encoding="utf-8").splitlines()
    start = next(i for i, line in enumerate(readme) if line.startswith("ascep size "))
    command, expected = [], {}
    for line in readme[start:]:
        if line.startswith("#"):
            key, _, value = line.lstrip("# ").partition(": ")
            expected[key] = value
        elif line.startswith("```"):
            break
        else:
            command.append(line.rstrip("\\").strip())
    assert expected, "no commented output found under the README's `ascep size` example"

    argv = " ".join(command).split()
    assert argv[0] == "ascep"
    done = _child("from ascep.cli import main; import sys; sys.exit(main(sys.argv[1:]))", *argv[1:])
    assert done.returncode == 0, done.stderr
    assert _key_values(done.stdout) == expected


# --- agent-profile: measured agent-loop numbers, and the merge that must stay consistent --


def _agent_step(message_id: str, inp: int, out: int) -> dict:
    return {
        "type": "step-finish",
        "messageID": message_id,
        "tokens": {"input": inp, "output": out, "reasoning": 0, "cache": {"read": 0, "write": 0}},
    }


def _agent_tool(message_id: str, start: int, end: int) -> dict:
    return {
        "type": "tool",
        "messageID": message_id,
        "tool": "read",
        "callID": f"c{start}",
        "state": {"status": "completed", "input": {}, "time": {"start": start, "end": end}},
    }


def _agent_export(tmp_path: pathlib.Path, name: str, *, growth: int) -> pathlib.Path:
    """A three-turn tool-calling session whose prompt grows by `growth` tokens per turn.

    Two steps per turn, so a command that counted messages instead of steps would report
    three requests where there were six.
    """
    messages, clock, prompt = [], 0, 4_000
    for turn in range(3):
        mid = f"{name}-m{turn}"
        messages.append(
            {
                "info": {
                    "id": mid,
                    "sessionID": name,
                    "role": "assistant",
                    "time": {"created": clock, "completed": clock + 9_000},
                    "modelID": "m",
                    "providerID": "p",
                },
                "parts": [
                    _agent_step(mid, prompt, 300),
                    _agent_tool(mid, clock + 2_000, clock + 5_000),
                    _agent_step(mid, prompt + growth, 200),
                ],
            }
        )
        clock += 12_000
        prompt += growth
    return _write(tmp_path, f"{name}.json", {"info": {"id": name}, "messages": messages})


def test_agent_profile_counts_requests_as_api_calls_not_as_turns(tmp_path, capsys):
    """A tool-calling turn is several requests, and requests_per_session must say so.

    Counting assistant messages would report 3 where 6 API calls were made, halving demand
    and reporting an agent loop as costing what a chat turn costs.
    """
    export = _agent_export(tmp_path, "s1", growth=1_000)
    assert main(["agent-profile", str(export)]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["requests_per_session"] == 6
    assert out["agent_loop"]["turns_per_session"] == 3
    assert out["archetypes"] == ["code_agent"]


def test_agent_profile_keeps_diagnostics_off_stdout(tmp_path, capsys):
    """stdout must stay one parseable JSON document even when there is plenty to warn about.

    The command is meant to be piped; a warning printed to stdout corrupts the document and
    the failure shows up as a JSON parse error somewhere else entirely.
    """
    export = _agent_export(tmp_path, "s1", growth=1_000)
    assert main(["agent-profile", str(export)]) == 0
    captured = capsys.readouterr()
    json.loads(captured.out)
    assert "session_max_context_tokens is null" in captured.err
    assert "compaction_resume_tokens is null" in captured.err


def test_agent_profile_says_which_export_it_could_not_read(tmp_path, capsys):
    """A campaign profiles a directory at once, so the diagnostic has to name the file."""
    good = _agent_export(tmp_path, "s1", growth=1_000)
    bad = _write(tmp_path, "bad.json", {"nope": 1})
    assert main(["agent-profile", str(good), str(bad)]) == 1
    assert "bad.json" in capsys.readouterr().err


def test_agent_profile_flags_sessions_too_unalike_to_average(tmp_path, capsys):
    """A mean over sessions of different kinds of work describes no session that ran.

    Averaging a 4,000-token session with a 40,000-token one yields a workload nobody
    exercised, and the resulting capacity number is precise about a fiction.
    """
    small = _agent_export(tmp_path, "small", growth=200)
    large = _agent_export(tmp_path, "large", growth=20_000)
    assert main(["agent-profile", str(small), str(large)]) == 0
    assert "spread: input_tokens_per_request" in capsys.readouterr().err


def test_agent_profile_fragment_alone_is_not_a_workload_declaration(tmp_path, capsys):
    """Without --into the output is a fragment, and the help text must not overpromise.

    The workload layer sets additionalProperties false and requires two dozen fields no
    transcript can supply, so a user who pipes the fragment into `validate` should be told
    by this test's existence, not by a confusing schema error.
    """
    export = _agent_export(tmp_path, "s1", growth=1_000)
    assert main(["agent-profile", str(export)]) == 0
    fragment = json.loads(capsys.readouterr().out)
    assert "_provenance" in fragment
    assert "ascep_version" not in fragment


def test_agent_profile_merge_produces_a_declaration_that_still_validates(tmp_path, capsys):
    """--into must emit a complete, schema-valid workload, not a spliced-together dict.

    `_provenance` is not a schema property and the layer forbids extras, so carrying it
    through the merge would turn a working declaration into an invalid one.
    """
    export = _agent_export(tmp_path, "s1", growth=1_000)
    out = tmp_path / "merged.json"
    assert (
        main(
            [
                "agent-profile",
                str(export),
                "--into",
                str(WORKLOAD),
                "--session-max-context-tokens",
                "131072",
                "-o",
                str(out),
            ]
        )
        == 0
    )
    capsys.readouterr()
    merged = json.loads(out.read_text(encoding="utf-8"))
    assert "_provenance" not in merged

    from ascep import validation

    assert validation.validate("workload", merged) == []
    assert merged["agent_loop"]["session_max_context_tokens"] == 131_072


def test_agent_profile_merge_re_derives_the_stored_derived_fields(tmp_path, capsys):
    """Overlaying measured inputs and leaving avg_context_tokens alone is the quiet failure.

    The merged declaration would validate and render while its KV footprint still described
    the chat workload it replaced -- here roughly an order of magnitude too small, which is
    exactly the direction that makes an agent workload look affordable.
    """
    export = _agent_export(tmp_path, "s1", growth=1_000)
    out = tmp_path / "merged.json"
    assert main(["agent-profile", str(export), "--into", str(WORKLOAD), "-o", str(out)]) == 0
    err = capsys.readouterr().err
    assert "recomputed: avg_context_tokens" in err

    base = json.loads(WORKLOAD.read_text(encoding="utf-8"))
    merged = json.loads(out.read_text(encoding="utf-8"))
    model = capacity.Workload(
        **{k: merged[k] for k in _WORKLOAD_KEYS if k in merged and merged[k] is not None}
    )
    assert merged["avg_context_tokens"] == model.avg_context_tokens()
    assert merged["demand_tok_s"] == model.demand_tok_s()
    assert merged["avg_context_tokens"] > base["avg_context_tokens"]


def test_agent_profile_merge_refuses_to_overwrite_a_measured_context_length(tmp_path, capsys):
    """Replacing a measurement with the estimator is the one edit that must never be quiet.

    A publisher who measured avg_context_tokens directly has better evidence than the
    estimator does; silently recomputing it would downgrade their report without saying so.
    """
    base = json.loads(WORKLOAD.read_text(encoding="utf-8"))
    base["avg_context_tokens_tag"] = "M"
    target = _write(tmp_path, "measured.json", base)
    export = _agent_export(tmp_path, "s1", growth=1_000)
    assert main(["agent-profile", str(export), "--into", str(target)]) == 1
    assert "tagged (M)" in capsys.readouterr().err


def test_agent_profile_merge_keeps_a_context_limit_the_transcript_cannot_know(tmp_path, capsys):
    """Re-running without --session-max-context-tokens must not erase a declared ceiling.

    The limit governs when the loop compacts and is a serving choice; wiping it to null on
    a routine re-profile would silently remove the constraint from the declaration.
    """
    base = json.loads(WORKLOAD.read_text(encoding="utf-8"))
    base["agent_loop"] = {
        "turns_per_session": 4,
        "tool_calls_per_turn": 2.0,
        "compaction_resume_tokens": None,
        "session_max_context_tokens": 200_000,
    }
    target = _write(tmp_path, "declared.json", base)
    export = _agent_export(tmp_path, "s1", growth=1_000)
    out = tmp_path / "merged.json"
    assert main(["agent-profile", str(export), "--into", str(target), "-o", str(out)]) == 0
    capsys.readouterr()
    merged = json.loads(out.read_text(encoding="utf-8"))
    assert merged["agent_loop"]["session_max_context_tokens"] == 200_000
    assert merged["agent_loop"]["turns_per_session"] == 3


def test_agent_profile_refuses_to_clobber_an_existing_file_without_force(tmp_path, capsys):
    """The file this would overwrite may be a hand-annotated declaration."""
    export = _agent_export(tmp_path, "s1", growth=1_000)
    out = tmp_path / "workload.json"
    out.write_text("{}", encoding="utf-8")
    assert main(["agent-profile", str(export), "-o", str(out)]) == 2
    assert out.read_text(encoding="utf-8") == "{}"
    assert main(["agent-profile", str(export), "-o", str(out), "--force"]) == 0
    capsys.readouterr()
    assert json.loads(out.read_text(encoding="utf-8"))["archetypes"] == ["code_agent"]
