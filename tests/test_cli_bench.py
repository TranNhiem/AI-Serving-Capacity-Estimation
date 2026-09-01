"""Acceptance tests for ``ascep bench``, the command that spends the GPU hours.

Every other command in this toolkit reads a file someone already wrote. This one produces the
evidence the rest of the protocol grades, which makes its failure modes categorically worse:
a wrong number here is not caught downstream, it *is* the downstream. Four properties are
therefore asserted harder than the code they cover would suggest.

The run is declared before it happens. Chapter 7 requires the window, the drain deadline, the
think time, the warm-up and the SLO gates to be fixed in advance, so ``bench`` takes a config
file and refuses flags that would let any of them move between repetitions. The config is
copied verbatim into the bundle: an operator who tuned a gate after seeing rung 8 has to have
edited a file that is published next to the results.

The run is bound to a declared machine. A load generator can see latency and nothing else --
not the GPU model, not the tensor-parallel degree, not the business demand the numbers are
supposed to serve. Those four layer documents are inputs to ``bench``, not outputs of it, and
they are schema-checked before the first request goes out. Discovering after four hours of
Slurm time that ``serving.json`` was malformed is discovering it too late, and C3 has no way
to bind a measurement to a topology nobody wrote down.

Nothing is defaulted that the protocol requires you to declare. A missing ``drain_deadline_s``
is an error naming section 7.6, not a quiet 30 seconds. The general rule in this repository is
that a skeleton must not validate, and the command-line equivalent is that an under-specified
config must not run.

It does not grade its own output. ``bench`` emits a run and a measured tier, and claims the
weakest grade the schema will let it claim, because a harness that certifies its own results
is the exact failure the negative corpus in ``examples/negative`` exists to demonstrate.
"""

from __future__ import annotations

import json
import pathlib
import re

import pytest

from ascep.bench import run as bench_run
from ascep.cli import main

pytest.importorskip("httpx", reason="ascep bench needs the [run] extra")

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]

#: The four layer documents bench reads rather than invents, borrowed from the negative
#: corpus baseline because those blocks are known to validate standalone. Any valid pair of
#: hardware/serving documents would do; what these tests care about is that bench refuses to
#: run without them and copies them through unchanged.
DECLARED = json.loads((REPO_ROOT / "examples" / "negative" / "baseline.json").read_text())


# --- a config that is complete, so tests can subtract from it -------------------------


def _config(tmp_path: pathlib.Path, **overrides) -> dict:
    """A fully declared config. Every test that checks a refusal removes one key from this.

    The window is a fraction of a second because these tests run a fake adapter at memory
    speed; the dry-run tests that talk about wall clock override it with a realistic one.
    """
    config = {
        "endpoint": {
            "base_url": "http://127.0.0.1:9",
            "model": "test-model",
            "timeout_s": 30.0,
        },
        "declarations": {
            "hardware": "hardware.json",
            "model": "model.json",
            "serving": "serving.json",
            "workload": "workload.json",
        },
        "workload": {
            "corpus": "synthetic",
            "input_tokens": 512,
            "output_tokens": 128,
            "ignore_eos": True,
            "cache_policy": "unique-prefix",
            "seed": 11,
            "think_time_s": 0.01,
            "run_label": "acceptance",
        },
        "window": {
            "window_s": 0.4,
            "drain_deadline_s": 0.2,
            "warmup_requests": 2,
        },
        "ladder": {
            "concurrency": [1, 2, 4],
            "repetitions": 3,
            "throughput_collapse_ratio": 0.5,
        },
        "slo_gates": {
            "ttft_p95_max_s": 2.0,
            "itl_p95_max_s": 0.15,
            "e2e_p95_max_s": 60.0,
            "error_rate_max_pct": 1.0,
            "declared_before_run": True,
        },
        "output": {
            "bundle_dir": str(tmp_path / "bundle"),
            "report_path": str(tmp_path / "report.json"),
            "engine_logs_path": str(tmp_path / "engine.log"),
            "container_digest": "sha256:" + "b" * 64,
        },
    }
    for dotted, value in overrides.items():
        section, _, key = dotted.partition(".")
        if value is _DROP:
            config[section].pop(key, None)
        else:
            config[section][key] = value
    return config


_DROP = object()


def _write(tmp_path: pathlib.Path, config: dict, **layer_overrides) -> str:
    """Write the config, the four declarations it points at, and a stand-in engine log."""
    for layer in ("hardware", "model", "serving", "workload"):
        document = layer_overrides.get(layer, DECLARED[layer])
        (tmp_path / f"{layer}.json").write_text(json.dumps(document, indent=2), encoding="utf-8")
    (tmp_path / "engine.log").write_text("fixture engine log\n", encoding="utf-8")
    path = tmp_path / "bench.json"
    path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    return str(path)


def _dry_run(tmp_path: pathlib.Path, **overrides):
    path = _write(tmp_path, _config(tmp_path, **overrides))
    return main(["bench", path, "--dry-run"])


def _report(tmp_path: pathlib.Path) -> dict:
    return json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))


# --- the plan is inspectable before any GPU time is spent -----------------------------


def test_dry_run_prints_the_plan_and_exits_zero_without_touching_the_network(tmp_path, capsys):
    """The endpoint in these configs is a closed port. If --dry-run connects, this hangs or
    fails, which is the assertion."""
    assert _dry_run(tmp_path) == 0
    out = capsys.readouterr().out
    assert "1, 2, 4" in out or ("1" in out and "4" in out)


def test_dry_run_states_the_window_count_and_the_wall_clock_it_implies(tmp_path, capsys):
    """Three rungs, three repetitions and the section 5 confirmation, at a 20 s window and a
    10 s drain, is at least 300 s of measured time before warm-up. An operator who learns
    that after submitting the Slurm job learns it too late, and the run that gets killed at
    the wall clock is the one that had the highest concurrency rung in it."""
    _dry_run(tmp_path, **{"window.window_s": 20.0, "window.drain_deadline_s": 10.0})
    out = capsys.readouterr().out
    assert "9" in out, "three rungs x three repetitions is nine graded windows; say so"
    assert "10" in out, "the confirmation repetition is a tenth window; do not hide it"
    numbers = {int(n) for n in re.findall(r"\d+", out)}
    assert any(n >= 300 for n in numbers), f"no wall-clock estimate in: {out}"


def test_dry_run_writes_no_bundle_and_no_report(tmp_path):
    _dry_run(tmp_path)
    assert not (tmp_path / "bundle").exists()
    assert not (tmp_path / "report.json").exists()


# --- an under-specified config does not run -------------------------------------------


@pytest.mark.parametrize(
    ("dropped", "section"),
    [
        ("window.drain_deadline_s", "7.6"),
        ("window.window_s", "7"),
        ("window.warmup_requests", "7.3"),
        ("workload.think_time_s", "7"),
        ("workload.ignore_eos", "7"),
        ("workload.cache_policy", "7"),
        ("workload.seed", "7"),
        ("slo_gates.declared_before_run", "C7"),
        ("declarations.hardware", "C3"),
        ("declarations.serving", "C3"),
    ],
)
def test_a_config_missing_a_declaration_the_protocol_requires_is_refused(
    tmp_path, capsys, dropped, section
):
    """Each of these has an obvious default, and every one of those defaults is a lie about
    what was declared. A drain deadline that appears by itself was never fixed before timing;
    a think time of zero turns a closed loop into an unthrottled one and quietly changes the
    workload being measured; a topology bench guessed from the endpoint is a topology C3
    cannot bind the measurement to."""
    assert _dry_run(tmp_path, **{dropped: _DROP}) != 0
    err = capsys.readouterr().err
    assert dropped.split(".")[-1] in err, f"the error must name the missing key: {err}"
    assert section in err, f"the error must name what requires it ({section}): {err}"


def test_gates_declared_after_the_run_are_refused_rather_than_recorded(tmp_path, capsys):
    """C7 is not a field to fill in honestly and carry on. If the gates were not fixed in
    advance there is no gated measurement to make, and running anyway produces a sustainable
    tier that cannot be published."""
    assert _dry_run(tmp_path, **{"slo_gates.declared_before_run": False}) != 0
    assert "C7" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("override", "expected"),
    [
        ({"ladder.concurrency": [1, 4, 4]}, "increasing"),
        ({"ladder.concurrency": [8, 4, 1]}, "increasing"),
        ({"ladder.repetitions": 2}, "at least 3"),
        ({"ladder.throughput_collapse_ratio": 0.2}, "0.5"),
        ({"ladder.throughput_collapse_ratio": 1.0}, "1.0"),
        ({"endpoint.timeout_s": 0}, "timeout_s"),
    ],
)
def test_a_ladder_that_cannot_be_graded_as_declared_is_refused_before_the_first_request(
    tmp_path, capsys, override, expected
):
    """Each of these produces a number rather than an error if it is left to run. A repeated
    rung pools six repetitions into one operating point and publishes it as three; a
    descending ladder has no lower COMPLETE rung to hold the collapse test against, so
    collapse silently stops being tested; two repetitions are graded INVALID only after the
    GPU hours are spent; a collapse ratio at 1.0 calls the flat top of a healthy throughput
    curve a queueing failure. The grading policy is therefore built during config validation,
    which is the only moment when refusing is still free."""
    assert _dry_run(tmp_path, **override) != 0
    assert expected in capsys.readouterr().err


@pytest.mark.parametrize("written", ["http://h:8000/v1", "http://h:8000/v1/"])
def test_a_base_url_carrying_the_api_route_is_refused_rather_than_doubled(
    tmp_path, capsys, written
):
    """`http://host:8000/v1` is what every OpenAI client example puts in front of a user, and
    the adapter appends the route itself, so this one requests /v1/v1/chat/completions. The
    resulting 404s are scored as server errors rather than as a broken config: the ladder
    climbs every rung against a URL that serves nothing and publishes a 100% error rate as a
    property of the endpoint."""
    assert _dry_run(tmp_path, **{"endpoint.base_url": written}) != 0
    err = capsys.readouterr().err
    assert "/v1/v1/chat/completions" in err, f"the error must show what it would request: {err}"


def test_four_null_slo_gates_are_refused_rather_than_recorded_as_declared(tmp_path, capsys):
    """This is the most dangerous config bench would otherwise accept: C7 is satisfied in
    shape -- the gates were declared before the run, and they are all null -- so every window
    passes by definition, every rung grades COMPLETE, and the sustainable tier it publishes
    is the measured tier wearing an SLO label."""
    nulls = {
        "slo_gates.ttft_p95_max_s": None,
        "slo_gates.itl_p95_max_s": None,
        "slo_gates.e2e_p95_max_s": None,
        "slo_gates.error_rate_max_pct": None,
    }
    assert _dry_run(tmp_path, **nulls) != 0
    assert "C7" in capsys.readouterr().err


def test_an_engine_log_that_is_not_there_is_refused_before_the_run_not_after_it(
    tmp_path, monkeypatch, capsys
):
    """The bundle hashes the engine's own log after the last window, when the records exist
    nowhere but in RAM. A typo in that path discovered then costs the whole run, so it is
    checked while the cost of being wrong is still zero."""
    path = _write(tmp_path, _config(tmp_path, **{"output.engine_logs_path": "nowhere.log"}))
    _run_offline(monkeypatch)
    assert main(["bench", path]) == 2
    err = capsys.readouterr().err
    assert "nowhere.log" in err and "C8" in err


# --- a config is self-contained: outputs land beside it, from any working directory ---


def test_a_run_from_a_foreign_working_directory_writes_beside_the_config_not_the_cwd(
    tmp_path, monkeypatch
):
    """The behaviour being pinned: outputs used to resolve against the process cwd while
    inputs resolved against the config's directory, so a config declaring `bundle` was
    self-contained on exactly one side, and a run launched from anywhere else split its
    bundle from the declarations it was measured against. The run must chdir for real --
    asserting on path strings without a foreign cwd would pass against the old code, which
    was correct precisely when the cwd happened to be the config's directory. The zero
    return also asserts that the C8 engine-log check passes on a relative
    engine_logs_path from that foreign cwd, since that check gates the run."""
    config_dir = tmp_path / "cfg"
    config_dir.mkdir()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    path = _write(
        config_dir,
        _config(
            config_dir,
            **{
                "output.bundle_dir": "bundle",
                "output.report_path": "report.json",
                "output.engine_logs_path": "engine.log",
            },
        ),
    )
    _run_offline(monkeypatch)
    monkeypatch.chdir(elsewhere)
    assert main(["bench", path]) == 0
    assert (config_dir / "bundle").is_dir(), "the bundle belongs beside the config"
    assert (config_dir / "report.json").is_file(), "the draft belongs beside the config"
    assert not (elsewhere / "bundle").exists()
    assert not (elsewhere / "report.json").exists()


def test_the_reproduction_table_names_the_declared_engine_log_not_the_resolved_path(
    tmp_path, monkeypatch
):
    """A report that names an absolute path from the machine that produced it cannot be
    checked by anyone else. The harness needs the resolved path to hash the file; what it
    publishes is the string the operator wrote."""
    config_dir = tmp_path / "cfg"
    config_dir.mkdir()
    path = _write(config_dir, _config(config_dir, **{"output.engine_logs_path": "engine.log"}))
    _run_offline(monkeypatch)
    assert main(["bench", path]) == 0
    report = json.loads((config_dir / "report.json").read_text(encoding="utf-8"))
    assert report["reproduction"]["engine_logs_path"] == "engine.log"


def test_the_c8_refusal_names_both_the_declared_string_and_where_it_resolved(
    tmp_path, monkeypatch, capsys
):
    """The refusal that cost a live run said `calib/vllm_server.out is not a file`, which
    was true of the working directory and false of the config's, and told the operator
    nothing about which was meant. The message must carry the declared string, the path it
    resolved to, and the rule that produced it."""
    config_dir = tmp_path / "cfg"
    config_dir.mkdir()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    path = _write(config_dir, _config(config_dir, **{"output.engine_logs_path": "nowhere.log"}))
    _run_offline(monkeypatch)
    monkeypatch.chdir(elsewhere)
    assert main(["bench", path]) == 2
    err = capsys.readouterr().err
    assert "nowhere.log" in err
    expected = pathlib.Path(path).resolve().parent / "nowhere.log"
    assert str(expected) in err, f"the refusal must say where it actually looked: {err}"
    assert "config file's directory" in err, f"the refusal must state the rule: {err}"


def test_an_absolute_bundle_dir_is_honoured_verbatim_from_a_foreign_working_directory(
    tmp_path, monkeypatch
):
    """Path.joinpath's absolute-wins rule is the only thing standing between a declared
    absolute bundle path and the harness silently relocating it under the config's
    directory. It falls out of the expression, which is exactly why it needs its own
    assertion: a regression here breaks every published config with absolute outputs
    without touching a relative one."""
    config_dir = tmp_path / "cfg"
    config_dir.mkdir()
    target = tmp_path / "absolute-target"
    target.mkdir()
    # Beside the absolute bundle, not beside the config: C8 requires the log to sit under
    # the report directory, and that directory is wherever the absolute path put it.
    (target / "engine.log").write_text("fixture engine log\n", encoding="utf-8")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    path = _write(
        config_dir,
        _config(
            config_dir,
            **{
                "output.bundle_dir": str(target / "bundle"),
                "output.report_path": str(target / "report.json"),
                "output.engine_logs_path": "engine.log",
            },
        ),
    )
    _run_offline(monkeypatch)
    monkeypatch.chdir(elsewhere)
    assert main(["bench", path]) == 0
    assert (target / "bundle").is_dir(), "an absolute bundle_dir must not be relocated"
    assert (target / "report.json").is_file()
    assert not (config_dir / "bundle").exists()
    assert not (elsewhere / "bundle").exists()


def test_an_unknown_key_in_the_config_is_an_error_not_a_shrug(tmp_path, capsys):
    """A typo in `drain_deadline_sec` is otherwise indistinguishable from omitting it, and the
    run proceeds on a default the operator believes they overrode."""
    assert _dry_run(tmp_path, **{"window.drain_deadline_sec": 10.0}) != 0
    assert "drain_deadline_sec" in capsys.readouterr().err


# --- the declarations are checked before the first request ----------------------------


def test_a_declaration_that_fails_its_own_schema_stops_the_run_at_the_dry_run(tmp_path, capsys):
    """Four hours of Slurm time later is the wrong moment to find out that `serving.json` was
    malformed, and the report built on it would fail validation for a cause the operator
    created before the run rather than during it."""
    broken = dict(DECLARED["serving"], tensor_parallel="four")
    path = _write(tmp_path, _config(tmp_path), serving=broken)
    assert main(["bench", path, "--dry-run"]) != 0
    err = capsys.readouterr().err
    assert "serving" in err
    assert "tensor_parallel" in err


def test_a_declaration_file_that_is_not_there_is_named_in_the_error(tmp_path, capsys):
    path = _write(tmp_path, _config(tmp_path, **{"declarations.model": "nowhere.json"}))
    assert main(["bench", path, "--dry-run"]) != 0
    assert "nowhere.json" in capsys.readouterr().err


def test_the_declared_layers_are_copied_into_the_report_unchanged(tmp_path, monkeypatch):
    """Bench measures; it does not edit the operator's declarations on the way past. A
    harness that normalises `serving.json` into the report publishes a topology the operator
    never wrote, and the diff against their own file is where they would have caught it."""
    path = _write(tmp_path, _config(tmp_path))
    _run_offline(monkeypatch)
    assert main(["bench", path]) == 0
    report = _report(tmp_path)
    for layer in ("hardware", "model", "serving", "workload"):
        assert report[layer] == DECLARED[layer]


# --- what it emits, and what it refuses to emit ---------------------------------------


def test_the_config_is_copied_verbatim_into_the_bundle(tmp_path, monkeypatch):
    """Not re-serialised from the parsed objects. A config round-tripped through the loader
    records what the harness understood, and the one failure worth catching here is the
    harness understanding something other than what the operator wrote."""
    config = _config(tmp_path)
    path = _write(tmp_path, config)
    _run_offline(monkeypatch)
    assert main(["bench", path]) == 0
    copied = json.loads((tmp_path / "bundle" / "bench-config.json").read_text(encoding="utf-8"))
    assert copied == config


def test_the_emitted_report_claims_the_floor_and_never_grades_itself_up(tmp_path, monkeypatch):
    """A harness that certifies its own results is the failure examples/negative exists to
    show. The schema requires a `conformance` claim, so bench cannot decline to make one --
    it makes the weakest one instead, and `ascep conformance` is what may raise it.

    Asserted against the checker rather than a literal string: an emitted claim that is merely
    *equal* to today's computed grade becomes an overstatement the day a rule gets stricter,
    and the harness would be shipping OVERSTATED reports without a test noticing.
    """
    from ascep.conformance import check

    path = _write(tmp_path, _config(tmp_path))
    _run_offline(monkeypatch)
    main(["bench", path])
    report = _report(tmp_path)
    assert report["conformance"] == "non-conforming"
    assert not check(report).overstated


def test_the_emitted_report_says_in_its_note_that_it_is_an_ungraded_draft(tmp_path, monkeypatch):
    """The claim alone reads as a verdict on the hardware. The note has to say that the
    sections the harness could not observe are unfilled, or the first reader concludes the
    machine failed rather than that the report is half-written."""
    path = _write(tmp_path, _config(tmp_path))
    _run_offline(monkeypatch)
    main(["bench", path])
    note = _report(tmp_path)["conformance_note"].lower()
    assert "ascep conformance" in note
    assert "draft" in note or "not been graded" in note


def test_the_fields_bench_cannot_know_are_null_with_a_reason_not_absent(tmp_path, monkeypatch):
    """A load generator cannot see the roofline or the sizing policy. C1 says those are nulls
    with reasons, and emitting a report with the keys missing hands the operator a file that
    fails validation for reasons they did not cause."""
    path = _write(tmp_path, _config(tmp_path))
    _run_offline(monkeypatch)
    main(["bench", path])
    report = _report(tmp_path)
    roofline = report["roofline_comparison"]
    assert "decode_tok_s_theoretical" in roofline
    if roofline["decode_tok_s_theoretical"] is None:
        assert roofline.get("decode_tok_s_theoretical_u_reason", "").startswith("(U)")


def test_every_unmeasured_field_bench_leaves_behind_is_listed_as_an_assumption(
    tmp_path, monkeypatch
):
    """Section 7 of the report is the list a reviewer reads to decide what the numbers cannot
    settle. A bench draft has a long one -- no roofline, no sizing policy, one topology -- and
    a harness that emits an empty list has told the reviewer there is nothing to check."""
    path = _write(tmp_path, _config(tmp_path))
    _run_offline(monkeypatch)
    main(["bench", path])
    assumptions = _report(tmp_path)["unmeasured_assumptions"]
    assert assumptions, "a draft with four empty sections has assumptions to declare"
    for entry in assumptions:
        assert entry["field"] and "TODO" not in entry["field"]
        assert entry["impact_if_wrong"] and "TODO" not in entry["impact_if_wrong"]
        assert entry["cost_to_measure"] and "TODO" not in entry["cost_to_measure"]


def test_the_emitted_report_validates_against_the_capacity_report_schema(tmp_path, monkeypatch):
    """The output of the harness is the input of every other command here. A report that
    needs hand-editing before `ascep validate` accepts it is a report most people will
    hand-edit wrongly."""
    from ascep.validation import validate

    path = _write(tmp_path, _config(tmp_path))
    _run_offline(monkeypatch)
    main(["bench", path])
    assert validate("capacity-report", _report(tmp_path)) == []


def test_one_run_result_row_per_concurrency_rung_carrying_that_rungs_numbers(tmp_path, monkeypatch):
    """The ladder is the measurement. A report that publishes only the winning rung throws
    away the shape of the curve, which is the part that says whether the tier is a plateau or
    a cliff edge one request wide."""
    path = _write(tmp_path, _config(tmp_path))
    _run_offline(monkeypatch)
    main(["bench", path])
    results = _report(tmp_path)["run"]["results"]
    assert [row["concurrency"] for row in results] == [1, 2, 4]


def test_the_results_row_publishes_the_servers_token_count_not_the_configured_one(
    tmp_path, monkeypatch
):
    """The row is tagged (M) and C4 binds its throughput figures to the context beside them.
    The configured input_tokens is what was asked for -- with the synthetic corpus it is a
    whitespace word count, which a real tokenizer turns into a quarter more tokens -- so
    copying it into a measured row attributes the throughput to a context nobody served."""
    path = _write(tmp_path, _config(tmp_path))
    _run_offline(monkeypatch, reported_input_tokens=640)
    main(["bench", path])
    rows = _report(tmp_path)["run"]["results"]
    assert [row["input_tokens"] for row in rows] == [640, 640, 640]


def test_a_failed_rungs_row_carries_the_reason_sentence_and_its_section_citation(
    tmp_path, monkeypatch
):
    """outcome and slo_pass answer different questions, so a row can honestly carry one
    verdict beside the other -- but only the reason sentence tells the reader which
    repetition failed the rung and under which rule. Publishing the verdict without the text
    forces the operator to re-derive the grading from records.jsonl, work the reasons
    already did, so the assertion is on the section 5 citation and the worst-served-user
    rule rather than on a merely non-empty list."""
    path = _write(tmp_path, _config(tmp_path, **{"slo_gates.itl_p95_max_s": 0.01}))
    _run_offline(monkeypatch, token_gap_s=0.05)
    assert main(["bench", path]) == 0
    rows = _report(tmp_path)["run"]["results"]
    assert [row["concurrency"] for row in rows] == [1, 2, 4]
    assert [row["outcome"] for row in rows] == ["failed", "failed", "failed"]
    for row in rows:
        reasons = row["reasons"]
        assert isinstance(reasons, list) and reasons
        assert all(isinstance(sentence, str) for sentence in reasons)
        assert any("(section 5)" in sentence for sentence in reasons)
        assert any("worst served user" in sentence for sentence in reasons)


def test_a_complete_rungs_row_omits_reasons_because_there_is_nothing_to_reconcile(
    tmp_path, monkeypatch
):
    """The schema requires reasons only where the outcome needs one, so the healthy run
    proves the conditional shape by asserting the key's absence: an always-present empty
    array would be noise on every complete rung and would bind nothing."""
    path = _write(tmp_path, _config(tmp_path))
    _run_offline(monkeypatch)
    assert main(["bench", path]) == 0
    rows = _report(tmp_path)["run"]["results"]
    assert [row["outcome"] for row in rows] == ["complete", "complete", "complete"]
    assert all("reasons" not in row for row in rows)


def test_a_failed_row_that_declines_to_say_why_does_not_validate(tmp_path, monkeypatch):
    """A rung that says "capacity ends here" and declines to say why is the exact skeleton
    this protocol refuses to accept. The guard goes through the real validator on a report
    bench actually emitted -- once with the key stripped and once with it emptied, because
    "absent" and "an empty array" are the same refusal."""
    from ascep.validation import validate

    path = _write(tmp_path, _config(tmp_path, **{"slo_gates.itl_p95_max_s": 0.01}))
    _run_offline(monkeypatch, token_gap_s=0.05)
    main(["bench", path])
    report = _report(tmp_path)
    assert validate("capacity-report", report) == []
    without_reasons = json.loads(json.dumps(report))
    for row in without_reasons["run"]["results"]:
        row.pop("reasons", None)
    assert validate("capacity-report", without_reasons) != []
    with_empty_reasons = json.loads(json.dumps(report))
    for row in with_empty_reasons["run"]["results"]:
        row["reasons"] = []
    assert validate("capacity-report", with_empty_reasons) != []


def test_every_window_in_the_bundle_is_labelled_with_the_repetition_it_was(tmp_path, monkeypatch):
    """Section 7.5 grades a rung on the dispersion across its repetitions. Windows that all
    claim to be repetition 0 make that dispersion unrecoverable from the bundle, and the
    labelling is the only thing that survives into someone else's re-analysis.

    The tenth window is the section 5 confirmation at the boundary rung, numbered one past
    the counted three: grading partitions it out by its `post_search` flag, but the index is
    what names the window in the bundle and salts its request ids, and a fourth window
    numbered 0 would collide with the first."""
    path = _write(tmp_path, _config(tmp_path))
    _run_offline(monkeypatch)
    main(["bench", path])
    configs = json.loads((tmp_path / "bundle" / "run_configs.json").read_text(encoding="utf-8"))
    seen = [(w["policy"]["concurrency"], w["policy"]["repetition"]) for w in configs["windows"]]
    assert seen == [(c, r) for c in (1, 2, 4) for r in (0, 1, 2)] + [(4, 3)]


# --- the boundary rung is confirmed after the search, or no tier is published ----------


def test_the_boundary_rung_is_re_run_once_after_the_search_that_selected_it(
    tmp_path, monkeypatch, capsys
):
    """Section 5 will not publish a Sustainable figure on the strength of the search that
    found it. The boundary is the rung the stopping rule stopped at *because* it passed, so
    the three windows behind it are the evidence that selection conditioned on. One further
    repetition, taken when no decision depends on its outcome, is the difference between the
    rung the search landed on and a rung the system sustains."""
    path = _write(tmp_path, _config(tmp_path))
    _run_offline(monkeypatch)
    assert main(["bench", path]) == 0
    assert "confirmation at concurrency=4" in capsys.readouterr().err


def test_a_confirmed_boundary_is_published_as_the_sustainable_tier(tmp_path, monkeypatch):
    """The four tiers are the point of the report, and Sustainable is the only one a load
    generator can produce. A harness that ran the ladder, passed every gate, confirmed the
    boundary and still published nothing would have spent the GPU hours for a measured tier
    that ignores the SLOs."""
    path = _write(tmp_path, _config(tmp_path))
    _run_offline(monkeypatch)
    main(["bench", path])
    tier = _report(tmp_path)["capacity_tiers"]["sustainable"]
    assert tier["max_concurrent_users"] == 4
    assert tier["provenance"] == "M"


def test_the_confirmation_window_is_not_counted_as_one_of_the_declared_repetitions(
    tmp_path, monkeypatch
):
    """Section 6 wants three independent repetitions and section 5 wants a fourth taken
    afterwards; a harness that let the confirmation stand in for one of the three would
    satisfy neither. The rung's published row has to be the median of the counted windows,
    the same population its graded throughput came from."""
    path = _write(tmp_path, _config(tmp_path))
    _run_offline(monkeypatch)
    main(["bench", path])
    report = _report(tmp_path)
    assert report["run"]["repeats"] == 3
    assert [row["concurrency"] for row in report["run"]["results"]] == [1, 2, 4]


def test_an_unconfirmed_boundary_publishes_no_sustainable_tier_and_says_why(tmp_path, monkeypatch):
    """The interrupt lands before the confirmation, so the ladder has three passing windows
    at every rung and no post-search evidence for any of them. Publishing the top rung anyway
    would turn `the highest rung we happened to reach` into `the highest rung that works`,
    which is the most flattering single sentence a capacity report can contain."""
    path = _write(tmp_path, _config(tmp_path))
    _run_offline(monkeypatch, interrupt_after_s=1.0)
    main(["bench", path])
    tier = _report(tmp_path)["capacity_tiers"]["sustainable"]
    assert tier["max_concurrent_users"] is None
    assert tier["max_concurrent_users_u_reason"].startswith("(U)")


def test_the_four_declarations_are_hashed_into_the_bundle_beside_the_config(tmp_path, monkeypatch):
    """The report carries parsed copies of the declarations, and a parsed copy is not
    evidence: a reader holding the report cannot tell whether the hardware block it claims to
    have been measured on is the one that was declared before the run. Covered by the
    manifest, C3's binding stays checkable after publication."""
    from ascep.bench.persist import verify_bundle

    path = _write(tmp_path, _config(tmp_path))
    _run_offline(monkeypatch)
    main(["bench", path])
    copied = tmp_path / "bundle" / "declarations" / "serving.json"
    assert json.loads(copied.read_text(encoding="utf-8")) == DECLARED["serving"]
    copied.write_text(json.dumps(dict(DECLARED["serving"], gpu_count=8)), encoding="utf-8")
    assert any("serving.json" in problem for problem in verify_bundle(tmp_path / "bundle"))


def test_the_bundle_it_writes_verifies(tmp_path, monkeypatch):
    from ascep.bench.persist import verify_bundle

    path = _write(tmp_path, _config(tmp_path))
    _run_offline(monkeypatch)
    main(["bench", path])
    assert verify_bundle(tmp_path / "bundle") == []


def test_the_copied_config_is_covered_by_the_bundle_manifest(tmp_path, monkeypatch):
    """An artifact in the bundle that the manifest does not hash is an artifact anyone can
    edit after publication without `verify_bundle` saying a word, and the config is the one
    artifact where a single edited number rewrites what the run claims to have been."""
    from ascep.bench.persist import verify_bundle

    path = _write(tmp_path, _config(tmp_path))
    _run_offline(monkeypatch)
    main(["bench", path])
    copied = tmp_path / "bundle" / "bench-config.json"
    copied.write_text(copied.read_text(encoding="utf-8").replace("0.4", "9.9"), encoding="utf-8")
    assert any("bench-config.json" in problem for problem in verify_bundle(tmp_path / "bundle"))


def test_it_refuses_to_overwrite_an_existing_bundle(tmp_path, monkeypatch, capsys):
    """The GPU hours are already spent and the records cannot be regenerated."""
    path = _write(tmp_path, _config(tmp_path))
    _run_offline(monkeypatch)
    assert main(["bench", path]) == 0
    assert main(["bench", path]) != 0
    assert "bundle" in capsys.readouterr().err.lower()


# --- the run is survivable ------------------------------------------------------------


def test_a_scheduler_kill_arrives_as_an_interrupt_rather_than_ending_the_process():
    """These runs are submitted as batch jobs, and both ways they end early -- `scancel` and
    the job's wall clock -- arrive as SIGTERM, whose default action ends the process where it
    stands. Every completed window is in RAM at that moment, so the default action turns
    hours of real measurement into nothing at all. The interrupt path already bundles what
    finished; the whole fix is to arrive on it."""
    import os
    import signal
    import time

    from ascep.bench import run as bench_run

    before = signal.getsignal(signal.SIGTERM)
    with bench_run._sigterm_as_interrupt():
        with pytest.raises(KeyboardInterrupt):
            os.kill(os.getpid(), signal.SIGTERM)
            for _ in range(200):
                time.sleep(0.001)
    assert signal.getsignal(signal.SIGTERM) is before, "the handler outlived the run"


def test_an_interrupted_ladder_still_writes_the_rungs_that_finished(tmp_path, monkeypatch):
    """The worst outcome is not a failed run, it is a run that succeeded and left nothing
    behind. A ladder killed at the wall clock has real measurements for its lower rungs, and
    those are the rungs most likely to be sustainable anyway."""
    path = _write(tmp_path, _config(tmp_path))
    _run_offline(monkeypatch, interrupt_after_s=1.0)
    main(["bench", path])
    records = (tmp_path / "bundle" / "records.jsonl").read_text(encoding="utf-8")
    assert records.strip(), "the completed windows were lost"


def test_an_interrupted_ladder_says_the_result_is_a_lower_bound(tmp_path, monkeypatch):
    """A ladder that stopped early found the highest concurrency it *reached*, not the
    highest that works. Published without that caveat it understates the hardware, and the
    next person orders GPUs against it."""
    path = _write(tmp_path, _config(tmp_path))
    _run_offline(monkeypatch, interrupt_after_s=1.0)
    main(["bench", path])
    blob = json.dumps(_report(tmp_path)).lower()
    assert "lower bound" in blob or "lower_bound" in blob


# --- offline stand-in for a server ----------------------------------------------------


def _run_offline(
    monkeypatch,
    interrupt_after_s: float | None = None,
    reported_input_tokens: int = 512,
    token_gap_s: float = 0.0005,
):
    """Replace the adapter with one that answers instantly, so the suite needs no server.

    Patched at the adapter boundary rather than at the transport, because the point of these
    tests is the assembly -- config to ladder to bundle to report -- and a fake transport
    would drag HTTP framing into tests that are not about it.

    ``interrupt_after_s`` is a deadline rather than a request count on purpose: a count that
    lands mid-ladder on one machine lands before the first window completes on a slower one,
    and the test would then assert that an empty bundle is a bundle.

    ``token_gap_s`` widens the simulated inter-token gap so a test can cross a declared ITL
    gate; it moves timestamps only, never the order or the outcome of a request, so the
    default keeps every existing run exactly as green as it was.
    """
    import time

    import ascep.cli as cli
    from ascep.bench.records import Outcome, RequestRecord

    started = time.monotonic()

    class _Fake:
        name = "fake"

        def __init__(self, *a, **k):
            pass

        async def aclose(self):
            pass

        async def issue(self, spec, *, clock, sink=None):
            if interrupt_after_s is not None and time.monotonic() - started > interrupt_after_s:
                raise KeyboardInterrupt
            t = clock()
            return RequestRecord(
                request_id=spec.request_id,
                issued_ts=t,
                outcome=Outcome.OK,
                first_token_ts=t + 0.001,
                token_ts=[t + 0.001 + token_gap_s * i for i in range(8)],
                end_ts=t + 0.005 + token_gap_s * 7,
                output_tokens=spec.max_tokens or 128,
                input_tokens=reported_input_tokens,
            )

    monkeypatch.setattr(cli, "_bench_adapter", lambda config: _Fake(), raising=False)


def _media_corpus(tmp_path: pathlib.Path) -> None:
    """Write a one-record multimodal corpus and the image it names, as real files.

    Nothing here decodes the JPEG; the point is that the loader resolves and touches the
    same relative paths an operator's corpus would name.
    """
    (tmp_path / "images").mkdir()
    (tmp_path / "images" / "a.jpg").write_bytes(b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\xff\xd9")
    record = {
        "image": "images/a.jpg",
        "width": 1920,
        "height": 1080,
        "conversations": [
            {"from": "human", "value": "<image>\nWhat is happening?"},
            {"from": "gpt", "value": "A lathe."},
        ],
        "id": "r1",
    }
    (tmp_path / "corpus.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")


def _media_workload(tmp_path: pathlib.Path, **overrides):
    """Build the workload for the one-record corpus, through the real config path."""
    _media_corpus(tmp_path)
    config = _config(
        tmp_path,
        **{
            "workload.corpus": "corpus.jsonl",
            "workload.media_root": ".",
            "workload.media_max_records": None,
            **overrides,
        },
    )
    return bench_run._build_workload(config, str(tmp_path))


# --- the text path every published run came through does not move ---------------------


def test_a_text_only_config_still_builds_the_plain_workload_and_a_string_prompt(tmp_path):
    """Every published run in this repository was produced by the text path, and the
    optional-key split touched the validator every one of those configs goes through. A
    change that made those configs invalid, or that quietly altered the RequestSpec they
    generate, would rewrite results that are already citable."""
    workload = bench_run._build_workload(_config(tmp_path), str(tmp_path))
    assert type(workload).__name__ == "Workload"
    assert "media_shape" not in workload.manifest()
    content = workload.for_repetition(0)(0).messages[0]["content"]
    assert isinstance(content, str)


# --- a media run declares its measured media shape ------------------------------------


def test_a_media_run_carries_the_measured_media_shape_in_its_manifest(tmp_path):
    """C4 requires images_per_request and its kin beside any throughput figure, and the
    only version of those numbers that is not someone's recollection is the one measured
    off the corpus. An absent key and a zeroed one say different things, which is why the
    text run has no media_shape at all rather than a zeroed one. media_bytes_resident is
    derived from the rendered request itself, so the assertion stays exact without
    pinning a number that depends on the fixture's encoding."""
    workload = _media_workload(tmp_path)
    assert type(workload).__name__ == "_MediaShapeWorkload"
    assert type(workload.source).__name__ == "MultimodalJsonlCorpus"
    content = workload.for_repetition(0)(0).messages[0]["content"]
    image_part = next(part for part in content if part["type"] == "image_url")
    expected_resident = len(image_part["image_url"]["url"])
    assert workload.manifest()["media_shape"] == {
        "images_per_request": 1.0,
        "videos_per_request": 0.0,
        "image_resolution_mix": [{"width": 1920, "height": 1080, "share": 1.0}],
        "records": 1,
        "records_with_reasoning": 0,
        "media_bytes_resident": expected_resident,
    }
    assert workload.manifest()["media_placeholders_stripped"] is False


def test_a_media_request_sends_the_image_as_base64_and_strips_the_marker(tmp_path):
    """The request that reaches the server is the workload being measured. If the marker
    survived into the text part, or the image part went missing, the ladder would publish
    a media run that was really a text run with extra tokens."""
    workload = _media_workload(tmp_path)
    content = workload.for_repetition(0)(0).messages[0]["content"]
    assert isinstance(content, list)
    assert [part["type"] for part in content] == ["text", "image_url"]
    assert content[0]["text"].startswith("upx-")
    assert "<image>" not in content[0]["text"]
    assert content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_url_transport_sends_the_declared_prefix_and_never_reads_the_image_bytes(tmp_path):
    """With url transport the server fetches the media, so the client has no business
    reading the file after the corpus is loaded. A client that still reads the bytes is
    paying the base64 cost while claiming the url one."""
    workload = _media_workload(
        tmp_path,
        **{
            "workload.image_input_transport": "url",
            "workload.media_url_prefix": "http://h/m/",
        },
    )
    (tmp_path / "images" / "a.jpg").unlink()
    content = workload.for_repetition(0)(0).messages[0]["content"]
    assert content[1]["image_url"]["url"] == "http://h/m/images/a.jpg"


# --- media misdeclarations are refused while refusing is still free -------------------


def test_media_root_on_a_synthetic_corpus_is_refused_as_a_text_run_under_a_media_label(
    tmp_path, capsys
):
    """That config asks for a media run and would otherwise get a text run, publishing
    text numbers under a media label."""
    assert _dry_run(tmp_path, **{"workload.media_root": "."}) != 0
    err = capsys.readouterr().err
    assert "'media_root' is set but 'corpus' is 'synthetic'" in err
    assert "would silently measure a text one" in err


def test_a_media_root_that_is_not_a_directory_is_refused_before_the_first_request(tmp_path, capsys):
    """A run that cannot find its images does not fail, it measures something else.
    Refusing before the first request is the whole point, because the alternative is
    discovering it in a report after the GPU hours are spent."""
    _media_corpus(tmp_path)
    overrides = {
        "workload.corpus": "corpus.jsonl",
        "workload.input_tokens": None,
        "workload.media_root": "nowhere",
    }
    assert _dry_run(tmp_path, **overrides) != 0
    err = capsys.readouterr().err
    assert "workload media_root is not an existing directory" in err
    assert "measures a text workload under a media label" in err


def test_url_transport_without_a_url_prefix_is_refused_as_a_url_that_resolves_to_nothing(
    tmp_path, capsys
):
    """The server fetches the media from that base URL. Without it every request carries
    a URL that resolves to nothing, and the run measures 404s rather than images."""
    overrides = {
        "workload.corpus": "corpus.jsonl",
        "workload.input_tokens": None,
        "workload.media_root": ".",
        "workload.image_input_transport": "url",
    }
    _media_corpus(tmp_path)
    assert _dry_run(tmp_path, **overrides) != 0
    err = capsys.readouterr().err
    assert "'image_input_transport' is 'url' but 'media_url_prefix' is not set" in err


def test_a_url_prefix_with_base64_transport_is_refused_because_it_would_be_ignored(
    tmp_path, capsys
):
    """The key would be silently ignored, and a config whose keys do not all take effect
    is a config the operator misread."""
    _media_corpus(tmp_path)
    overrides = {
        "workload.corpus": "corpus.jsonl",
        "workload.input_tokens": None,
        "workload.media_root": ".",
        "workload.media_url_prefix": "http://h/m/",
    }
    assert _dry_run(tmp_path, **overrides) != 0
    err = capsys.readouterr().err
    assert "'media_url_prefix' is set but 'image_input_transport' is 'base64'" in err


@pytest.mark.parametrize(
    ("override", "expected"),
    [
        (
            {"workload.media_root": 5},
            "config key 'media_root' (section 'workload') must be str or null; got 5",
        ),
        (
            {"workload.media_max_records": "10"},
            "config key 'media_max_records' (section 'workload') must be int or null; got '10'",
        ),
    ],
)
def test_a_media_key_with_the_wrong_json_type_is_refused_with_its_citation(
    tmp_path, capsys, override, expected
):
    """A string that looks like a number is not a number, and accepting it would let the
    config declare one thing while the run does another."""
    assert _dry_run(tmp_path, **override) != 0
    err = capsys.readouterr().err
    assert expected in err
    assert "section 9" in err


def test_an_unknown_workload_key_is_still_an_error_after_the_optional_keys_were_added(
    tmp_path, capsys
):
    """The optional-key split widened the permitted set, and the risk it introduces is
    that a typo becomes a run nobody declared. The refusal has to survive the widening."""
    assert _dry_run(tmp_path, **{"workload.bogus": 1}) != 0
    err = capsys.readouterr().err
    assert "unknown key 'bogus' in section 'workload' of the bench config" in err
    assert "expected keys:" in err
    assert "media_root" in err


def test_a_missing_required_workload_key_is_still_named_after_the_optional_keys_were_added(
    tmp_path, capsys
):
    """The required loop is the one every published config was validated against. Adding
    optional keys must not turn a missing cache_policy into an accepted default."""
    overrides = {"workload.cache_policy": _DROP, "workload.media_root": "."}
    assert _dry_run(tmp_path, **overrides) != 0
    err = capsys.readouterr().err
    assert "bench config is missing 'cache_policy' (section 'workload')" in err
    assert "section 7 requires it" in err


# --- input_tokens is consumed only by the synthetic corpus -----------------------------


def test_input_tokens_on_a_jsonl_corpus_is_refused_because_nothing_would_check_it(tmp_path, capsys):
    """The corpus's own records fix the prompt length, so a declared number beside a real
    corpus is read, type-checked and range-checked and then used for nothing: a published
    config can claim 4096 tokens over a corpus averaging 722 and no rule fires. A warning
    in a scrollback is not visible to a reader of the published config, so the refusal
    names the corpus and says where the number belongs."""
    overrides = {"workload.corpus": "corpus.jsonl", "workload.input_tokens": 722}
    assert _dry_run(tmp_path, **overrides) != 0
    err = capsys.readouterr().err
    assert "corpus.jsonl" in err, f"the error must name the corpus path: {err}"
    assert "workload declaration" in err, f"the error must say where the number belongs: {err}"


def test_input_tokens_null_on_a_jsonl_corpus_passes_this_rule_and_fails_only_elsewhere(
    tmp_path, capsys
):
    """Null is the declaration that the corpus, not the config, fixes the prompt length.
    This config names a corpus file that does not exist, so it must get all the way to the
    corpus-existence check -- proving the null itself was not refused."""
    overrides = {"workload.corpus": "nowhere.jsonl", "workload.input_tokens": None}
    assert _dry_run(tmp_path, **overrides) != 0
    err = capsys.readouterr().err
    assert "'input_tokens' must be null" not in err, f"the null was refused: {err}"
    assert "corpus file not found" in err, f"the run stopped before the corpus check: {err}"


def test_a_synthetic_corpus_with_input_tokens_null_is_refused_because_it_sizes_the_corpus(
    tmp_path, capsys
):
    """SyntheticCorpus is generated to exactly the length this key declares, so a null
    leaves nothing to build the prompts from; the refusal has to say the synthetic corpus
    has no other source for its length, or an operator reads it as a type complaint rather
    than as a missing measurement."""
    assert _dry_run(tmp_path, **{"workload.input_tokens": None}) != 0
    err = capsys.readouterr().err
    assert "no other source" in err, f"the error must say why null cannot stand: {err}"


# --- output_tokens and ignore_eos encode three states, and none of them silently ------


def test_ignore_eos_false_with_a_declared_length_puts_that_ceiling_on_every_request(tmp_path):
    """The live defect's exact config: output_tokens: 512, ignore_eos: false.

    The old two-state build read the 512, type-checked it, range-checked it and then
    constructed the uncapped plan, so the request went out with no cap at all -- and one
    degenerate generation ate an entire 90-second window, collapsing the concurrency-1
    rung's output throughput 9x across repetitions. This is the assertion whose absence
    let that through."""
    config = _config(tmp_path, **{"workload.ignore_eos": False, "workload.output_tokens": 512})
    workload = bench_run._build_workload(config, str(tmp_path))
    spec = workload.for_repetition(0)(0)
    assert spec.max_tokens == 512
    assert "ignore_eos" not in spec.extra


def test_ignore_eos_true_with_no_length_is_refused_as_an_instruction_to_generate_forever(
    tmp_path, capsys
):
    """ignore_eos removes the model's only way to stop on its own, so with no length it
    asks the server to generate until the context limit on every single request. That is
    a decode storm wearing a benchmark's name, and the refusal must say so."""
    assert _dry_run(tmp_path, **{"workload.output_tokens": None}) != 0
    err = capsys.readouterr().err
    assert "context limit" in err
    assert "every single request" in err


def test_ignore_eos_false_with_a_null_length_is_accepted_and_sends_no_max_tokens(tmp_path):
    """Uncapped is a legal measurement as long as it was declared: false with a null
    output_tokens puts no length on the wire at all, and nothing between the config and
    the adapter may grow a cap the operator did not ask for."""
    overrides = {"workload.ignore_eos": False, "workload.output_tokens": None}
    assert _dry_run(tmp_path, **overrides) == 0
    workload = bench_run._build_workload(_config(tmp_path, **overrides), str(tmp_path))
    assert workload.for_repetition(0)(0).max_tokens is None


def test_the_results_row_publishes_the_measured_context_length_not_the_configured_request_shape(
    tmp_path, monkeypatch
):
    """C4 binds every throughput figure in the row to the context beside it, and the only
    context length that is not a request is the one the server counted. The configured
    512 words of synthetic filler tokenise to more than 512 tokens, so a row carrying the
    configured 640 (512 + 128) would publish its throughput against a context no request
    was served at -- while the server actually accounted 768 per request."""
    path = _write(tmp_path, _config(tmp_path))
    _run_offline(monkeypatch, reported_input_tokens=640)
    assert main(["bench", path]) == 0
    rows = _report(tmp_path)["run"]["results"]
    assert [row["context_tokens"] for row in rows] == [768.0, 768.0, 768.0]


def test_a_rung_the_server_never_counted_leaves_the_context_out_rather_than_inventing_one(
    tmp_path, monkeypatch
):
    """`run.results[]` carries an anyOf, not a required pair: `ascep init` reports it as a
    decision -- context_tokens OR input_tokens -- and the skeleton emits neither, so
    neither has a `_u_reason` companion and `_unknown` will not invent one. A server that
    counts nothing therefore leaves bench with no branch it can satisfy, and the honest
    outcome is the refusal it already prints, not a context length back-computed from the
    request shape. This pins the absence: filling the key from the config would put a
    number in a row tagged (M) that no request was served at."""
    path = _write(tmp_path, _config(tmp_path))
    _run_offline_with_usage(monkeypatch, lambda index: (None, None))
    assert main(["bench", path]) == 3, (
        "an unvalidatable draft must be reported, not shipped quietly"
    )
    rows = _report(tmp_path)["run"]["results"]
    assert all("context_tokens" not in row for row in rows)


def test_the_context_mean_is_taken_over_complete_records_not_as_a_sum_of_two_means(
    tmp_path, monkeypatch
):
    """Half the records report both counts (600 + 100) and half report an input count with
    no output count (800, None). mean(all inputs) + mean(all outputs) is 700 + 100 = 800,
    a context length no request ever occupied; the mean over the per-record sums of the
    complete records is 700. Only 700 may be published in a row tagged (M) -- this is the
    assertion that fails if the helper is ever "simplified" into adding two means."""
    path = _write(tmp_path, _config(tmp_path))
    _run_offline_with_usage(
        monkeypatch, lambda index: (600, 100) if index % 2 == 0 else (800, None)
    )
    assert main(["bench", path]) == 0
    rows = _report(tmp_path)["run"]["results"]
    assert all(row["context_tokens"] == 700.0 for row in rows)


def _run_offline_with_usage(monkeypatch, usage_for):
    """Patch the adapter as _run_offline does, with per-request usage from ``usage_for``.

    ``usage_for`` receives the request ordinal, counting from zero across the whole ladder,
    and returns the ``(input_tokens, output_tokens)`` the record will carry; ``None`` on
    either side is a server that answered the request without counting that side.
    """
    import ascep.cli as cli
    from ascep.bench.records import Outcome, RequestRecord

    issued = {"n": 0}

    class _Fake:
        name = "fake"

        def __init__(self, *a, **k):
            pass

        async def aclose(self):
            pass

        async def issue(self, spec, *, clock, sink=None):
            index = issued["n"]
            issued["n"] = index + 1
            input_tokens, output_tokens = usage_for(index)
            t = clock()
            return RequestRecord(
                request_id=spec.request_id,
                issued_ts=t,
                outcome=Outcome.OK,
                first_token_ts=t + 0.001,
                token_ts=[t + 0.001 + 0.0005 * i for i in range(8)],
                end_ts=t + 0.005 + 0.0005 * 7,
                output_tokens=output_tokens,
                input_tokens=input_tokens,
            )

    monkeypatch.setattr(cli, "_bench_adapter", lambda config: _Fake(), raising=False)
