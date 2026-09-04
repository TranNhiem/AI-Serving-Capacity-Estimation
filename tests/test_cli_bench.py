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
from bench_cli_support import (
    _DROP,
    DECLARED,
    _config,
    _dry_run,
    _report,
    _run_offline,
    _write,
    assert_draft_validates,
)

from ascep.bench import run as bench_run
from ascep.cli import main

pytest.importorskip("httpx", reason="ascep bench needs the [run] extra")

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
        ({"ladder.concurrency": [4, 8]}, "at least 3 rungs"),
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
    collapse silently stops being tested; a two-rung ladder runs to completion and then fails
    draft validation on run.results minItems, reported as a defect in bench rather than as
    the ladder the operator declared; two repetitions are graded INVALID only after the
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


def test_the_reproduction_table_names_the_engine_log_the_bundle_froze(tmp_path, monkeypatch):
    """A report that names an absolute path from the machine that produced it cannot be
    checked by anyone else, and a report that names the live log names a file the server
    keeps changing. Below the snapshot cap the harness copies the log into the bundle and
    publishes the copy, so the path resolves for a reader and the bytes stay put."""
    config_dir = tmp_path / "cfg"
    config_dir.mkdir()
    path = _write(config_dir, _config(config_dir, **{"output.engine_logs_path": "engine.log"}))
    _run_offline(monkeypatch)
    assert main(["bench", path]) == 0
    report = json.loads((config_dir / "report.json").read_text(encoding="utf-8"))
    published = report["reproduction"]["engine_logs_path"]
    assert not pathlib.Path(published).is_absolute()
    assert (config_dir / published).read_text(encoding="utf-8") == "fixture engine log\n"


def test_a_run_whose_server_keeps_logging_still_produces_a_bundle_that_verifies(
    tmp_path, monkeypatch
):
    """The end-to-end form of the defect a GB200 campaign found. Every real run leaves the
    server up -- the next rung needs it -- so the log grows between the bundle being written
    and anyone checking it, and the pre-snapshot harness published a bundle that could never
    verify. Asserted through the CLI because the failure needed the whole path: the C8 check
    forces the log inside the report directory, and the writer then hashed it in place."""
    from ascep.bench.persist import verify_bundle

    config_dir = tmp_path / "cfg"
    config_dir.mkdir()
    path = _write(config_dir, _config(config_dir, **{"output.engine_logs_path": "engine.log"}))
    _run_offline(monkeypatch)
    assert main(["bench", path]) == 0
    with (config_dir / "engine.log").open("a", encoding="utf-8") as fp:
        fp.write("Avg prompt throughput: 0.0 tokens/s, Avg generation throughput: 0.0\n")
    assert verify_bundle(config_dir / "bundle") == []


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
    hand-edit wrongly. The session replay asserts the same thing through the same helper,
    so the two cannot become a strong check and a weaker copy of it."""
    assert_draft_validates(tmp_path, monkeypatch, _config(tmp_path))


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
