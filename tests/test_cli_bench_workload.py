"""What ``ascep bench`` sends, given what the config declared it should send.

Split out of ``test_cli_bench``, which grades the command's contract -- refuse early, write
beside the config, never grade yourself. These grade the layer under it: how the seven
optional ``workload`` keys turn a corpus into traffic, and how the three ``output_tokens``
and ``ignore_eos`` states reach the wire. The two are separate subjects and the second is
where the defects have been, because it is the only place a run can measure something other
than what its config says it measured and still look green.
"""

from __future__ import annotations

import json
import pathlib
import re

import pytest
from bench_cli_support import (
    _DROP,
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
    assert type(workload).__name__ == "MediaShapeWorkload"
    assert type(workload.source).__name__ == "MultimodalJsonlCorpus"
    content = workload.for_repetition(0)(0).messages[0]["content"]
    image_part = next(part for part in content if part["type"] == "image_url")
    expected_resident = len(image_part["image_url"]["url"])
    assert workload.manifest()["media_shape"] == {
        "images_per_request": 1.0,
        "videos_per_request": 0.0,
        "image_resolution_mix": [{"width": 1920, "height": 1080, "share": 1.0}],
        "image_resolution_mix_distinct": 1,
        "image_resolution_mix_listed_share": 1.0,
        "image_resolution_mix_coverage": 1.0,
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


def test_a_text_corpus_reads_the_declared_prompt_field_and_not_a_hardcoded_one(tmp_path):
    """'prompt_field' used to be honoured only when 'media_root' was set; a text corpus was
    pinned to "messages", which no protocol section names and no published config carries.
    Every post-training corpus nests its prompt under "conversations", so a text campaign
    against a real dataset died at load -- and an operator who declared the right field had
    the declaration silently ignored."""
    corpus = tmp_path / "text.jsonl"
    corpus.write_text(
        json.dumps({"turns": [{"from": "human", "value": "declared prompt"}]}) + "\n",
        encoding="utf-8",
    )
    config = _config(
        tmp_path,
        **{
            "workload.corpus": str(corpus),
            "workload.input_tokens": None,
            "workload.cache_policy": "declared-workload",
            "workload.prompt_field": "turns",
        },
    )
    workload = bench_run._build_workload(config, str(tmp_path))
    assert workload.source.field == "turns"
    assert workload.for_repetition(0)(0).messages[0]["content"] == "declared prompt"


def test_a_text_corpus_with_no_prompt_field_declared_defaults_to_conversations(tmp_path):
    """The default the optional-key table documents. It had documented "conversations" while
    the code used "messages" the whole time, so the table described a config that did not
    exist."""
    corpus = tmp_path / "text.jsonl"
    corpus.write_text(
        json.dumps({"conversations": [{"from": "human", "value": "default prompt"}]}) + "\n",
        encoding="utf-8",
    )
    config = _config(
        tmp_path,
        **{
            "workload.corpus": str(corpus),
            "workload.input_tokens": None,
            "workload.cache_policy": "declared-workload",
        },
    )
    workload = bench_run._build_workload(config, str(tmp_path))
    assert workload.source.field == "conversations"
    assert workload.for_repetition(0)(0).messages[0]["content"] == "default prompt"


def test_stripping_media_placeholders_from_a_media_run_is_refused(tmp_path, capsys):
    """The key declares the text-only variant of a media-bearing corpus. Accepting it
    alongside 'media_root' would let a report claim the media was stripped from a run that
    sent every image."""
    corpus = tmp_path / "mm.jsonl"
    corpus.write_text(
        json.dumps({"conversations": [{"from": "human", "value": "<image> q"}], "image": ["a.png"]})
        + "\n",
        encoding="utf-8",
    )
    assert (
        _dry_run(
            tmp_path,
            **{
                "workload.corpus": str(corpus),
                "workload.input_tokens": None,
                "workload.media_root": ".",
                "workload.strip_media_placeholders": True,
            },
        )
        != 0
    )
    err = capsys.readouterr().err
    assert "'strip_media_placeholders' is true" in err


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


# --- a captured agent session, replayed (section 10.8) --------------------------------


def _shapes_file(
    tmp_path: pathlib.Path,
    sessions: list[dict] | None = None,
    *,
    version: int = 1,
    shared_prefix_tokens: int = 0,
    session_count: int = 1,
    steps: int = 3,
) -> str:
    """Write a captured-shapes file for the replay tests and return its path as a string.

    The default capture is the smallest thing that is still a session: several turns whose
    prompts grow. A one-step "session", or one whose prompts were all the same size, would
    pass every shape check while carrying none of the growth the replay exists to measure,
    so a test built on it could go green against a replay that had flattened the shape.

    Most callers here never get as far as reading the file -- the refusals they pin fire
    before the shapes are loaded -- but `bench` rejects a 'corpus' naming no real file
    before it checks anything else, and a config refused on the path would prove nothing
    about the rule under test.
    """
    if sessions is None:
        sessions = [
            {
                "session_id": f"capture-{session_index}",
                "steps": [
                    {
                        "turn_index": step_index,
                        "prompt_tokens": 8 + step_index * 4,
                        "output_tokens": 4,
                        "gap_s": 0.0,
                        "resets_prefix": False,
                    }
                    for step_index in range(steps)
                ],
            }
            for session_index in range(session_count)
        ]
    document = {
        "ascep_shapes_version": version,
        "shared_prefix_tokens": shared_prefix_tokens,
        "sessions": sessions,
    }
    path = tmp_path / "shapes.json"
    path.write_text(json.dumps(document, indent=2), encoding="utf-8")
    return str(path)


def _replay_config(tmp_path: pathlib.Path, shapes: str, **overrides) -> dict:
    """A config that declares the replay honestly, so a test's own override is the only
    thing left for the harness to trip over.

    Every key here is one the replay refuses when it is declared: a test that forgot one
    would be graded by whichever refusal happens to run first, and would keep passing after
    the rule it was written for had been deleted.
    """
    return _config(
        tmp_path,
        **{
            "workload.corpus": shapes,
            "workload.replay_sessions": True,
            "workload.input_tokens": None,
            "workload.output_tokens": None,
            "workload.ignore_eos": False,
            "workload.think_time_s": 0.0,
            "workload.cache_policy": "declared-workload",
            **overrides,
        },
    )


def _recording_adapter(monkeypatch, issued: list) -> None:
    """Answer every request instantly while keeping the traffic shape it was handed.

    ``issued`` collects ``(request_id, prompt_text, max_tokens)``. The prompt never survives
    into the bundle -- records carry counts, not text -- so a replay that flattened the
    shape, or that handed rung eight the strings rung one already left in the prefix cache,
    would leave nothing in the artefacts to detect it by. This is the only place the
    evidence exists.
    """
    import ascep.cli as cli
    from ascep.bench.records import Outcome, RequestRecord

    class _Fake:
        name = "fake"

        def __init__(self, *a, **k):
            pass

        async def aclose(self):
            pass

        async def issue(self, spec, *, clock, sink=None):
            # Prompt content lives in messages and nowhere else (adapters/base.py): a test
            # reading spec.content or spec.prompt would raise, not fail, and an errored test
            # is one somebody deletes rather than fixes.
            prompt = spec.messages[0]["content"]
            issued.append((spec.request_id, prompt, spec.max_tokens))
            t = clock()
            ttft = 0.001
            return RequestRecord(
                request_id=spec.request_id,
                issued_ts=t,
                outcome=Outcome.OK,
                first_token_ts=t + ttft,
                token_ts=[t + ttft + 0.0005 * i for i in range(8)],
                end_ts=t + ttft + 0.004 + 0.0005 * 7,
                output_tokens=spec.max_tokens or 128,
                input_tokens=len(prompt.split()),
            )

    monkeypatch.setattr(cli, "_bench_adapter", lambda config: _Fake(), raising=False)


def test_replaying_a_synthetic_corpus_is_refused_because_there_is_nothing_to_replay(
    tmp_path, capsys
):
    """A synthetic corpus holds no captured sessions, so accepting this would publish the
    session-replay label over traffic whose turns neither grow nor carry any of the capture's
    structure."""
    assert _dry_run(tmp_path, **{"workload.replay_sessions": True}) == 2
    error = capsys.readouterr().err
    assert "'replay_sessions' is true but 'corpus' is 'synthetic'" in error
    assert "there is nothing to replay" in error
    assert "`ascep agent-profile --shapes`" in error
    assert "(section 10)" in error


def test_replaying_sessions_with_a_media_root_is_refused_because_the_capture_carries_no_images(
    tmp_path, capsys
):
    """A replay sends the capture's token counts, not image data. Accepting a media root
    would publish a multimodal workload label over requests that carried no media at all."""
    media_dir = tmp_path / "media"
    media_dir.mkdir()
    assert (
        _dry_run(
            tmp_path,
            **{
                "workload.corpus": _shapes_file(tmp_path),
                "workload.replay_sessions": True,
                "workload.input_tokens": None,
                "workload.media_root": str(media_dir),
            },
        )
        == 2
    )
    error = capsys.readouterr().err
    assert "'replay_sessions' is true and 'media_root' is set" in error
    assert "carry token counts and timings, not images" in error
    assert "publishing a media label" in error
    assert "(section 10)" in error


def test_a_declared_input_length_on_a_shapes_file_is_refused_by_the_existing_corpus_rule(
    tmp_path, capsys
):
    """A fixed 512-token declaration contradicts a capture whose prompts grow every turn, so
    the published workload would describe traffic the run did not send.

    The refusal is deliberately the generic section 7 corpus-file rule, not a replay-specific
    one: a shapes file is a corpus file, so a second check could never fire after it. An
    unreachable rule is one a reader trusts and a maintainer breaks without either noticing,
    which is why this test pins that section 10 stays out of it.
    """
    assert (
        _dry_run(
            tmp_path,
            **{
                "workload.corpus": _shapes_file(tmp_path),
                "workload.replay_sessions": True,
            },
        )
        == 2
    )
    error = capsys.readouterr().err
    assert "must be null when 'corpus' names a corpus file" in error
    assert "(section 7)" in error
    assert "(section 10)" not in error


def test_a_declared_output_length_is_refused_because_the_capture_carries_one_per_step(
    tmp_path, capsys
):
    """Half of what a session costs is the output side of its shape: early turns answer in a
    sentence, late turns in pages. One declared length would flatten that side and report a
    run cheaper than the one measured."""
    shapes = _shapes_file(tmp_path)
    assert (
        _dry_run(
            tmp_path,
            **{
                "workload.corpus": shapes,
                "workload.replay_sessions": True,
                "workload.input_tokens": None,
                "workload.output_tokens": 128,
                "workload.ignore_eos": False,
                "workload.think_time_s": 0.0,
                "workload.cache_policy": "declared-workload",
            },
        )
        == 2
    )
    error = capsys.readouterr().err
    assert "an output length is declared" in error
    assert "(section 10)" in error


def test_ignore_eos_alone_is_refused_by_the_existing_rule_that_a_length_must_travel_with_it(
    tmp_path, capsys
):
    """ignore_eos with no length beside it flattens the output side exactly as a declared
    number does -- it asks the server to generate to the context limit on every step.

    The refusal is the pre-existing section 7 pairing rule, not a replay-specific one, and
    that is the point of pinning it: because ignore_eos true with a null length can never
    reach the replay checks, a section 10 clause for it would be a rule that had never once
    run, which is exactly the kind a reader trusts and a maintainer breaks unnoticed. What
    the replay does refuse is the reachable half -- a declared length -- one test above.
    """
    shapes = _shapes_file(tmp_path)
    assert (
        _dry_run(
            tmp_path,
            **{
                "workload.corpus": shapes,
                "workload.replay_sessions": True,
                "workload.input_tokens": None,
                "workload.output_tokens": None,
                "workload.ignore_eos": True,
                "workload.think_time_s": 0.0,
                "workload.cache_policy": "declared-workload",
            },
        )
        == 2
    )
    error = capsys.readouterr().err
    assert "must be a positive integer when 'ignore_eos' is true" in error
    assert "(section 7)" in error
    assert "(section 10)" not in error


def test_a_non_zero_think_time_is_refused_because_the_capture_already_replays_each_gap(
    tmp_path, capsys
):
    """The capture carries the measured pause after every step. A think time on top idles
    twice per turn, stretching every session past anything that was observed, and the report
    would then charge the model for a step rate the doubled idle time actually caused."""
    shapes = _shapes_file(tmp_path)
    assert (
        _dry_run(
            tmp_path,
            **{
                "workload.corpus": shapes,
                "workload.replay_sessions": True,
                "workload.input_tokens": None,
                "workload.output_tokens": None,
                "workload.ignore_eos": False,
                "workload.cache_policy": "declared-workload",
                "workload.think_time_s": 0.5,
            },
        )
        == 2
    )
    error = capsys.readouterr().err
    assert "a second idle period" in error
    assert "(section 10)" in error


def test_unique_prefix_is_refused_because_a_replayed_session_shares_prefixes_on_purpose(
    tmp_path, capsys
):
    """Turn k's prompt begins with turn k-1's, and that reuse is the deployment being
    measured. A replay published under 'unique-prefix' would deny in its config exactly what
    its results depended on, and a reader comparing cache policies would grade this run
    against cards that played by different rules."""
    shapes = _shapes_file(tmp_path)
    assert (
        _dry_run(
            tmp_path,
            **{
                "workload.corpus": shapes,
                "workload.replay_sessions": True,
                "workload.input_tokens": None,
                "workload.output_tokens": None,
                "workload.ignore_eos": False,
                "workload.think_time_s": 0.0,
                "workload.cache_policy": "unique-prefix",
            },
        )
        == 2
    )
    error = capsys.readouterr().err
    assert "shares prefixes on purpose" in error
    assert "(section 10)" in error


def test_a_missing_shapes_file_is_refused_by_the_corpus_check_before_anything_is_replayed(
    tmp_path, capsys
):
    """A shapes path pointing at nothing must fail as a missing corpus, not as a malformed
    capture. A config allowed past here would spend the window replaying traffic that does
    not exist and publish the result as a measured zero rather than as a refusal.

    Every other declaration is nulled out, because the section 7 input-length rule sits ahead
    of the path check and would answer first: a test that left input_tokens at its default
    would be pinning that rule instead, and would keep passing after this one was deleted.
    """
    path = _write(tmp_path, _replay_config(tmp_path, "not-there-shapes.json"))
    assert main(["bench", path, "--dry-run"]) == 2
    error = capsys.readouterr().err
    assert "corpus file not found" in error
    assert "(section 7)" in error


def test_a_shapes_file_with_the_wrong_version_is_refused_and_the_refusal_names_the_path(
    tmp_path, capsys
):
    """A shapes file from another schema version describes steps whose fields this replay
    does not understand, so replaying it would publish captured-per-step lengths read out of
    the wrong places in each record.

    The assertion stops at the wrapper's own words and the filename: the sentence inside the
    loader's ValueError belongs to that module's contract, and restating it here would couple
    this suite to wording it does not own.
    """
    shapes = _shapes_file(tmp_path, version=2)
    path = _write(tmp_path, _replay_config(tmp_path, shapes))
    assert main(["bench", path, "--dry-run"]) == 2
    error = capsys.readouterr().err
    assert "cannot be replayed" in error
    assert pathlib.Path(shapes).name in error


def test_a_config_that_omits_replay_sessions_entirely_still_runs_the_ordinary_text_path(
    tmp_path, monkeypatch
):
    """The key is optional, so its absence must mean the text ladder the other keys describe.
    Exit 0 alone cannot tell "replay correctly off" from "replay on and silently empty", so
    the manifest is the assertion: a text run must carry no session plan at all."""
    path = _write(tmp_path, _config(tmp_path))
    _run_offline(monkeypatch)
    assert main(["bench", path]) == 0
    configs = json.loads((tmp_path / "bundle" / "run_configs.json").read_text(encoding="utf-8"))
    assert "session_plan" not in configs["workload"]


def test_each_replayed_request_carries_the_prompt_and_output_lengths_of_its_captured_step(
    tmp_path, monkeypatch
):
    """Flattening either side of a captured shape changes the load being measured: one common
    prompt size hides the growth that drives prefill, one common output size hides the mixed
    decode cost. Either way the numbers stay valid-looking while describing a workload nobody
    captured."""
    steps = [
        {
            "turn_index": 0,
            "prompt_tokens": 3,
            "output_tokens": 4,
            "gap_s": 0.0,
            "resets_prefix": False,
        },
        {
            "turn_index": 1,
            "prompt_tokens": 5,
            "output_tokens": 2,
            "gap_s": 0.0,
            "resets_prefix": False,
        },
        {
            "turn_index": 2,
            "prompt_tokens": 8,
            "output_tokens": 3,
            "gap_s": 0.0,
            "resets_prefix": False,
        },
    ]
    shapes = _shapes_file(tmp_path, [{"session_id": "agent-a", "steps": steps}])
    path = _write(
        tmp_path,
        _replay_config(tmp_path, shapes),
    )
    issued: list = []
    _recording_adapter(monkeypatch, issued)

    assert main(["bench", path]) == 0
    assert issued
    observed = {(len(prompt.split()), max_tokens) for _, prompt, max_tokens in issued}
    assert observed == {(step["prompt_tokens"], step["output_tokens"]) for step in steps}


def test_steps_of_a_session_replay_arrive_in_order_and_each_prompt_extends_the_last(
    tmp_path, monkeypatch
):
    """Token counts survive a loss of session structure; cache behaviour does not.
    Independent prompts of these lengths look like an ordinary corpus and overstate serving
    cost, and prompts that reset discard the prefix reuse a real agent gets from carrying its
    earlier turns forward. Only the text can tell the three apart."""
    steps = [
        {
            "turn_index": 0,
            "prompt_tokens": 4,
            "output_tokens": 2,
            "gap_s": 0.0,
            "resets_prefix": False,
        },
        {
            "turn_index": 1,
            "prompt_tokens": 7,
            "output_tokens": 3,
            "gap_s": 0.0,
            "resets_prefix": False,
        },
        {
            "turn_index": 2,
            "prompt_tokens": 10,
            "output_tokens": 1,
            "gap_s": 0.0,
            "resets_prefix": False,
        },
    ]
    shapes = _shapes_file(tmp_path, [{"session_id": "agent-b", "steps": steps}])
    path = _write(
        tmp_path,
        _replay_config(tmp_path, shapes),
    )
    issued: list = []
    _recording_adapter(monkeypatch, issued)

    assert main(["bench", path]) == 0
    # Grouped by the session the request_id names, not sliced off the front of the stream:
    # warm-up stops the moment its request quota is met, which is almost always mid-session,
    # so the first three requests straddle two sessions and the prefix assertion would fail
    # on traffic that was perfectly correct.
    by_session: dict[str, list[str]] = {}
    for request_id, prompt, _ in issued:
        by_session.setdefault(request_id.rsplit("-i", 1)[0], []).append(prompt)
    whole = [prompts for prompts in by_session.values() if len(prompts) == len(steps)]
    assert whole, f"no session ran to completion: {[len(v) for v in by_session.values()]}"
    for prompts in whole:
        assert [len(prompt.split()) for prompt in prompts] == [
            step["prompt_tokens"] for step in steps
        ]
        for previous, current in zip(prompts, prompts[1:]):
            assert current.startswith(previous)


def test_the_bundle_carries_the_session_plan_instead_of_a_single_output_length(
    tmp_path, monkeypatch
):
    """A manifest that only says a replay happened leaves a reader unable to tell one captured
    set of sessions from another, and a fixed output field would point them at a length that
    never governed the run. The bundle has to publish the same digest and size as the plan it
    replayed, and name the per-step basis."""
    steps = [
        {
            "turn_index": 0,
            "prompt_tokens": 5,
            "output_tokens": 3,
            "gap_s": 0.0,
            "resets_prefix": False,
        },
        {
            "turn_index": 1,
            "prompt_tokens": 9,
            "output_tokens": 2,
            "gap_s": 0.0,
            "resets_prefix": False,
        },
    ]
    shapes = _shapes_file(
        tmp_path, [{"session_id": "agent-c", "steps": steps}], shared_prefix_tokens=5
    )
    path = _write(
        tmp_path,
        _replay_config(tmp_path, shapes),
    )
    _run_offline(monkeypatch)

    assert main(["bench", path]) == 0
    manifest = json.loads((tmp_path / "bundle" / "run_configs.json").read_text(encoding="utf-8"))[
        "workload"
    ]
    loaded_shapes, shared_prefix_tokens = bench_run.sessions.load_shapes(shapes)
    expected = bench_run.sessions.ReplaySessionPlan(
        shapes=loaded_shapes,
        seed=11,
        tokenizer=lambda text: len(text.split()),
        shared_prefix_tokens=shared_prefix_tokens,
        label=pathlib.Path(shapes).stem,
    ).manifest()
    assert manifest["session_plan"] == expected
    assert manifest["output_basis"] == "captured-per-step"
    assert "output_tokens" not in manifest


def test_every_replay_window_reports_how_many_sessions_it_started_and_completed(
    tmp_path, monkeypatch
):
    """A window ends on the clock, so an unfinished session loses its expensive later turns.
    Counting only completions makes that truncation invisible and lets a reader infer a
    workload that reached deeper turns than the evidence contains; the started count is the
    denominator that stops zero completions reading as a satisfactory session ladder."""
    shapes = _shapes_file(tmp_path, session_count=16, steps=4)
    path = _write(tmp_path, _replay_config(tmp_path, shapes))
    _run_offline(monkeypatch)
    assert main(["bench", path]) == 0
    configs = json.loads((tmp_path / "bundle" / "run_configs.json").read_text(encoding="utf-8"))
    assert configs["windows"]
    for window in configs["windows"]:
        assert 0 <= window["sessions_completed"] <= window["sessions_started"]


def test_a_text_run_does_not_publish_zero_session_counts_that_look_like_failed_replays(
    tmp_path, monkeypatch
):
    """A truthful zero on a text run would say either that nothing was replayed or that every
    replayed session failed to finish. Those two readings drive different fixes, so the bundle
    leaves the keys absent rather than flattening them to 0."""
    path = _write(tmp_path, _config(tmp_path))
    _run_offline(monkeypatch)
    assert main(["bench", path]) == 0
    configs = json.loads((tmp_path / "bundle" / "run_configs.json").read_text(encoding="utf-8"))
    assert configs["windows"]
    for window in configs["windows"]:
        assert "sessions_started" not in window
        assert "sessions_completed" not in window


def test_no_two_ladder_rungs_replay_the_same_prompt_text(tmp_path, monkeypatch):
    """Shared prompts between rungs let an upper rung answer out of the cache a lower one
    filled, flattering its tier with prefill the system never had to do. That is the
    flattering failure: measured capacity climbs with concurrency and the published ceiling
    sits further away than the one actually reached."""
    shapes = _shapes_file(tmp_path, session_count=16, steps=4)
    path = _write(tmp_path, _replay_config(tmp_path, shapes))
    issued: list = []
    _recording_adapter(monkeypatch, issued)
    assert main(["bench", path]) == 0

    # A replayed request_id is `c{concurrency}-r{repetition}-s{ordinal}-i{index}` -- the
    # session ordinal sits between the repetition and the index, and there is no run_label
    # in front, so the text-path pattern would match nothing and the test would pass on an
    # empty set of rungs.
    rung_of = re.compile(r"^c(\d+)-r\d+-s\d+-i\d+$")
    prompts_by_rung: dict[int, set[str]] = {}
    for request_id, prompt, _ in issued:
        found = rung_of.match(request_id)
        assert found is not None, request_id
        prompts_by_rung.setdefault(int(found.group(1)), set()).add(prompt)

    assert set(prompts_by_rung) == {1, 2, 4}
    rungs = list(prompts_by_rung.values())
    for index, left in enumerate(rungs):
        for right in rungs[index + 1 :]:
            assert left.isdisjoint(right)


def test_a_session_run_still_validates_against_the_capacity_report_schema(tmp_path, monkeypatch):
    """Adding session evidence must not make the report unreadable to anyone validating the
    documented shape. The check is the shared one the text run makes, so the two cannot drift
    apart into a strong version and a weaker copy beside it."""
    shapes = _shapes_file(tmp_path, session_count=16, steps=4)
    assert_draft_validates(tmp_path, monkeypatch, _replay_config(tmp_path, shapes))
