"""What a workload must declare before it is allowed to generate a single request.

Chapter 7 section 2 and section 3 are the two places a benchmark most cheaply lies to
itself: by not saying whether token counts were fixed or sampled, and by letting a prefix
cache serve traffic no production user would ever generate. Both failures inflate
throughput, neither leaves a trace in the records, and a reader holding only the report
cannot detect either. These tests hold the workload layer to declaring both.

The module under test is stdlib-only on purpose. Reconstructing the exact prompt sequence
from a published manifest is the strongest reproduction check there is, and it must not
require installing our HTTP client to perform.
"""

import base64
import json

import pytest

from ascep.bench.adapters.base import RequestSpec
from ascep.bench.workloads import (
    _FILLER_WORDS,
    CACHE_POLICIES,
    CappedOutput,
    FixedOutput,
    JsonlCorpus,
    ModelDecidedOutput,
    MultimodalJsonlCorpus,
    SyntheticCorpus,
    Workload,
)


# A stand-in tokenizer with the one property that matters here: it is deterministic and it
# disagrees with character count. Using len(text) would let a length fit that never
# tokenizes anything pass, which is the exact confusion section 7.2 is written against.
def words(text: str) -> int:
    return len(text.split())


def _synthetic(input_tokens: int = 64, **kw) -> SyntheticCorpus:
    return SyntheticCorpus(input_tokens=input_tokens, tokenizer=words, **kw)


def _workload(**kw) -> Workload:
    base = dict(
        source=_synthetic(),
        output_plan=FixedOutput(output_tokens=128, ignore_eos=True),
        cache_policy="disabled",
        seed=7,
        think_time_s=0.0,
        run_label="t",
    )
    base.update(kw)
    return Workload(**base)


def _corpus_file(tmp_path, prompts, field="question"):
    path = tmp_path / "corpus.jsonl"
    path.write_text("\n".join(json.dumps({field: p, "id": i}) for i, p in enumerate(prompts)))
    return path


# --- section 7.2: fixed or sampled, and never silently either -------------------------


def test_a_fixed_output_plan_must_say_whether_eos_was_ignored():
    """A fixed output length is only fixed if the model was forbidden to stop early.

    Without ignore_eos the number is a ceiling, not a length, and a run that reports
    "512 output tokens" while the model emitted 60 has published a workload it did not
    run -- with a decode cost an order of magnitude off.
    """
    with pytest.raises(TypeError):
        FixedOutput(output_tokens=512)


def test_the_manifest_labels_the_token_basis_the_run_actually_used():
    fixed = _workload().manifest()
    assert fixed["output_basis"] == "fixed"
    assert fixed["ignore_eos"] is True

    decided = _workload(output_plan=ModelDecidedOutput()).manifest()
    assert decided["output_basis"] == "model-decided"
    # Not "sampled": nobody sampled it. Section 7.2 demands the run say which of the two it
    # was, and answering "sampled" for a length the model chose claims a declared
    # distribution that does not exist.
    assert decided["ignore_eos"] is False


def test_a_model_decided_output_plan_sends_no_max_tokens():
    """max_tokens=None means server default. Coercing it to a number here would change the
    workload invisibly, because the record shows only what the server was asked for."""
    spec = _workload(output_plan=ModelDecidedOutput()).for_repetition(0)(0)
    assert spec.max_tokens is None


def test_a_fixed_output_plan_puts_its_budget_on_every_spec():
    plan = FixedOutput(output_tokens=97, ignore_eos=True)
    spec = _workload(output_plan=plan).for_repetition(0)(3)
    assert spec.max_tokens == 97
    assert spec.extra.get("ignore_eos") is True


def test_a_capped_output_plan_puts_a_ceiling_on_every_spec_and_says_nothing_about_eos():
    """EOS is honoured and the number is a ceiling, not a length: the production-realistic
    mode. The defect this mode fixes was a key that never made it onto the wire, so the
    absence of ignore_eos is asserted, not merely its value -- an extra carrying
    ignore_eos: false is the same request at the server while claiming a key nobody asked
    for."""
    spec = _workload(output_plan=CappedOutput(output_tokens=512)).for_repetition(0)(0)
    assert spec.max_tokens == 512
    assert "ignore_eos" not in spec.extra


def test_a_capped_output_plan_refuses_a_non_positive_ceiling():
    """A ceiling of zero stops nothing: it can only be a config typo, and a typo the
    harness accepts is a workload nobody declared."""
    with pytest.raises(ValueError, match="positive int"):
        CappedOutput(output_tokens=0)


def test_the_three_output_plans_leave_manifests_a_reader_can_tell_apart():
    """A bundle that cannot tell "capped at 512, EOS honoured" from "uncapped" cannot be
    used to reproduce the run: the live defect published the second while the config
    declared the first."""
    fixed = _workload(output_plan=FixedOutput(output_tokens=64, ignore_eos=True)).manifest()
    capped = _workload(output_plan=CappedOutput(output_tokens=64)).manifest()
    decided = _workload(output_plan=ModelDecidedOutput()).manifest()
    assert fixed["output_basis"] == "fixed"
    assert fixed["ignore_eos"] is True and fixed["output_tokens"] == 64
    assert capped["output_basis"] == "capped"
    assert capped["ignore_eos"] is False and capped["output_tokens"] == 64
    assert decided["output_basis"] == "model-decided"
    assert decided["ignore_eos"] is False and "output_tokens" not in decided
    serialised = {json.dumps(m, sort_keys=True) for m in (fixed, capped, decided)}
    assert len(serialised) == 3, "the three modes must not collapse into one manifest"


def test_a_synthetic_corpus_cannot_be_built_without_a_tokenizer():
    """Section 7.2: token quantities MUST be defined after tokenization.

    A generator targeting a token count using words or characters produces materially
    different prompts per language and per tokenizer, so the declared input_tokens would be
    a number no two runs agree on.
    """
    with pytest.raises(TypeError):
        SyntheticCorpus(input_tokens=64)


def test_a_synthetic_prompt_tokenizes_to_exactly_the_declared_length():
    spec = _workload(source=_synthetic(input_tokens=64)).for_repetition(0)(0)
    assert words(spec.messages[0]["content"]) == 64


def test_a_non_positive_token_target_is_refused():
    with pytest.raises(ValueError, match="token"):
        _synthetic(input_tokens=0)


def test_synthetic_filler_is_drawn_from_the_common_word_vocabulary():
    """`ascep bench` counts words, so one word must cost the served model about one token.

    That is a property of the words, not of the counting. Measured on Gemma 4, the random
    hex filler this generator used to emit cost 7.98 tokens per word, so a config asking for
    input_tokens 1,500 sent roughly 12,000 and the avg_context_tokens, KV-floor and TTFT
    figures downstream all described a workload nobody declared. Nothing else in the suite
    would notice: the word count still lands on the target exactly, which is the number
    every other synthetic test asserts on.
    """
    prompt = _synthetic(input_tokens=200).render(seed_material=11, prefix=None)
    vocabulary = set(_FILLER_WORDS)
    assert set(prompt.split()) <= vocabulary
    assert len(vocabulary) >= 100, "too small a vocabulary and prompts share cacheable prefixes"


def test_two_synthetic_prompts_share_no_cacheable_prefix():
    """A vLLM prefix-cache block is sixteen tokens; a shared head that long would let rung
    two answer out of the cache rung one filled, and the throughput floor would be a
    measurement of the cache.

    Drawing filler from a closed vocabulary made this possible in a way unique hex strings
    never were, so it is pinned rather than argued.
    """
    corpus = _synthetic(input_tokens=200)
    first = corpus.render(seed_material=1, prefix=None).split()
    second = corpus.render(seed_material=2, prefix=None).split()
    shared = 0
    for left, right in zip(first, second):
        if left != right:
            break
        shared += 1
    assert shared < 16


def test_a_synthetic_corpus_refuses_a_target_it_cannot_hit_exactly():
    """Better to fail than to publish input_tokens=1024 for a 1019-token prompt: the
    shortfall is invisible in the report and it moves the prefill cost.

    The target is unreachable here rather than merely invalid, which is the case the
    construction-time bounds check does not cover -- a tokenizer that counts in twos can
    never land on an odd number however the filler is chosen.
    """
    even_only = SyntheticCorpus(input_tokens=63, tokenizer=lambda t: 2 * len(t.split()))
    with pytest.raises(ValueError, match="exactly 63"):
        even_only.render(seed_material=1, prefix=None)


# --- section 7.3: cache control is a declaration, not a hope --------------------------


def test_the_cache_policy_vocabulary_is_closed():
    assert set(CACHE_POLICIES) == {
        "disabled",
        "cleared",
        "unique-prefix",
        "declared-workload",
        "unknown",
    }


def test_an_unrecognised_cache_policy_is_refused_by_name():
    with pytest.raises(ValueError, match="cache"):
        _workload(cache_policy="probably-off")


def test_an_unknown_cache_policy_must_carry_its_u_reason():
    """Rule C1: null is publishable, unjustified null is not. 'unknown' is the null of
    section 7.3, and the reason is what stops it being used as a shrug."""
    with pytest.raises(ValueError, match="reason"):
        _workload(cache_policy="unknown")
    w = _workload(cache_policy="unknown", unknown_cache_reason="managed endpoint, no cache API")
    assert w.manifest()["cache_policy"] == "unknown"
    assert w.manifest()["cache_policy_u_reason"]


def test_unique_prefix_actually_makes_every_prompt_distinct(tmp_path):
    """Declaring unique-prefix over a corpus that repeats verbatim would be a false
    declaration the server cheerfully rewards with cache hits."""
    path = _corpus_file(tmp_path, ["same prompt"] * 8)
    w = _workload(source=JsonlCorpus(path=path, field="question"), cache_policy="unique-prefix")
    spec_of = w.for_repetition(0)
    texts = [spec_of(i).messages[0]["content"] for i in range(8)]
    assert len(set(texts)) == 8


def test_unique_prefix_varies_across_repetitions_too(tmp_path):
    """Section 7.5 wants independent repetitions. Reusing repetition 0's prompts in
    repetition 1 hands repetition 1 a warm prefix cache that production never has, so the
    dispersion across repetitions measures cache state rather than run-to-run variance."""
    path = _corpus_file(tmp_path, ["same prompt"] * 4)
    w = _workload(source=JsonlCorpus(path=path, field="question"), cache_policy="unique-prefix")
    first = {w.for_repetition(0)(i).messages[0]["content"] for i in range(4)}
    second = {w.for_repetition(1)(i).messages[0]["content"] for i in range(4)}
    assert not (first & second)


def test_a_policy_other_than_unique_prefix_leaves_the_corpus_text_alone(tmp_path):
    """Injecting a prefix nobody asked for changes input_tokens, and the report would
    attribute the difference to the corpus."""
    path = _corpus_file(tmp_path, ["exact corpus text"])
    w = _workload(source=JsonlCorpus(path=path, field="question"), cache_policy="cleared")
    assert w.for_repetition(0)(0).messages[0]["content"] == "exact corpus text"


def test_a_unique_prefix_on_a_synthetic_corpus_still_hits_the_declared_length():
    """The prefix is part of the prompt, so it is part of the token budget. Padding to the
    target and then prepending would overshoot every declared input_tokens in the report."""
    w = _workload(source=_synthetic(input_tokens=64), cache_policy="unique-prefix")
    assert words(w.for_repetition(0)(0).messages[0]["content"]) == 64


def test_the_manifest_says_whether_the_prefix_moved_the_token_count(tmp_path):
    """A JSONL corpus has no room to absorb the prefix, so its prompts really are longer
    than the file. That is acceptable and it must be visible."""
    path = _corpus_file(tmp_path, ["a", "b"])
    jsonl = _workload(source=JsonlCorpus(path=path, field="question"), cache_policy="unique-prefix")
    assert jsonl.manifest()["prefix_adds_tokens"] is True
    synthetic = _workload(source=_synthetic(), cache_policy="unique-prefix")
    assert synthetic.manifest()["prefix_adds_tokens"] is False


# --- section 7.2: think time is declared, and a distribution is not a scalar ----------


def test_think_time_has_no_default():
    """Section 7.2: for closed-loop runs think time MUST be declared, and zero is valid only
    if the product really submits the next request immediately. A default of zero turns that
    conditional permission into the silent norm and saturates the server with fewer users."""
    with pytest.raises(TypeError):
        Workload(
            source=_synthetic(),
            output_plan=ModelDecidedOutput(),
            cache_policy="disabled",
            seed=1,
            run_label="t",
        )


def test_negative_think_time_is_refused():
    with pytest.raises(ValueError, match="think"):
        _workload(think_time_s=-0.1)


# --- reproduction: the manifest is the bundle's half of the promise -------------------


def test_the_same_seed_and_index_rebuild_the_same_request(tmp_path):
    path = _corpus_file(tmp_path, [f"prompt {i}" for i in range(32)])

    def build():
        return _workload(source=JsonlCorpus(path=path, field="question"), seed=11)

    a = [build().for_repetition(2)(i).messages for i in range(16)]
    b = [build().for_repetition(2)(i).messages for i in range(16)]
    assert a == b


def test_a_different_seed_draws_a_different_sequence(tmp_path):
    path = _corpus_file(tmp_path, [f"prompt {i}" for i in range(64)])

    def texts(seed):
        w = _workload(source=JsonlCorpus(path=path, field="question"), seed=seed)
        return [w.for_repetition(0)(i).messages[0]["content"] for i in range(24)]

    assert texts(1) != texts(2)


def test_request_ids_are_unique_across_repetitions():
    """Records from several repetitions land in one bundle. Colliding ids make the bundle
    ambiguous and would let the driver's dedupe silently drop a real request."""
    w = _workload()
    ids = {w.for_repetition(r)(i).request_id for r in range(3) for i in range(20)}
    assert len(ids) == 60


def test_request_ids_are_unique_across_ladder_rungs_as_well():
    """A ladder writes one bundle for every window of every rung, so the rung is part of a
    request's identity too. Repetition alone collides on the second rung, and persist refuses
    the whole bundle at the end of the run -- after the GPU hours are spent."""
    w = _workload()
    ids = {
        w.for_repetition(r, concurrency=c)(i).request_id
        for c in (1, 2, 4)
        for r in range(3)
        for i in range(20)
    }
    assert len(ids) == 180


def test_unique_prefix_varies_across_rungs_so_a_high_rung_cannot_inherit_a_warm_cache(tmp_path):
    """Rungs run in ascending order, so a rung that replays a lower rung's prompts is served
    partly out of a prefix cache the lower rung filled. The bias only ever inflates the high
    rungs, which is exactly where the sustainable concurrency is decided."""
    path = _corpus_file(tmp_path, ["same prompt"] * 4)
    w = _workload(source=JsonlCorpus(path=path, field="question"), cache_policy="unique-prefix")
    low = {w.for_repetition(0, concurrency=1)(i).messages[0]["content"] for i in range(4)}
    high = {w.for_repetition(0, concurrency=64)(i).messages[0]["content"] for i in range(4)}
    assert not (low & high)


def test_the_manifest_carries_everything_needed_to_regenerate_the_prompts(tmp_path):
    """Section 7.2: seed, sampler, distribution parameters and the sampled sequences MUST be
    part of the reproduction bundle. An index-addressed sampler satisfies the last of those
    by construction -- but only if the seed, the corpus identity and the draw rule ship too.
    """
    path = _corpus_file(tmp_path, [f"prompt {i}" for i in range(8)])
    m = _workload(source=JsonlCorpus(path=path, field="question"), seed=5).manifest()
    for key in (
        "seed",
        "sampler",
        "cache_policy",
        "think_time_s",
        "output_basis",
        "corpus_name",
        "corpus_digest",
        "corpus_size",
        "corpus_field",
        "media_placeholders_stripped",
        "run_label",
    ):
        assert key in m, f"the manifest cannot regenerate the run without {key}"
    assert m["seed"] == 5
    assert m["corpus_size"] == 8


def test_the_corpus_digest_changes_when_the_corpus_does(tmp_path):
    """A manifest naming a path is worthless if the file behind it was edited. The digest is
    what makes 'same corpus' checkable rather than asserted."""
    path = _corpus_file(tmp_path, ["a", "b"])
    before = JsonlCorpus(path=path, field="question").digest
    _corpus_file(tmp_path, ["a", "c"])
    after = JsonlCorpus(path=path, field="question").digest
    assert before != after


def test_the_manifest_is_json_serialisable(tmp_path):
    """It ships in the reproduction bundle as JSON. A dataclass or a Path in there fails at
    write time, after the GPU hours have been spent."""
    path = _corpus_file(tmp_path, ["a"])
    m = _workload(source=JsonlCorpus(path=path, field="question")).manifest()
    assert json.loads(json.dumps(m)) == m


# --- the corpus reader: silence is the failure mode -----------------------------------


def test_an_empty_corpus_is_refused(tmp_path):
    path = tmp_path / "empty.jsonl"
    path.write_text("")
    with pytest.raises(ValueError, match="empty"):
        JsonlCorpus(path=path, field="question")


def test_a_missing_field_names_the_field_and_the_line(tmp_path):
    path = tmp_path / "c.jsonl"
    path.write_text(json.dumps({"question": "ok"}) + "\n" + json.dumps({"other": "x"}) + "\n")
    with pytest.raises(ValueError) as exc:
        JsonlCorpus(path=path, field="question")
    assert "question" in str(exc.value) and "2" in str(exc.value)


def test_a_multimodal_record_is_refused_rather_than_flattened(tmp_path):
    """The image is worth thousands of input tokens. Dropping it to keep the text would
    publish an input_tokens figure for a prompt nobody sent, so a corpus this reader cannot
    represent has to stop the run instead of quietly becoming a text-only one.
    """
    path = tmp_path / "mm.jsonl"
    path.write_text(json.dumps({"question": [{"type": "image", "url": "a.png"}]}) + "\n")
    with pytest.raises(ValueError, match="text"):
        JsonlCorpus(path=path, field="question")


def test_a_media_placeholder_left_in_the_text_is_refused(tmp_path):
    """The shape real post-training corpora actually arrive in, and the one that gets past
    a type check.

    A record whose prompt is the string "<image> Q1: what year ..." is a valid string, so
    nothing above catches it. Sent to a text endpoint it is a five-token prompt where
    production sends fifteen hundred, and every input_tokens, TTFT and prefill figure in the
    report is then measured on a workload that does not exist. Refusing it is the only
    honest option until this module can carry the image itself.
    """
    path = tmp_path / "ph.jsonl"
    path.write_text(json.dumps({"question": "<image> Q1: what year was it built?"}) + "\n")
    with pytest.raises(ValueError, match="image"):
        JsonlCorpus(path=path, field="question")


def test_stripping_media_placeholders_is_possible_but_must_be_declared(tmp_path):
    """Running the text half of a multimodal corpus is a legitimate experiment. It is a
    different workload from the corpus it came from, so it takes an explicit argument and
    lands in the manifest where a reader will see it."""
    path = tmp_path / "ph.jsonl"
    path.write_text(json.dumps({"question": "<image> Q1: what year was it built?"}) + "\n")
    corpus = JsonlCorpus(path=path, field="question", strip_media_placeholders=True)
    w = _workload(source=corpus, cache_policy="declared-workload")
    assert "<image>" not in w.for_repetition(0)(0).messages[0]["content"]
    assert w.manifest()["media_placeholders_stripped"] is True
    assert _workload().manifest()["media_placeholders_stripped"] is False


def test_a_nested_field_path_reaches_the_prompt_where_it_actually_lives(tmp_path):
    """Post-training corpora store the prompt inside a conversation list, not at the top
    level. Without a path, using one means pre-flattening the corpus into a second file --
    and the digest in the manifest then pins the copy rather than the dataset."""
    path = tmp_path / "conv.jsonl"
    path.write_text(
        json.dumps({"id": 1, "conversations": [{"from": "human", "value": "first prompt"}]}) + "\n"
    )
    corpus = JsonlCorpus(path=path, field="conversations.0.value")
    w = _workload(source=corpus, cache_policy="declared-workload")
    assert w.for_repetition(0)(0).messages[0]["content"] == "first prompt"
    assert w.manifest()["corpus_field"] == "conversations.0.value"


def test_a_nested_path_that_does_not_resolve_names_the_component_that_failed(tmp_path):
    """A bare "missing field" over a dotted path sends the operator to grep the wrong key."""
    path = tmp_path / "conv.jsonl"
    path.write_text(json.dumps({"conversations": [{"from": "human"}]}) + "\n")
    with pytest.raises(ValueError) as exc:
        JsonlCorpus(path=path, field="conversations.0.value")
    assert "value" in str(exc.value)


def test_a_corpus_smaller_than_the_run_is_reused_and_says_so(tmp_path):
    """Reuse is allowed -- most corpora are smaller than a saturation ladder -- but it is a
    cache-hit source, so corpus_size is in the manifest for a reader to divide by."""
    path = _corpus_file(tmp_path, ["a", "b", "c"])
    w = _workload(source=JsonlCorpus(path=path, field="question"), cache_policy="declared-workload")
    spec_of = w.for_repetition(0)
    texts = {spec_of(i).messages[0]["content"] for i in range(30)}
    assert texts <= {"a", "b", "c"}
    assert w.manifest()["corpus_size"] == 3


def test_the_temperature_is_passed_through_untouched():
    """Temperature changes output length, which changes decode cost. A harness default would
    make two runs of the same declared workload measure different things."""
    assert _workload().for_repetition(0)(0).temperature is None
    assert _workload(temperature=0.0).for_repetition(0)(0).temperature == 0.0


# --- multimodal corpora: the markers are honoured, not stripped ---------------------

# A few bytes of magic are enough: nothing in the loader decodes the payload, and the
# point of these tests is that the loader touches real paths, so the filesystem is not
# mocked.
_JPEG = b"\xff\xd8\xff\xe0" + bytes(16)
_PNG = b"\x89PNG\r\n\x1a\n" + bytes(16)


def _mm_record(rel_path, width, height, answer="A lathe.", rid="r1"):
    return {
        "image": rel_path,
        "width": width,
        "height": height,
        "conversations": [
            {"from": "human", "value": "<image>\nWhat is happening?"},
            {"from": "gpt", "value": answer},
        ],
        "id": rid,
    }


def _mm_corpus(tmp_path, records, media=()):
    root = tmp_path / "media"
    root.mkdir(exist_ok=True)
    for rel, data in media:
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    path = tmp_path / "mm.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    return path, root


def test_a_multimodal_corpus_reports_its_markers_as_honoured_not_stripped(tmp_path):
    """media_placeholders_stripped=False is not a missing feature here, it is the truth.

    The markers were replaced by real content parts, so the run that happened is the media
    run. Reporting True would describe the text-only variant -- a workload nobody ran --
    and a reader comparing the two reports would subtract numbers that were never measured.
    """
    path, root = _mm_corpus(
        tmp_path,
        [
            _mm_record("images/a.jpg", 1920, 1080, rid="r1"),
            _mm_record("images/b.png", 768, 432, rid="r2"),
        ],
        media=[("images/a.jpg", _JPEG), ("images/b.png", _PNG)],
    )
    corpus = MultimodalJsonlCorpus(path=path, media_root=root, transport="base64")
    assert corpus.size == 2
    assert corpus.media_placeholders_stripped is False


def test_render_content_replaces_the_marker_with_a_real_image_part(tmp_path):
    """The text part must not still carry "<image>": sent as text it is the five-token
    prompt MEDIA_PLACEHOLDER exists to refuse, and every input_tokens, TTFT and prefill
    figure in the report would describe a workload nobody ran."""
    path, root = _mm_corpus(
        tmp_path,
        [_mm_record("images/a.jpg", 1920, 1080)],
        media=[("images/a.jpg", _JPEG)],
    )
    corpus = MultimodalJsonlCorpus(path=path, media_root=root, transport="base64")
    parts = corpus.render_content(seed_material=1, prefix=None)
    assert parts[0] == {"type": "text", "text": "What is happening?"}
    assert parts[1]["type"] == "image_url"
    assert parts[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_the_data_url_mime_type_is_guessed_from_the_actual_file(tmp_path):
    """A png mislabeled as jpeg is a request the server may decode differently than the
    corpus intended, and the wrong prefix would be invisible in every record the run
    writes."""
    path, root = _mm_corpus(
        tmp_path,
        [_mm_record("images/b.png", 768, 432)],
        media=[("images/b.png", _PNG)],
    )
    corpus = MultimodalJsonlCorpus(path=path, media_root=root, transport="base64")
    parts = corpus.render_content(seed_material=1, prefix=None)
    assert parts[1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_url_transport_sends_a_fetchable_url_and_embeds_no_bytes(tmp_path):
    """With transport="url" the server fetches the media, so the request body must carry
    the prefix-joined URL and nothing else. Exact equality here is what proves no file
    bytes were read: a base64 payload would be a different transport than the one the
    manifest declares."""
    path, root = _mm_corpus(
        tmp_path,
        [_mm_record("images/a.jpg", 1920, 1080)],
        media=[("images/a.jpg", _JPEG)],
    )
    corpus = MultimodalJsonlCorpus(
        path=path, media_root=root, transport="url", url_prefix="http://host/media"
    )
    parts = corpus.render_content(seed_material=1, prefix=None)
    assert parts[1] == {
        "type": "image_url",
        "image_url": {"url": "http://host/media/images/a.jpg"},
    }


def test_render_refuses_to_turn_a_media_benchmark_into_a_text_one(tmp_path):
    """Returning the text alone would quietly turn a media benchmark into a text benchmark,
    and it would still produce plausible numbers -- lower prefill, faster TTFT, all of it
    measured on a workload nobody declared. The error has to name the way out, or the next
    caller will catch it and carry on with the text."""
    path, root = _mm_corpus(
        tmp_path,
        [_mm_record("images/a.jpg", 1920, 1080)],
        media=[("images/a.jpg", _JPEG)],
    )
    corpus = MultimodalJsonlCorpus(path=path, media_root=root, transport="base64")
    with pytest.raises(ValueError) as exc:
        corpus.render(seed_material=1, prefix=None)
    assert "render_content" in str(exc.value)
    assert "strip_media_placeholders" in str(exc.value)


def test_media_shape_reports_absent_video_as_zero_never_as_null(tmp_path):
    """0.0 means measured and genuinely absent; None means not reported. media_shape feeds
    the workload declaration directly, so emitting None here would put an unmeasurable
    claim into a published report -- and section 9.6 makes a reader trust that null really
    means nobody looked. media_bytes_resident follows the same rule: it is the measured
    total of the data URLs the corpus holds in memory, derived here from the fixture
    bytes so the equality stays exact rather than pinned to a magic number."""
    reasoning = _mm_record(
        "images/b.png",
        768,
        432,
        answer={"reasoning": "the workpiece spins", "answer": "A lathe."},
        rid="r2",
    )
    path, root = _mm_corpus(
        tmp_path,
        [_mm_record("images/a.jpg", 1920, 1080, rid="r1"), reasoning],
        media=[("images/a.jpg", _JPEG), ("images/b.png", _PNG)],
    )
    corpus = MultimodalJsonlCorpus(path=path, media_root=root, transport="base64")
    expected_resident = len(
        "data:image/jpeg;base64," + base64.b64encode(_JPEG).decode("ascii")
    ) + len("data:image/png;base64," + base64.b64encode(_PNG).decode("ascii"))
    assert corpus.media_shape() == {
        "images_per_request": 1.0,
        "videos_per_request": 0.0,
        "records": 2,
        "records_with_reasoning": 1,
        "image_resolution_mix": [
            {"width": 768, "height": 432, "share": 0.5},
            {"width": 1920, "height": 1080, "share": 0.5},
        ],
        "media_bytes_resident": expected_resident,
    }


def test_a_record_whose_markers_and_media_disagree_is_refused_with_counts_and_line(tmp_path):
    """Silently skipping the record is exactly the failure media_arrival_check exists to
    catch after the fact: the run would publish a media capacity figure measured on less
    media than it claims. Refusing at load time is cheaper than discovering it in a
    report, and the error has to say which line and how far apart the counts were."""
    bad = _mm_record("images/a.jpg", 1920, 1080, rid="r2")
    bad["conversations"][0]["value"] = "<image>\n<image>\nWhat is happening?"
    path, root = _mm_corpus(
        tmp_path,
        [_mm_record("images/a.jpg", 1920, 1080, rid="r1"), bad],
        media=[("images/a.jpg", _JPEG)],
    )
    with pytest.raises(ValueError) as exc:
        MultimodalJsonlCorpus(path=path, media_root=root, transport="base64")
    message = str(exc.value)
    assert "line 2" in message
    assert "2 <image>" in message
    assert "1 image" in message


def test_a_missing_media_file_is_refused_with_its_path_and_line(tmp_path):
    """A record whose media cannot be read is a media benchmark quietly becoming a text
    one -- the failure media_arrival_check exists to catch after the fact. Skipping it
    would publish a media capacity figure measured on less media than the report claims,
    so the loader refuses and names the path it could not find."""
    missing = _mm_record("images/gone.jpg", 1920, 1080, rid="r2")
    path, root = _mm_corpus(
        tmp_path,
        [_mm_record("images/a.jpg", 1920, 1080, rid="r1"), missing],
        media=[("images/a.jpg", _JPEG)],
    )
    with pytest.raises(ValueError) as exc:
        MultimodalJsonlCorpus(path=path, media_root=root, transport="base64")
    message = str(exc.value)
    assert "gone.jpg" in message
    assert "line 2" in message


def test_an_unknown_transport_is_refused(tmp_path):
    """The serving layer declares image_input_transport from the same vocabulary, so a
    transport outside it is one the report cannot name -- not a weaker option to coerce."""
    path, root = _mm_corpus(
        tmp_path,
        [_mm_record("images/a.jpg", 1920, 1080)],
        media=[("images/a.jpg", _JPEG)],
    )
    with pytest.raises(ValueError, match="transport"):
        MultimodalJsonlCorpus(path=path, media_root=root, transport="ftp")


def test_url_transport_without_a_url_prefix_is_refused(tmp_path):
    """Without url_prefix the corpus's relative paths resolve to nothing the server can
    fetch; accepting it would build requests whose media never arrives and report the
    resulting text-only answers as a media run."""
    path, root = _mm_corpus(
        tmp_path,
        [_mm_record("images/a.jpg", 1920, 1080)],
        media=[("images/a.jpg", _JPEG)],
    )
    with pytest.raises(ValueError, match="url_prefix"):
        MultimodalJsonlCorpus(path=path, media_root=root, transport="url")


def test_max_records_truncates_the_corpus_and_the_digest_says_so(tmp_path):
    """Two runs over "the same corpus" that saw different numbers of records are not
    comparable, and a digest that ignored the cap would assert they were. Truncation is a
    legitimate smoke run, but it is a different corpus and both the digest and the draw
    rule have to say so."""
    records = [
        _mm_record("images/a.jpg", 1920, 1080, rid="r1"),
        _mm_record("images/a.jpg", 1920, 1080, rid="r2"),
        _mm_record("images/a.jpg", 1920, 1080, rid="r3"),
    ]
    path, root = _mm_corpus(tmp_path, records, media=[("images/a.jpg", _JPEG)])
    full = MultimodalJsonlCorpus(path=path, media_root=root, transport="base64")
    capped = MultimodalJsonlCorpus(path=path, media_root=root, transport="base64", max_records=2)
    assert capped.size == 2
    assert "truncated" in capped.sampler_rule
    assert capped.digest != full.digest


def test_a_text_only_source_produces_a_byte_identical_request_spec(tmp_path):
    """render_content was added to PromptSource and Workload.next_spec now calls it instead
    of render. Every published run depends on the text path being untouched: a change here
    would silently rewrite results that are already citable. The content must stay a plain
    string -- a list of parts would be a different request on the wire -- and the id, the
    token budget and the extras must be exactly what the old code path produced."""
    path = _corpus_file(tmp_path, ["only prompt"])
    corpus = JsonlCorpus(path=path, field="question")
    spec = _workload(source=corpus, seed=11).for_repetition(2)(3)
    assert spec == RequestSpec(
        request_id="t-r2-i3",
        messages=[{"role": "user", "content": "only prompt"}],
        max_tokens=128,
        temperature=None,
        extra={"ignore_eos": True},
    )
    assert isinstance(spec.messages[0]["content"], str)


def test_a_base64_corpus_still_renders_after_its_media_file_is_deleted(tmp_path):
    """The bytes are resident at construction, so the request path touches no disk.

    The old code read and base64-encoded the file inside render_content: 1.93 ms of
    blocking work per generated request (1.24 ms mean read, 0.69 ms mean encode) on the
    driver's single-threaded event loop. That is a client-side ceiling of roughly 518
    requests per second that has nothing to do with the server, so a ladder climbing
    into it would report the load generator's limit as the model's throughput collapse.
    And because the loop is blocked for those 1.93 ms, every other in-flight request
    waits too, so the same milliseconds land in the ITL and TTFT samples of requests
    that read nothing -- arriving in the report as server latency. Deleting the file
    after construction and still rendering the identical data URL is the proof that
    cost is gone from the request path.
    """
    path, root = _mm_corpus(
        tmp_path,
        [_mm_record("images/a.jpg", 1920, 1080)],
        media=[("images/a.jpg", _JPEG)],
    )
    corpus = MultimodalJsonlCorpus(path=path, media_root=root, transport="base64")
    before = corpus.render_content(seed_material=1, prefix=None)
    (root / "images" / "a.jpg").unlink()
    after = corpus.render_content(seed_material=2, prefix=None)
    assert after == before


def test_an_unguessable_mime_type_is_refused_at_construction_naming_the_line(tmp_path):
    """A wrong MIME type is a request the server may decode differently than the corpus
    intended, and defaulting to jpeg would make that substitution invisible in every
    record the run writes. The refusal happens at construction -- before the first
    request, not mid-ladder -- and names the record's line, because the operator's next
    move is to look at that record."""
    path, root = _mm_corpus(
        tmp_path,
        [_mm_record("images/a.unknownext", 1920, 1080)],
        media=[("images/a.unknownext", _JPEG)],
    )
    with pytest.raises(ValueError) as exc:
        MultimodalJsonlCorpus(path=path, media_root=root, transport="base64")
    message = str(exc.value)
    assert "line 1" in message
    assert "refusing to default to jpeg" in message


def test_media_bytes_resident_is_the_sum_of_the_data_urls_the_corpus_emits(tmp_path):
    """The resident total is the number an operator sizes max_records against, so it
    must be the number the run actually puts on the wire. Computing the expectation by
    rendering every record -- rather than re-reading the files -- means the test fails
    if the resident total ever drifts from what is sent."""
    path, root = _mm_corpus(
        tmp_path,
        [
            _mm_record("images/a.jpg", 1920, 1080, rid="r1"),
            _mm_record("images/b.png", 768, 432, rid="r2"),
        ],
        media=[("images/a.jpg", _JPEG), ("images/b.png", _PNG)],
    )
    corpus = MultimodalJsonlCorpus(path=path, media_root=root, transport="base64")
    emitted = set()
    for seed in range(32):
        parts = corpus.render_content(seed_material=seed, prefix=None)
        emitted.add(parts[1]["image_url"]["url"])
    assert len(emitted) == 2, "every record must be rendered for the sum to mean anything"
    assert corpus.media_shape()["media_bytes_resident"] == sum(len(url) for url in emitted)


def test_media_bytes_resident_is_a_measured_zero_of_type_int_under_url_transport(tmp_path):
    """Under url transport nothing is read, so nothing is resident. That is a measured
    and genuinely empty total, and section 9.6 makes null mean "not reported" -- so the
    value must be the int 0, and a False or a None smuggled through the equality check
    would be a different claim than the one the corpus measured."""
    path, root = _mm_corpus(
        tmp_path,
        [_mm_record("images/a.jpg", 1920, 1080)],
        media=[("images/a.jpg", _JPEG)],
    )
    corpus = MultimodalJsonlCorpus(
        path=path, media_root=root, transport="url", url_prefix="http://host/media"
    )
    resident = corpus.media_shape()["media_bytes_resident"]
    assert type(resident) is int
    assert resident == 0


def test_two_records_sharing_one_image_hold_its_bytes_once(tmp_path):
    """The encode is keyed on the resolved path, so a corpus whose records share a media
    file pays the read, the encode and the resident memory once. Counting it per record
    would overstate the memory an operator sizes max_records against by the corpus's
    duplication factor."""
    path, root = _mm_corpus(
        tmp_path,
        [
            _mm_record("images/a.jpg", 1920, 1080, rid="r1"),
            _mm_record("images/a.jpg", 1920, 1080, rid="r2"),
        ],
        media=[("images/a.jpg", _JPEG)],
    )
    corpus = MultimodalJsonlCorpus(path=path, media_root=root, transport="base64")
    assert corpus.size == 2
    url = corpus.render_content(seed_material=1, prefix=None)[1]["image_url"]["url"]
    assert corpus.media_shape()["media_bytes_resident"] == len(url)
