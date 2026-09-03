"""The schemas are the enforceable half of the protocol; these tests keep them honest.

Two failure modes they guard against:
  * schema vocabulary drifting away from ascep.capacity, so a report validates but the
    formulas cannot consume it;
  * declaring a model whose attention family does not match the geometry supplied, which
    silently selects the wrong KV formula.
"""

import glob
import inspect
import json
import pathlib
import re

import pytest
from jsonschema import Draft202012Validator as Validator

from ascep.capacity import DTYPE_BYTES, Constraint, Provenance, Tier

ROOT = pathlib.Path(__file__).parent.parent
SCHEMAS = sorted(glob.glob(str(ROOT / "schemas" / "*.schema.json")))


@pytest.fixture(params=SCHEMAS, ids=lambda p: pathlib.Path(p).name)
def schema(request):
    return json.loads(pathlib.Path(request.param).read_text())


def test_schema_is_valid_draft_2020_12(schema):
    Validator.check_schema(schema)


def test_schema_declares_id_and_version(schema):
    assert schema.get("$id", "").startswith("https://")
    assert "ascep_version" in schema.get("properties", {})


def test_required_fields_are_defined(schema):
    props = schema.get("properties", {})
    assert not [r for r in schema.get("required", []) if r not in props]


def _enums(node, key):
    """Yield every enum declared for a property named `key`, at any depth."""
    if isinstance(node, dict):
        for k, v in node.items():
            if k == key and isinstance(v, dict) and "enum" in v:
                yield v["enum"]
            yield from _enums(v, key)
    elif isinstance(node, list):
        for v in node:
            yield from _enums(v, key)


def test_enums_match_capacity_vocabulary(schema):
    """A schema that allows a tier or constraint the code cannot represent is broken."""
    for key, allowed in (
        ("tier", {t.value for t in Tier}),
        ("binding_constraint", {c.value for c in Constraint}),
        ("provenance", {p.value for p in Provenance}),
    ):
        for enum in _enums(schema, key):
            # None is permitted on every enum: rule C1 requires an unknown value to be
            # recorded as null rather than omitted or guessed.
            assert set(enum) - {None} == allowed, f"{key} enum {enum} != {sorted(allowed)}"


def test_precision_enums_are_the_supported_dtypes(schema):
    for key in ("weight_precision", "kv_precision"):
        for enum in _enums(schema, key):
            unknown = set(enum) - set(DTYPE_BYTES) - {None}
            assert not unknown, f"{key} allows dtypes capacity.py cannot size: {sorted(unknown)}"


# --- model schema: attention family drives which geometry is mandatory ---------------

MODEL_SCHEMA = json.loads((ROOT / "schemas" / "model.schema.json").read_text())

BASE_MODEL = dict(
    ascep_version="0.1.0",
    model_id="example/model",
    revision="0" * 40,
    total_params=26_000_000_000,
    active_params=4_000_000_000,
    architecture="dense",
    weight_precision="bf16",
    kv_precision="bf16",
    n_layers=48,
    global_layer_frac=1.0,
    native_max_context_tokens=32768,
    weight_bytes_on_disk=52_000_000_000,
    licence="apache-2.0",
    # Both are required and non-nullable: a model that does not say which modalities it accepts
    # and which reasoning modes it has is one whose token cost per request cannot be predicted
    # at all, and thinking mode alone moves output length by two orders of magnitude.
    input_modalities=["text"],
    reasoning_modes=["non-thinking"],
)


def _valid(doc) -> bool:
    return not list(Validator(MODEL_SCHEMA).iter_errors(doc))


@pytest.mark.parametrize(
    "name,extra,expected",
    [
        ("gqa with head geometry", dict(attention_type="gqa", n_kv_heads=8, head_dim=128), True),
        ("gqa without head geometry", dict(attention_type="gqa"), False),
        (
            "mla with latent geometry",
            dict(attention_type="mla", kv_lora_rank=512, qk_rope_head_dim=64),
            True,
        ),
        ("mla without latent geometry", dict(attention_type="mla"), False),
        (
            "mla declared with only head geometry",
            dict(attention_type="mla", n_kv_heads=128, head_dim=128),
            False,
        ),
        (
            "ssm with per-sequence state",
            dict(attention_type="ssm", kv_bytes_per_sequence=262_144),
            True,
        ),
        ("ssm without per-sequence state", dict(attention_type="ssm"), False),
        (
            "sliding-window with window and global fraction",
            dict(
                attention_type="sliding-window",
                n_kv_heads=8,
                head_dim=128,
                sliding_window_tokens=4096,
                global_layer_frac=0.25,
            ),
            True,
        ),
        (
            "sliding-window without a window size",
            dict(
                attention_type="sliding-window",
                n_kv_heads=8,
                head_dim=128,
                global_layer_frac=0.25,
            ),
            False,
        ),
        (
            "unsupported precision",
            dict(attention_type="gqa", n_kv_heads=8, head_dim=128, weight_precision="int3"),
            False,
        ),
        (
            "unknown attention family",
            dict(attention_type="quantum", n_kv_heads=8, head_dim=128),
            False,
        ),
    ],
)
def test_attention_family_requires_matching_geometry(name, extra, expected):
    assert _valid({**BASE_MODEL, **extra}) is expected, name


def test_chapters_do_not_name_fields_the_schemas_reject(tmp_path):
    """The chapters tell a reader what to declare; the schemas decide whether it validates.

    A chapter naming `n_experts` while the schema requires `moe_experts` sends every reader
    who follows the documentation straight into a validation error, and it is invisible until
    someone tries. Engine flags and HF config keys are quoted deliberately in chapters 1-3
    (`gpu_memory_utilization` is vLLM's, `num_key_value_heads` is Hugging Face's), so only
    tokens that look like OUR declarations are checked.
    """
    import ascep.bench.driver as driver
    import ascep.bench.ladder as ladder
    import ascep.bench.metrics as metrics
    import ascep.bench.persist as persist
    import ascep.bench.records as records
    import ascep.bench.rereduce as rereduce
    import ascep.bench.sessions as sessions
    import ascep.bench.workloads as workloads
    import ascep.capacity as capacity

    # Every stdlib-only module a reader is expected to run. Their public names are our
    # vocabulary too: prose that says `apply_boundary_rules` is naming a function a reader
    # can call, not inventing a declaration field, and a check that could not tell the
    # difference would push writers to stop backticking real API names. persist and rereduce
    # are here because a bundle chapter that cannot write `verify_bundle` has to describe the
    # function instead, and a reader cannot grep for a description.
    modules = (
        capacity, records, metrics, ladder, driver, workloads, sessions, persist, rereduce
    )

    declared = set()
    for path in sorted((ROOT / "schemas").glob("*.schema.json")):
        stack = [json.loads(path.read_text())]
        while stack:
            node = stack.pop()
            if isinstance(node, dict):
                if isinstance(node.get("properties"), dict):
                    declared |= set(node["properties"])
                # A closed enum's members are published vocabulary in exactly the way a field
                # name is: `image_grounded` is the token an author has to write into a workload
                # layer and the token a reader greps a bundle for, and the schema is what makes
                # it spellable at all. Collecting only property names would leave a bundle
                # README unable to quote the archetype it declares, which pushes the writer to
                # paraphrase a value the reader then cannot search for.
                if isinstance(node.get("enum"), list):
                    declared |= {v for v in node["enum"] if isinstance(v, str)}
                stack.extend(node.values())
            elif isinstance(node, list):
                stack.extend(node)

    # Names the chapters may legitimately use that are not schema fields: our own function and
    # keyword-argument names, and third-party vocabulary the chapters quote on purpose.
    ours = set(declared)
    # Read the keyword names out of the signatures rather than listing them. A hand-kept list
    # fails in the direction that costs the most: adding an argument makes accurate prose about
    # it fail this test, so the pressure is on the writer to delete a true sentence.
    for module in modules:
        ours |= {n for n in dir(module) if not n.startswith("_")}
        for obj in list(vars(module).values()):
            if inspect.isclass(obj):
                ours |= {n for n in vars(obj) if not n.startswith("_")}
            for member in [obj, *(vars(obj).values() if inspect.isclass(obj) else [])]:
                if inspect.isfunction(member):
                    ours |= set(inspect.signature(member).parameters)
    # The bench config is a declaration format we own, but the loader refuses against a table
    # rather than a JSON schema, so chapter 7 documenting `bundle_dir` would otherwise read as
    # an invented field. Take the vocabulary from the table itself: a key the chapter documents
    # and the loader does not accept is precisely what this test exists to catch.
    # Both halves of the table. The optional half exists so a new capability is not a breaking
    # change to every operator's config; leaving it out here would make accurate prose about
    # `media_root` fail while a typo for it passed, which is the wrong way round.
    from ascep.bench.run import _KEY_CITATIONS, _OPTIONAL_KEY_CITATIONS

    for table in (_KEY_CITATIONS, _OPTIONAL_KEY_CITATIONS):
        ours |= set(table)
        for section_keys in table.values():
            ours |= set(section_keys)
    # `media_shape()` writes its keys straight into a published workload manifest, and most of
    # them are schema fields already -- but not all, and `media_bytes_resident` is the one a
    # reader most needs the prose to name, because it is what an operator sizes
    # `media_max_records` against. Take the keys from a real corpus rather than listing them:
    # a hand-kept copy would go stale in the direction that silences accurate prose.
    _mm_root = tmp_path / "vocab-media"
    (_mm_root / "images").mkdir(parents=True)
    (_mm_root / "images" / "a.jpg").write_bytes(b"\xff\xd8\xff\xe0" + bytes(16))
    _mm_path = tmp_path / "vocab.jsonl"
    _mm_path.write_text(
        json.dumps(
            {
                "image": "images/a.jpg",
                "width": 16,
                "height": 16,
                "conversations": [
                    {"from": "human", "value": "<image>\nWhat is happening?"},
                    {"from": "gpt", "value": "A lathe."},
                ],
                "id": "r1",
            }
        )
        + "\n"
    )
    ours |= set(
        workloads.MultimodalJsonlCorpus(
            path=_mm_path, media_root=_mm_root, transport="base64"
        ).media_shape()
    )
    # A manifest key is published vocabulary in exactly the way a schema field is -- the
    # bundle carries it and a reader greps for it -- but it is neither a config key nor a
    # dataclass field, so nothing above finds `output_basis`. Take them off a real manifest
    # for the same reason the media keys come off a real corpus.
    ours |= set(
        workloads.Workload(
            source=workloads.SyntheticCorpus(
                input_tokens=8, tokenizer=lambda text: len(text.split())
            ),
            output_plan=workloads.ModelDecidedOutput(),
            cache_policy="disabled",
            seed=1,
            think_time_s=0.0,
            run_label="vocab",
        ).manifest()
    )
    ours |= {
        # Intermediate names the chapters use when walking through a derivation by hand; they
        # are prose variables, so no signature will ever contain them.
        "total_tok_s",
        "users_thr",
        # A sampling knob, quoted as the model's own vocabulary.
        "top_k",
    }
    foreign = {
        # vLLM / SGLang / TensorRT-LLM launch flags, quoted as such
        "gpu_memory_utilization",
        "enable_prefix_caching",
        "chunked_prefill_size",
        "swap_space",
        "mem_fraction_static",
        "free_gpu_memory_fraction",
        "kv_cache_free_gpu_mem_fraction",
        "long_prefill_token_threshold",
        "max_num_tokens",
        "token_budget",
        "max_batch_size",
        "max_sequences",
        "max_input_len",
        "max_seq_len",
        "tp_size",
        "pp_size",
        # Hugging Face config.json keys
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "use_sliding_window",
        # hardware spec-sheet names
        "flops_per_s_dense",
        "ram_bytes",
        # Vision-preprocessor keys from the checkpoint's own config, quoted because chapter 9
        # tells a reader to go and read them. `longest_edge` in particular is the one that
        # caps a clip silently, and naming it is the whole point of the paragraph it appears
        # in -- a chapter that could not quote it would have to describe it, and a reader
        # cannot grep for a description.
        "longest_edge",
        "shortest_edge",
        "max_pixels",
        "min_pixels",
        "patch_size",
        "merge_size",
        "spatial_merge_size",
        "temporal_patch_size",
        # The Qwen2.5-VL preprocessor function that rounds a frame to multiples of the patch
        # size before it is tokenized. A video bundle that predicts a per-clip token count has
        # to name it, because the prediction is only checkable by someone who can find the
        # rounding rule -- paraphrased as "the resizer" it becomes an unverifiable assertion.
        "smart_resize",
        # Audio-preprocessor keys from the same checkpoint configs. A bundle that says "audio
        # is accepted and unpriceable" has to name the two keys that make it so, because the
        # claim is checkable only by reading them: `audio_token_id` is present while
        # `audio_config` is null, and a reader told only that "audio is declared" cannot tell
        # a missing tower from an unmeasured one.
        "audio_token_id",
        "audio_config",
        # Engine flags and request fields for multimodal and reasoning traffic
        "limit_mm_per_prompt",
        "mm_processor_kwargs",
        "disable_mm_preprocessor_cache",
        "enable_thinking",
        "chat_template_kwargs",
        "reasoning_effort",
        "reasoning_content",
        "image_url",
        "max_tokens",
        "max_completion_tokens",
        # OpenAI usage-block fields; the server's own token counts, which section 9.4 makes
        # the arbiter over any predicted figure
        "prompt_tokens",
        "completion_tokens",
        # Intermediate quantities of ascep.agent_profile.SessionProfile. They are not workload
        # fields and must never become any: they are the two terms kv_residency is built from,
        # and the changelog has to name them because the union-versus-sum decision behind
        # tool_blocked_seconds is the one that keeps residency below 1.0. A reader who cannot
        # grep the identifier cannot check the arithmetic.
        "generating_seconds",
        "tool_blocked_seconds",
    }
    allowed = ours | foreign

    # Everything a contributor is told to fill in, not just the chapters. The PR template once
    # asked for a `conclusion_sensitivity` field that has never existed in any schema; a
    # reviewer would have ticked the box, because a checklist item is read as authoritative.
    # The templates and issue forms are documentation with the same failure mode as prose.
    sources = sorted((ROOT / "protocol").glob("*.md"))
    sources += sorted((ROOT / "templates").glob("*.md"))
    sources += sorted((ROOT / ".github").rglob("*.yml"))
    sources += sorted((ROOT / ".github").rglob("*.md"))
    sources += [ROOT / "CONTRIBUTING.md", ROOT / "CHANGELOG.md", ROOT / "README.md"]
    sources += sorted((ROOT / "examples").rglob("README.md"))

    unknown = {}
    for doc in sources:
        if not doc.exists():
            continue
        rel = doc.relative_to(ROOT)
        for lineno, line in enumerate(doc.read_text(encoding="utf-8").splitlines(), 1):
            for tok in re.findall(r"`([a-z][a-z0-9]*(?:_[a-z0-9]+)+)`", line):
                if tok not in allowed:
                    unknown.setdefault(tok, f"{rel}:{lineno}")
    assert not unknown, (
        "docs name fields no schema declares — either add the field or fix the prose:\n  "
        + "\n  ".join(f"{k} ({v})" for k, v in sorted(unknown.items()))
    )


def test_attention_enum_matches_the_prose_chapter():
    """Chapter 2 tells readers which values are legal; the schema decides. They must agree,
    or a reader follows the documentation and gets a validation failure."""
    documented = (ROOT / "protocol" / "02-model.md").read_text()
    enum = MODEL_SCHEMA["properties"]["attention_type"]["enum"]
    missing = [v for v in enum if v is not None and f"`{v}`" not in documented]
    assert not missing, f"schema allows values chapter 2 never documents: {missing}"


def test_moe_must_declare_its_experts():
    """MoE sizing needs total params for memory and active for compute; both derive from
    the expert layout, so a bare architecture='moe' is not enough to size anything."""
    doc = {
        **BASE_MODEL,
        "architecture": "moe",
        "attention_type": "gqa",
        "n_kv_heads": 8,
        "head_dim": 128,
    }
    assert not _valid(doc)
    assert _valid({**doc, "moe_experts": 64, "moe_top_k": 8})


def test_a_reported_itl_percentile_must_name_its_population():
    """Section 4.1: an unlabelled ITL percentile cannot be recomputed from the records.

    Pooled-gaps and per-request-mean are both routinely called "the ITL distribution" and
    they do not share a tail, so a reader handed a bare `itl_p95_s` cannot tell whether a
    stall inside one request was counted fifty times or averaged away once.
    """
    run_schema = json.loads((ROOT / "schemas" / "run.schema.json").read_text())
    validator = Validator(run_schema)
    run = json.loads((ROOT / "examples" / "moe-26b-h100-tp2" / "report.json").read_text())["run"]
    assert not list(validator.iter_errors(run)), "the published example must still validate"

    def with_first_result(**changes):
        results = [dict(run["results"][0], **changes), *run["results"][1:]]
        return {**run, "results": results}

    unlabelled = with_first_result(itl_p95_s=0.031)
    del unlabelled["results"][0]["itl_p95_s_u_reason"]
    errors = list(validator.iter_errors(unlabelled))
    assert any("itl_population" in str(e.absolute_path) for e in errors), errors

    labelled = dict(unlabelled)
    labelled["results"] = [
        dict(
            unlabelled["results"][0],
            itl_population="pooled-gaps",
            tokens_per_stream_chunk=1.0,
        ),
        *unlabelled["results"][1:],
    ]
    del labelled["results"][0]["itl_population_u_reason"]
    assert not list(validator.iter_errors(labelled))

    assert list(validator.iter_errors(with_first_result(itl_population="mean-of-medians")))


def test_a_pooled_gaps_row_must_carry_the_tokens_per_chunk_figure_that_licenses_it():
    """Section 4.4: pooling is only inter-token latency while a chunk is one decode step.

    A server under load folds several decode steps into one SSE delta, and the pooled gaps
    then measure the transport, not the decoder. On the H100 rehearsal the concurrency-128
    rung packed eight tokens into the average chunk and its pooled p95 read 0.2953 s against
    a per-request 0.0247 s -- against a 0.05 s gate, sustainable capacity moved from 2,503
    tok/s to 1,180 tok/s on nothing but the choice of population. So a row claiming the
    pooled population owes the reader the factor that makes the claim checkable. Null with a
    reason is an acceptable answer; saying nothing is not, because an unsupported pooled
    percentile is indistinguishable from a supported one and that is the whole defect.
    """
    run_schema = json.loads((ROOT / "schemas" / "run.schema.json").read_text())
    validator = Validator(run_schema)
    run = json.loads((ROOT / "examples" / "negative" / "baseline.json").read_text())["run"]
    assert not list(validator.iter_errors(run)), "the baseline row declares its factor"
    assert run["results"][0]["itl_population"] == "pooled-gaps"

    def with_first_result(row):
        return {**run, "results": [row, *run["results"][1:]]}

    stripped = dict(run["results"][0])
    del stripped["tokens_per_stream_chunk"]
    errors = list(validator.iter_errors(with_first_result(stripped)))
    assert any("tokens_per_stream_chunk" in str(e.message) for e in errors), errors

    # Unmeasured is a permitted answer, so long as the row says so: C1 takes it from there.
    unmeasured = dict(
        stripped,
        tokens_per_stream_chunk=None,
        tokens_per_stream_chunk_u_reason="(U) the client did not stamp per-chunk arrivals",
    )
    assert not list(validator.iter_errors(with_first_result(unmeasured)))

    # A per-request row owes nothing here: its denominator is the server's own token count,
    # which no amount of transport coalescing can inflate.
    per_request = dict(stripped, itl_population="per-request-mean")
    assert not list(validator.iter_errors(with_first_result(per_request)))


# --- the pixel budget is a total count, and notes carry value-justifications ----------

SERVING_SCHEMA = json.loads((ROOT / "schemas" / "serving.schema.json").read_text())
HARDWARE_SCHEMA = json.loads((ROOT / "schemas" / "hardware.schema.json").read_text())
EXAMPLE_REPORT = json.loads((ROOT / "examples" / "moe-26b-h100-tp2" / "report.json").read_text())


def test_the_pixel_budget_is_named_for_the_quantity_it_measures():
    """A field named like an edge length invites a value 4,096 times too small, on the one
    serving setting chapter 9 says binds silently; the name must say total pixels."""
    block = SERVING_SCHEMA["properties"]["media_preprocessing"]
    assert "image_pixel_budget_px" in block["required"]
    assert "image_pixel_budget_px" in block["properties"]
    assert "image_longest_edge_px" not in json.dumps(SERVING_SCHEMA)


def test_every_declaration_layer_takes_notes_on_its_declared_values():
    """Chapter 9.10 says every declaration layer gains the object. A layer that missed the
    rename would reject the note an author was told to write, at the moment they write it."""
    for name in ("hardware", "model", "serving", "workload"):
        schema = json.loads((ROOT / "schemas" / f"{name}.schema.json").read_text())
        notes = schema["properties"].get("notes")
        assert notes, f"{name} declares no notes object"
        assert notes["additionalProperties"] == {"type": "string", "minLength": 1}


def test_a_note_on_a_declared_value_validates_and_the_suffix_spelling_does_not():
    """notes is the one slot for value-justifications, so additionalProperties: false must
    still refuse <field>_note -- otherwise the rejected mechanism validates too, and a
    typo'd note key survives exactly where review cannot see it."""
    validator = Validator(HARDWARE_SCHEMA)
    hardware = EXAMPLE_REPORT["hardware"]
    assert not list(validator.iter_errors(hardware)), "the published example must still validate"
    noted = {**hardware, "notes": {"cpu_cores": "Cluster QoS allocation, not the node count"}}
    assert not list(validator.iter_errors(noted))
    suffix = {**hardware, "cpu_cores_note": "the rejected mechanism, typed anyway"}
    errors = list(validator.iter_errors(suffix))
    assert errors and any(e.validator == "additionalProperties" for e in errors)


def test_an_empty_note_is_refused_because_it_says_nothing():
    """A key with no sentence behind it is the (U)-tag failure repeated: the slot is filled,
    review sees a note, and the reader learns nothing about why the value is what it is."""
    validator = Validator(HARDWARE_SCHEMA)
    blank = {**EXAMPLE_REPORT["hardware"], "notes": {"cpu_cores": ""}}
    assert list(validator.iter_errors(blank))


# --- a nullable property in a closed object needs somewhere to put its (U) reason ------


def _permits_null(prop) -> bool:
    """Whether a report may carry this property as JSON null.

    C1 demands a justification for every null it walks, so a property that admits null
    matters to the sweep below whatever else it declares. A $ref property is not nullable:
    every sibling-layer schema it targets declares "type": "object".
    """
    if not isinstance(prop, dict) or "$ref" in prop:
        return False
    declared = prop.get("type")
    if isinstance(declared, list):
        return "null" in declared
    if declared == "null":
        return True
    if declared is not None:
        return False
    if "enum" in prop:
        return None in prop["enum"]
    if "const" in prop:
        return prop["const"] is None
    return True


#: Nullable properties inside an additionalProperties: false object that deliberately have
#: no sibling <name>_u_reason, so the only channel left for justifying them is the global
#: section-7 register. Each needs a reason why a local slot would be wrong, because the
#: default answer is to add one -- image_pixel_budget_px promised a "(U) reason" in its own
#: description while offering nowhere to write one, and only this sweep noticed.
_JUSTIFIED_ONLY_IN_SECTION_7 = {
    # The register's own field. A sibling is schema-illegal, and the section-7 remedy is
    # self-parody -- an entry whose field is "value_used", clearing its own null on the way
    # past. Exempted in _walk_nulls too: the entry around it is the justification.
    "capacity-report.schema.json /properties/unmeasured_assumptions/items: value_used",
    # "U" is already a tag in the provenance enum, so an author who does not know a row's
    # provenance writes it in band. A local _u_reason would compete with that tag for the
    # same job, and C2 -- not C1 -- owns the case that actually misleads a reader, a null
    # tag beside real numbers.
    "capacity-report.schema.json /$defs/capacityRow: provenance",
    "capacity-report.schema.json /properties/scaling/items: provenance",
    "capacity-report.schema.json /properties/sizing_result: provenance",
    "run.schema.json /properties/results/items: provenance",
    # Null on the archetype selector is a declared position, not a gap: it says the report
    # claims no vertical block, which is what every pre-v0.5.0 artifact says by omission and
    # what C9 reads as "pass". A _u_reason would demand an excuse for the default answer.
    "workload.schema.json /: archetypes",
    # agent_loop's null is set by the archetypes gate, not by the author: the allOf pins it
    # to null for every non-agent workload. Asking that machine-forced null to justify
    # itself would put a (U) reason on nine reports out of ten for saying nothing at all.
    "workload.schema.json /: agent_loop",
    # Null residency means "divide by duty_cycle", the documented v0.4.0 behaviour, so the
    # null carries a value rather than hiding one. C10 warns on it under code_agent, which
    # is the case where the default is known-wrong -- that warning, not a (U) reason, is
    # what keeps the default from being silent.
    "workload.schema.json /: kv_residency",
    # Null means the session has no declared context cap. That is a real property of a
    # workload, not an unmeasured one, and the sibling would be justifying unboundedness.
    "workload.schema.json /properties/agent_loop: session_max_context_tokens",
}


def _scan_for_locally_unjustifiable_nulls(owning, node, pointer, docs, seen, hits) -> None:
    """Collect every nullable property of every closed object in the schema corpus.

    Local "#/..." refs are not followed: their targets are literal subtrees of the same
    document and the plain walk reaches them. Sibling-file refs ARE followed, by filename,
    with `seen` as the recursion guard -- without it two schemas that ref each other would
    loop forever and the trap this exists to catch would never be reported. Findings are
    attributed to the file that owns the offending node, so a ref hop does not disguise
    where the fix belongs.
    """
    if isinstance(node, list):
        for index, item in enumerate(node):
            _scan_for_locally_unjustifiable_nulls(
                owning, item, f"{pointer}/{index}", docs, seen, hits
            )
        return
    if not isinstance(node, dict):
        return
    identity = (owning, id(node))
    if identity in seen:
        return
    seen.add(identity)
    ref = node.get("$ref")
    if isinstance(ref, str) and not ref.startswith("#") and ref in docs:
        _scan_for_locally_unjustifiable_nulls(ref, docs[ref], "", docs, seen, hits)
    properties = node.get("properties")
    if node.get("additionalProperties") is False and isinstance(properties, dict):
        for name, prop in properties.items():
            if _permits_null(prop) and f"{name}_u_reason" not in properties:
                hits.add(f"{owning} {pointer or '/'}: {name}")
    for key, child in node.items():
        _scan_for_locally_unjustifiable_nulls(owning, child, f"{pointer}/{key}", docs, seen, hits)


def test_every_nullable_property_in_a_closed_object_can_be_justified_where_it_sits():
    """The value_used defect is a shape, not an instance. A nullable property in a closed
    object with no <name>_u_reason sibling can only be justified in section 7, one entry
    for every instance of that field in the report -- and for a field whose own description
    promises a "(U) reason", that is a promise the schema cannot keep. Sweep the corpus so
    the next one is caught here rather than by a contributor whose honest report will not
    pass the checker."""
    docs = {
        path.name: json.loads(path.read_text())
        for path in sorted((ROOT / "schemas").glob("*.schema.json"))
    }
    hits: set[str] = set()
    seen: set[tuple[str, int]] = set()
    for name, doc in docs.items():
        _scan_for_locally_unjustifiable_nulls(name, doc, "", docs, seen, hits)
    offenders = hits - _JUSTIFIED_ONLY_IN_SECTION_7
    assert not offenders, (
        "these properties admit null with no sibling <name>_u_reason to justify it in "
        "place, so C1 can only be satisfied from section 7. Add the sibling property, "
        "make the field non-nullable, or list it in _JUSTIFIED_ONLY_IN_SECTION_7 with the "
        "reason a local slot would be wrong:\n  " + "\n  ".join(sorted(offenders))
    )
    stale = _JUSTIFIED_ONLY_IN_SECTION_7 - hits
    assert not stale, (
        "exemptions matching no schema property; a dead exemption reads as standing "
        "permission to skip the sibling rule, so delete or re-justify:\n  "
        + "\n  ".join(sorted(stale))
    )


def test_no_u_reason_property_justifies_a_field_that_does_not_exist():
    """A justification slot for an absent field is worse than no slot: an author fills it,
    the null it was meant to cover still fails C1, and nothing says why. sizing_result
    carried floor_crossover_u_reason beside floor_crossover_context_tokens_u_reason, and
    only the second one was ever consulted."""
    orphans = []
    for path in sorted((ROOT / "schemas").glob("*.schema.json")):
        doc = json.loads(path.read_text())

        def walk(node, pointer, file=path.name):
            if isinstance(node, list):
                for index, item in enumerate(node):
                    walk(item, f"{pointer}/{index}")
                return
            if not isinstance(node, dict):
                return
            properties = node.get("properties")
            if isinstance(properties, dict):
                for name in properties:
                    if name.endswith("_u_reason") and name[: -len("_u_reason")] not in properties:
                        orphans.append(f"{file} {pointer or '/'}: {name}")
            for key, child in node.items():
                walk(child, f"{pointer}/{key}")

        walk(doc, "")
    assert not orphans, "\n  ".join(["justifications for fields no schema declares:"] + orphans)
