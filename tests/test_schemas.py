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


def test_chapters_do_not_name_fields_the_schemas_reject():
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
    import ascep.bench.records as records
    import ascep.capacity as capacity

    # Every stdlib-only module a reader is expected to run. Their public names are our
    # vocabulary too: prose that says `apply_boundary_rules` is naming a function a reader
    # can call, not inventing a declaration field, and a check that could not tell the
    # difference would push writers to stop backticking real API names.
    modules = (capacity, records, metrics, ladder, driver)

    declared = set()
    for path in sorted((ROOT / "schemas").glob("*.schema.json")):
        stack = [json.loads(path.read_text())]
        while stack:
            node = stack.pop()
            if isinstance(node, dict):
                if isinstance(node.get("properties"), dict):
                    declared |= set(node["properties"])
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
    from ascep.bench.run import _KEY_CITATIONS

    ours |= set(_KEY_CITATIONS)
    for section_keys in _KEY_CITATIONS.values():
        ours |= set(section_keys)
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
        ),
        *unlabelled["results"][1:],
    ]
    del labelled["results"][0]["itl_population_u_reason"]
    assert not list(validator.iter_errors(labelled))

    assert list(validator.iter_errors(with_first_result(itl_population="mean-of-medians")))
