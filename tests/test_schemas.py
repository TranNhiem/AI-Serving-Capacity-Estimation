"""The schemas are the enforceable half of the protocol; these tests keep them honest.

Two failure modes they guard against:
  * schema vocabulary drifting away from ascep.capacity, so a report validates but the
    formulas cannot consume it;
  * declaring a model whose attention family does not match the geometry supplied, which
    silently selects the wrong KV formula.
"""

import glob
import json
import pathlib
import re

import pytest
from jsonschema import Draft202012Validator as Validator

from ascep.capacity import DTYPE_BYTES, Constraint, Tier

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
    import ascep.capacity as capacity

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
    ours = declared | {n for n in dir(capacity) if not n.startswith("_")}
    ours |= {
        "gpus_per_replica",
        "min_kv_tokens",
        "max_gpus",
        "overhead_frac",
        "batch_size",
        "peak_concurrent_users",
        "total_tok_s",
        "users_thr",
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
