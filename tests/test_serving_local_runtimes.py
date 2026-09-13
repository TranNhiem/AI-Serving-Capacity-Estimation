"""Release 0.7.0 taught serving.schema.json that not every runtime batches like vLLM, and
conditionals nobody tests are conditionals that regress silently.

The failure modes these tests guard against:
  * a serial runtime (llama-cli, mlx_lm.generate, a default Ollama install) inheriting a
    concurrency claim by arithmetic: aggregate tok/s divided by per-user demand yields a
    user count the machine physically cannot honour, and only parallel_slots: 1 stops the
    throughput floor from multiplying by it;
  * a slotted runtime quoting the undivided context window, over-promising one user's KV by
    exactly the slot count, because nothing demanded context_tokens_per_slot;
  * kv_cache_quantized: "on" with no per-tensor element type -- a declaration that leaves
    the KV floor uncomputable while looking more honest than "off";
  * a report pinned to nothing -- no container digest, no build string -- so "llama.cpp,
    Q4_K_M" identifies neither the code nor the bytes;
  * an if/then branch demanding a field that can never exist, unsatisfiable with no test
    noticing (the same guard 0.6.0 added for the hardware branches).
"""

import copy
import json
import pathlib

from jsonschema import Draft202012Validator as Validator

ROOT = pathlib.Path(__file__).parent.parent
SCHEMA = json.loads((ROOT / "schemas" / "serving.schema.json").read_text())

# The published GB200 vLLM declaration is the base every local-runtime fixture mutates
# from; it is read from the repository, not copied, because a hand-copied fixture drifts
# away from the file it claims to mirror.
BASE = json.loads((ROOT / "examples" / "gb200-gemma4-31b-tp1" / "serving.json").read_text())

REMOVE = object()


def _declaration(**overrides):
    """Deep-copy the base declaration and apply one mutation per keyword; REMOVE deletes."""
    doc = copy.deepcopy(BASE)
    for key, value in overrides.items():
        if value is REMOVE:
            doc.pop(key, None)
        else:
            doc[key] = value
    return doc


# The base migrated the way section B intends: eight new declarations, seven of them null
# with a (U) reason. runtime_build is the exception -- this example's container_digest is
# already null, so the pin-something rule (conditional 4) makes an eighth null a violation,
# and the build string is the price of staying uncontainerised.
MIGRATED_VLLM = dict(
    ascep_version="0.7.0",
    runtime_class="server-batched",
    runtime_build="vllm 0.18.2rc1.dev73+gdb7a17ecc (git db7a17ecc)",
    parallel_slots=None,
    parallel_slots_u_reason=(
        "(U) on a server-batched engine admitted concurrency lives in the scheduler, not "
        "a slot count; no separate cap was declared"
    ),
    context_tokens_per_slot=None,
    context_tokens_per_slot_u_reason=(
        "(U) a server-batched engine has no slots; max_model_len bounds every sequence"
    ),
    context_overflow_policy=None,
    context_overflow_policy_u_reason=(
        "(U) the run predates the field; admission behaviour for over-long prompts was "
        "not verified against this build's source"
    ),
    kv_cache_type_k=None,
    kv_cache_type_k_u_reason=(
        "(U) kv_cache_quantized is off; the element type follows the model default and "
        "was not read back from the engine"
    ),
    kv_cache_type_v=None,
    kv_cache_type_v_u_reason="(U) as for kv_cache_type_k",
    weights_residency=None,
    weights_residency_u_reason=(
        "(U) the run predates the field; weights were loaded from the container image "
        "into GPU memory but residency under pressure was not separately verified"
    ),
    model_artifact_sha256=None,
    model_artifact_sha256_u_reason=(
        "(U) the weights were read from a shared filesystem snapshot whose digest was "
        "not recorded at capture time"
    ),
)


def _valid(doc):
    return not list(Validator(SCHEMA).iter_errors(doc))


def test_runtime_class_is_the_one_field_with_no_null_and_no_excuse():
    """The discriminator keys every conditional added in this release: a document that
    omits it falls into no branch at all, and a document that spells it freely
    ("one-at-a-time") dodges all five on a typo. Required-ness and the enum are one guard.
    """
    assert not _valid(_declaration(**dict(MIGRATED_VLLM, runtime_class=REMOVE)))
    assert not _valid(_declaration(**dict(MIGRATED_VLLM, runtime_class="sometimes")))
    for member in ("server-batched", "slotted", "serial"):
        assert member in SCHEMA["properties"]["runtime_class"]["enum"], member


def _serial(**extra):
    """A serial mlx_lm fixture: everything server-shaped comes out, the 1 goes in."""
    fixture = dict(
        MIGRATED_VLLM,
        framework="mlx_lm",
        framework_version="0.24.0",
        runtime_class="serial",
        runtime_build="mlx-lm 0.24.0 (pip wheel, macOS 15.4 arm64)",
        parallel_slots=1,
        parallel_slots_u_reason=REMOVE,
        batching_mode=None,
        batching_mode_u_reason=(
            "(U) mlx_lm.generate admits one request at a time; there is no scheduler "
            "for the field to name"
        ),
        context_tokens_per_slot=None,
        context_tokens_per_slot_u_reason=(
            "(U) a serial runtime has no slots; max_model_len bounds the single session"
        ),
        context_overflow_policy="refuse",
        context_overflow_policy_u_reason=REMOVE,
        kv_cache_quantized="off",
        kv_cache_type_k="f16",
        kv_cache_type_k_u_reason=REMOVE,
        kv_cache_type_v="f16",
        kv_cache_type_v_u_reason=REMOVE,
        weights_residency="resident",
        weights_residency_u_reason=REMOVE,
        model_artifact_sha256="3bf2" * 16,
        model_artifact_sha256_u_reason=REMOVE,
        notes={
            "parallel_slots": (
                "A second request while the first generates is queued by the harness, "
                "never admitted; the throughput floor may multiply by exactly 1."
            ),
        },
    )
    fixture.update(extra)
    return _declaration(**fixture)


def _slotted(**extra):
    """A llama-server --parallel fixture: a divided window, per-tensor KV types, no image."""
    fixture = dict(
        MIGRATED_VLLM,
        framework="llama.cpp",
        framework_version="b4589",
        runtime_class="slotted",
        runtime_build="llama.cpp b4589 (f7f1d9a)",
        parallel_slots=4,
        parallel_slots_u_reason=REMOVE,
        context_tokens_per_slot=8192,
        context_tokens_per_slot_u_reason=REMOVE,
        batching_mode=None,
        batching_mode_u_reason=(
            "(U) llama-server --parallel schedules independent slots; there is no "
            "shared batched forward pass for the field to describe"
        ),
        context_overflow_policy="shift",
        context_overflow_policy_u_reason=REMOVE,
        kv_cache_quantized="on",
        kv_cache_type_k="q8_0",
        kv_cache_type_k_u_reason=REMOVE,
        kv_cache_type_v="f16",
        kv_cache_type_v_u_reason=REMOVE,
        weights_residency="mmap-paged",
        weights_residency_u_reason=REMOVE,
        model_artifact_sha256=None,
        model_artifact_sha256_u_reason=(
            "(U) hashing a 62 GB weights file was declined; the quantisation name and "
            "requantiser are on record instead"
        ),
        notes={
            "context_tokens_per_slot": (
                "llama-server -c 32768 --parallel 4 divides the window: each slot bounds "
                "one session at 8192 tokens, not the 32768 a reader quotes off the "
                "command line."
            ),
        },
    )
    fixture.update(extra)
    return _declaration(**fixture)


def test_a_serial_runtime_must_write_its_single_slot_as_a_number():
    """parallel_slots is the number the throughput floor multiplies by. On a serial runtime
    a null is a shrug and any integer above 1 declares a concurrency the engine never
    admitted; the branch must force the literal 1, present and non-null.
    """
    assert not _valid(_serial(parallel_slots=REMOVE))
    assert not _valid(_serial(parallel_slots=None))
    assert not _valid(_serial(parallel_slots=2))
    assert _valid(_serial())


def test_a_server_runtime_may_declare_more_than_one_admitted_request():
    """The const-1 branch fires on serial only. On server-batched, parallel_slots is an
    optional integer that max_num_seqs usually carries, and a declared value must
    validate -- otherwise the branch leaks sideways and gags a legitimate declaration.
    """
    assert _valid(_declaration(**dict(MIGRATED_VLLM, parallel_slots=8)))


def test_a_serial_runtime_has_no_scheduler_and_must_not_invent_one():
    """A runtime that admits one request has no batching discipline, and a serial
    declaration quoting "continuous" claims a scheduler it does not have -- the exact
    over-promise the runtime_class split exists to catch. batching_mode must be null, and
    the conformance check's own (U) demand then applies on its own.
    """
    assert not _valid(_serial(batching_mode="continuous"))
    assert not _valid(_serial(batching_mode="static"))
    assert _valid(_serial())


def test_a_server_runtime_keeps_its_scheduler_vocabulary():
    """The null branch must not leak onto the runtime the field was written for:
    batching_mode describes the scheduler inside a server-batched engine, and forbidding
    it there would silently un-declare continuous batching on every existing vLLM report.
    """
    assert _valid(_declaration(**dict(MIGRATED_VLLM, batching_mode="continuous")))
    assert _valid(_declaration(**dict(MIGRATED_VLLM, batching_mode="static")))


def test_a_slotted_runtime_must_divide_its_window_among_slots():
    """A slotted engine splits its configured context across slots, so the per-slot figure
    -- not max_model_len -- bounds one user's KV. Missing or null, a reader sizes KV for
    four concurrent full-window sessions and over-promises by exactly the slot count. The
    slot count itself must be present for the same reason.
    """
    assert not _valid(
        _slotted(context_tokens_per_slot=REMOVE, context_tokens_per_slot_u_reason=REMOVE)
    )
    assert not _valid(_slotted(context_tokens_per_slot=None))
    assert not _valid(_slotted(parallel_slots=REMOVE, parallel_slots_u_reason=REMOVE))
    assert not _valid(_slotted(parallel_slots=None))
    assert _valid(_slotted())


def test_a_server_runtime_is_not_asked_for_a_per_slot_window():
    """The slotted branch adds requirements in one direction only. A server-batched engine
    has no slots, so demanding the field of it would force a fabricated number onto every
    vLLM declaration that validates today.
    """
    doc = dict(
        MIGRATED_VLLM, context_tokens_per_slot=REMOVE, context_tokens_per_slot_u_reason=REMOVE
    )
    assert _valid(_declaration(**doc))


def test_kv_quantization_on_without_element_types_leaves_the_floor_uncomputable():
    """The KV floor needs bytes per element; "on" without kv_cache_type_k / _v says a
    quantisation happened and withholds its size, which is worse than "off" -- off at
    least implies a known element size. Either type missing or null must fail.
    """
    assert not _valid(_declaration(**dict(MIGRATED_VLLM, kv_cache_quantized="on")))
    assert not _valid(
        _declaration(**dict(MIGRATED_VLLM, kv_cache_quantized="on", kv_cache_type_k="q8_0"))
    )
    assert not _valid(
        _declaration(**dict(MIGRATED_VLLM, kv_cache_quantized="on", kv_cache_type_v="q4_0"))
    )
    assert _valid(
        _declaration(
            **dict(
                MIGRATED_VLLM,
                kv_cache_quantized="on",
                kv_cache_type_k="q8_0",
                kv_cache_type_v="q4_0",
            )
        )
    )


def test_kv_quantization_off_may_leave_the_types_unmeasured():
    """ "off" implies the model's default element size, so the floor stays computable and
    the per-tensor types are honestly null-with-a-reason. Refusing that branch would
    punish the declaration that told the reader more, and the migration fixture is its
    witness.
    """
    assert _valid(_declaration(**MIGRATED_VLLM))


def test_a_report_that_pins_nothing_is_graded_down_rather_than_refused():
    """R11 asks a report to pin its code by a container digest, a released framework_version
    or a runtime_build, and this is the test that the schema deliberately does NOT enforce it.

    A fifth conditional was drafted -- container_digest null implies runtime_build non-null --
    and cut after checking the corpus: examples/bench-config, gb200-qwen25-vl-32b-multi-image
    and moe-26b-h100-tp2 each declare both null, with a (U) reason saying in as many words
    that the report is thereby partial. ASCEP already grades that. A validator rejection would
    not produce a pinned report; it would produce an invented build string, which is the exact
    failure R11 exists to prevent. If someone reinstates that branch, this test is what tells
    them three published examples stop validating.
    """
    unpinned = dict(
        MIGRATED_VLLM,
        framework_version=None,
        framework_version_u_reason="(U) not recorded at run time",
        runtime_build=None,
        runtime_build_u_reason="(U) the build was not recorded",
    )
    assert _valid(_declaration(**unpinned))


def test_the_migrated_vllm_declaration_still_validates():
    """The intended migration cost is seven lines and no moved number: five statements --
    runtime_class, context_overflow_policy, weights_residency and the two 'n-a' KV types --
    plus two honest refusals. Anything stricter breaks every published vLLM report.
    """
    doc = _declaration(**MIGRATED_VLLM)
    assert _valid(doc), [e.message for e in Validator(SCHEMA).iter_errors(doc)]


def test_a_complete_llama_server_slotted_declaration_validates():
    """llama-server --parallel is the case the slotted branch was written for: a divided
    window, asymmetric KV quantisation, mmap-paged weights, no container image. If this
    declaration does not validate, the branch exists nowhere but in the prose.
    """
    doc = _slotted()
    assert _valid(doc), [e.message for e in Validator(SCHEMA).iter_errors(doc)]


def test_a_complete_serial_mlx_declaration_validates():
    """Serial MLX is the case the release exists for at all: one admitted request, a
    refuse-on-overflow policy, resident weights, and a digest instead of a container. The
    honest version of this machine must be a pass, or the escape hatches are decorative.
    """
    doc = _serial()
    assert _valid(doc), [e.message for e in Validator(SCHEMA).iter_errors(doc)]


def test_every_conditional_requirement_names_a_field_the_schema_defines():
    """test_required_fields_are_defined walks only the top-level required array, so a typo
    inside a then.required demands a field that can never exist: a branch nobody can pass
    and no other test would notice. The release pins exactly four branches, and every name
    each branch requires must be a declared property.
    """
    clauses = SCHEMA.get("allOf", [])
    assert len(clauses) == 4, (
        "0.7.0 pins four serving conditionals; a silent fifth or a lost one changes the gate"
    )
    for clause in clauses:
        for field in clause.get("then", {}).get("required", []):
            assert field in SCHEMA["properties"], (
                f"then.required names undefined field {field!r}: the branch is unsatisfiable"
            )


def test_every_published_serving_block_declares_runtime_class():
    """A 0.7.0 migration regression guard, mirroring the 0.6.0 hardware one: an example
    without the discriminator validates nothing about which branch it belongs to, and
    published examples are what readers copy. Bundle directories are skipped on purpose --
    a bundle is manifest-pinned captured evidence whose declarations were deliberately not
    migrated, not a place to rewrite bytes under a signature.
    """
    blocks = []
    for path in sorted((ROOT / "examples").rglob("*.json")):
        if "bundle" in path.parts:
            continue
        data = json.loads(path.read_text())
        if path.name == "serving.json":
            blocks.append((path, data))
        elif path.name == "report.json" and isinstance(data.get("serving"), dict):
            blocks.append((path, data["serving"]))
    assert blocks, "no published serving blocks found -- the guard went vacuous"
    for path, block in blocks:
        assert "runtime_class" in block, path.relative_to(ROOT)
