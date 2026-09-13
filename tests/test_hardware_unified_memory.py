"""Release 0.6.0 gave hardware.schema.json its first conditional logic, and conditions nobody
tests are conditions that regress silently.

The failure modes these tests guard against:
  * a unified-memory part (DGX Spark, Apple Silicon) inventing a per-GPU VRAM number the
    silicon does not have, so the fabricated figure validates clean and feeds
    ascep.capacity.kv_pool_bytes;
  * the if/then branches drifting from the property list, so a conditional demands a field
    that can never exist and the rule becomes unsatisfiable without any test noticing;
  * the published examples straddling the 0.6.0 migration with blocks that never declare
    which memory architecture they describe.
"""

import copy
import json
import pathlib

import pytest
from jsonschema import Draft202012Validator as Validator

ROOT = pathlib.Path(__file__).parent.parent
SCHEMA = json.loads((ROOT / "schemas" / "hardware.schema.json").read_text())

# The published GB200 declaration is the base every discrete test mutates from; it is read
# from the repository, not copied, because a hand-copied fixture drifts away from the file
# it claims to mirror.
BASE = json.loads((ROOT / "examples" / "gb200-gemma4-31b-tp1" / "hardware.json").read_text())

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


# What the 0.6.0 branch demands of a unified part: the bindable budget, what kind of memory
# backs it, and whether the part holds its datasheet figures under sustained load.
UNIFIED = dict(
    memory_architecture="unified",
    vram_bytes_per_gpu=REMOVE,
    usable_memory_bytes=103_000_000_000,
    memory_type="LPDDR5X",
    thermal_behavior="not-assessed",
)


def _valid(doc):
    return not list(Validator(SCHEMA).iter_errors(doc))


def test_a_block_without_memory_architecture_is_refused_at_the_door():
    """The discriminator is the switch for every conditional below, so a declaration that
    omits it must not be left to fall into whichever branch a validator happens to evaluate.
    """
    assert not _valid(_declaration(memory_architecture=REMOVE))


def test_a_discrete_part_that_declares_no_vram_figure_is_refused():
    """Without vram_bytes_per_gpu a discrete declaration carries no memory budget at all,
    and every KV figure derived from it would be a number with no input behind it.
    """
    assert not _valid(_declaration(vram_bytes_per_gpu=REMOVE))


def test_a_unified_part_is_not_asked_for_a_per_gpu_vram_number():
    """Unified silicon has no separately bindable VRAM, so a branch that still required the
    field would force every DGX Spark and Apple report to lie to validate.
    """
    assert _valid(_declaration(**UNIFIED))


@pytest.mark.parametrize("field", ["usable_memory_bytes", "memory_type", "thermal_behavior"])
def test_a_unified_part_missing_any_unified_budget_field_is_refused(field):
    """The three unified-only fields exist because each one, left silent, produces a specific
    overstatement: usable_memory_bytes omitted leaves no budget at all, memory_type omitted
    makes a shared LPDDR pool indistinguishable from dedicated HBM next to the bandwidth
    figure, and thermal_behavior omitted lets a throttling part pose as a sustained one.
    """
    over = dict(UNIFIED)
    over[field] = REMOVE
    assert not _valid(_declaration(**over)), field


def test_a_discrete_part_is_not_burdened_with_unified_only_fields():
    """The conditionals point one way: they add requirements to unified parts, and they must
    not turn around and start demanding unified fields of the HBM parts every existing
    example declares.
    """
    for field in ("usable_memory_bytes", "memory_type", "thermal_behavior"):
        assert field not in BASE
    assert _valid(_declaration())


def test_a_unified_part_carrying_a_vram_number_is_refused_rather_than_documented():
    """This is the fabricated-partition case. Before 0.6.0 the only thing stopping a reporter
    inventing a per-accelerator memory figure on unified silicon was a sentence in the
    description, and an unenforced MUST validates clean and then feeds kv_pool_bytes. The
    branch must refuse the number, not annotate it.
    """
    # Overriding through the dict rather than a second keyword: UNIFIED already carries
    # vram_bytes_per_gpu=REMOVE, so passing it again is a duplicate-keyword TypeError, and the
    # test would fail for a reason that has nothing to do with the schema.
    fabricated = dict(UNIFIED, vram_bytes_per_gpu=96_000_000_000)
    assert not _valid(_declaration(**fabricated))


def test_a_discrete_part_quoting_a_bindable_unified_pool_is_refused():
    """The mirror case: a discrete GPU that declares usable_memory_bytes is describing the
    wrong pool -- vram_bytes_per_gpu carries its budget, and the schema must say so rather
    than let the two figures coexist for a reader to sum.
    """
    assert not _valid(_declaration(usable_memory_bytes=180_000_000_000))


def test_intra_node_interconnect_is_demanded_exactly_when_a_node_has_more_than_one_gpu():
    """Tensor-parallel claims are non-reproducible without the intra-node link, so it is
    required at gpus_per_node > 1; on a single-device part no such link exists, the field
    must not be demanded, and the sanctioned literal 'n-a' must validate -- anything else
    would force single-device reporters to fabricate an interconnect.
    """
    assert not _valid(_declaration(interconnect_intra_node=REMOVE))
    assert _valid(
        _declaration(
            gpu_count=1,
            gpus_per_node=1,
            interconnect_intra_node=REMOVE,
        )
    )
    assert _valid(
        _declaration(
            gpu_count=1,
            gpus_per_node=1,
            interconnect_intra_node="n-a",
        )
    )


def test_inter_node_interconnect_is_demanded_exactly_when_the_deployment_spans_nodes():
    """Multi-node scaling claims need the fabric named; a one-node campaign has no inter-node
    path and must not be asked to invent one. The base declaration carries the field as a
    justified null, so the checks remove the key entirely to exercise requirement, not value.
    """
    assert not _valid(
        _declaration(
            nodes=2,
            interconnect_inter_node=REMOVE,
            interconnect_inter_node_u_reason=REMOVE,
        )
    )
    assert _valid(
        _declaration(
            interconnect_inter_node=REMOVE,
            interconnect_inter_node_u_reason=REMOVE,
        )
    )


def test_flops_precision_tracks_the_flops_number_not_the_presence_of_the_field():
    """A quoted FLOP/s figure with no precision is unusable -- a sparse FP4 headline read as
    dense BF16 overstates the prefill bound by about 2x per precision step -- so the precision
    is demanded whenever the number is non-null. Apple publishes no GPU FLOP/s figure at any
    precision, so the null-with-a-reason branch is the expected honest path there and must
    be a pass, not an error.
    """
    assert not _valid(_declaration(dense_flops_precision=REMOVE))
    assert _valid(
        _declaration(
            dense_bf16_flops_per_s=None,
            dense_bf16_flops_per_s_u_reason=(
                "(U) the vendor publishes no dense FLOP/s figure at any precision"
            ),
            dense_flops_precision=REMOVE,
        )
    )


def test_exclusive_process_is_a_valid_node_exclusivity_claim():
    """No Mac can declare 'exclusive' because WindowServer always holds GPU time; without
    'exclusive-process' every Apple report would carry the HW-7 contamination banner forever
    and the banner would stop meaning anything. The enum must accept the honest value.
    """
    assert _valid(_declaration(node_exclusivity="exclusive-process"))


def test_a_complete_dgx_spark_declaration_validates():
    """The extension exists so this declaration can be written; if it does not validate, the
    unified branch is unusable on exactly the hardware it was written for."""
    spark = _declaration(
        ascep_version="0.6.0",
        gpu_model="NVIDIA GB10 Grace Blackwell",
        gpu_count=1,
        nodes=1,
        gpus_per_node=1,
        memory_architecture="unified",
        vram_bytes_per_gpu=REMOVE,
        # A conservative bindable budget, well under total RAM: the OS reservation and any
        # GPU wired-memory limit leave less than the 128 GB sticker for weights plus KV.
        usable_memory_bytes=96_000_000_000,
        memory_type="LPDDR5X",
        thermal_behavior="not-assessed",
        interconnect_intra_node="n-a",
        interconnect_inter_node="n-a",
        interconnect_inter_node_u_reason=REMOVE,
        node_exclusivity="exclusive",
        driver_version="580.126.09",
        compute_runtime_version="CUDA 13.0",
        hbm_bandwidth_bytes_s=273_000_000_000,
        dense_bf16_flops_per_s=None,
        dense_bf16_flops_per_s_u_reason=(
            "(U) the published 1 PFLOP figure is FP4 sparse; NVIDIA publishes no dense "
            "BF16 FLOP/s figure for the GB10, so the prefill theoretical tier is uncomputed"
        ),
        dense_flops_precision=REMOVE,
        cpu_model="Arm Cortex-X925 / Cortex-A725",
        cpu_cores=20,
        system_ram_bytes=137_438_953_472,
        storage_class="local NVMe",
        model_load_path="node-local NVMe",
        notes={
            # HW-9: the method behind usable_memory_bytes must be on record on a unified part.
            # (HW-8 is the pre-existing single-vs-multi-node rule, left where published
            # reports already cite it.)
            "usable_memory_bytes": (
                "Established by loading a 60 GB bf16 checkpoint and growing the KV pool until "
                "the wired-memory limit rejected further allocation; the quoted figure is the "
                "maximum the framework could actually bind, not the 128 GB of installed RAM."
            ),
        },
    )
    assert _valid(spark), [e.message for e in Validator(SCHEMA).iter_errors(spark)]


def test_a_complete_apple_m3_ultra_declaration_validates():
    """Apple Silicon is the case that justified every escape hatch at once: no separately
    versioned accelerator driver, no GPU exclusivity, no published FLOP/s, and throttling
    under sustained load. If the schema cannot take the honest version of this machine, the
    escape hatches exist nowhere but in the descriptions."""
    mac = _declaration(
        ascep_version="0.6.0",
        gpu_model="Apple M3 Ultra",
        gpu_count=1,
        nodes=1,
        gpus_per_node=1,
        memory_architecture="unified",
        vram_bytes_per_gpu=REMOVE,
        usable_memory_bytes=350_000_000_000,
        usable_memory_bytes_u_reason=REMOVE,
        memory_type="LPDDR5",
        thermal_behavior="throttles-under-sustained-load",
        interconnect_intra_node="n-a",
        interconnect_inter_node="n-a",
        interconnect_inter_node_u_reason=REMOVE,
        node_exclusivity="exclusive-process",
        driver_version="n-a",
        driver_version_u_reason=REMOVE,
        compute_runtime_version="macOS 15.4, Metal 3.2, MLX 0.24",
        hbm_bandwidth_bytes_s=819_000_000_000,
        dense_bf16_flops_per_s=None,
        dense_bf16_flops_per_s_u_reason=(
            "(U) Apple publishes no GPU FLOP/s figure at any precision; the prefill "
            "theoretical tier and roofline_efficiency are therefore null"
        ),
        dense_flops_precision=REMOVE,
        cpu_model="Apple M3 Ultra",
        cpu_cores=32,
        system_ram_bytes=549_755_813_888,
        storage_class="local NVMe",
        model_load_path="node-local NVMe",
        notes={
            "usable_memory_bytes": (
                "macOS defaults iogpu.wired_limit_mb to roughly 65-75% of RAM; the quoted "
                "figure is what the framework could bind under that limit, measured by "
                "allocation growth to failure, not the 512 GB of installed memory."
            ),
        },
    )
    assert _valid(mac), [e.message for e in Validator(SCHEMA).iter_errors(mac)]


def test_every_published_hardware_block_declares_memory_architecture():
    """A 0.6.0 migration regression guard: an example without the discriminator validates
    nothing about which branch it belongs to, and published examples are what readers copy.
    Bundle directories are skipped on purpose -- a bundle is manifest-pinned captured
    evidence whose declarations were deliberately not migrated, and rewriting those bytes
    and re-signing the manifest is exactly the laundering the manifest exists to detect.
    """
    blocks = []
    for path in sorted((ROOT / "examples").rglob("*.json")):
        if "bundle" in path.parts:
            continue
        data = json.loads(path.read_text())
        if path.name == "hardware.json":
            blocks.append((path, data))
        elif path.name == "report.json" and isinstance(data.get("hardware"), dict):
            blocks.append((path, data["hardware"]))
    assert blocks, "no published hardware blocks found -- the guard went vacuous"
    for path, block in blocks:
        assert "memory_architecture" in block, path.relative_to(ROOT)


def test_every_conditional_requirement_names_a_field_the_schema_defines():
    """test_required_fields_are_defined only walks the top-level required array, so a typo
    inside an if/then branch would demand a field that can never exist: a rule nobody could
    ever pass and no test would ever notice. Every then.required name must be a property."""
    for clause in SCHEMA.get("allOf", []):
        for field in clause.get("then", {}).get("required", []):
            assert field in SCHEMA["properties"], (
                f"then.required names undefined field {field!r}: the branch is unsatisfiable"
            )
