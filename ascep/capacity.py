"""Transparent capacity formulas for the AI Serving Capacity Estimation Protocol.

Every function here is a closed-form model with its assumptions stated. Nothing calls a
GPU, a server, or the network — this module is the *analytic* half of the protocol and is
meant to be readable, unit-testable, and arguable. The empirical half — the sweep driver and
the metric reducers that turn raw per-request records into the figures below — is **not in
this release**; chapter 7 specifies the procedure so an existing harness can stand in, and
``examples/*/build_report.py`` shows the hand-off.

The protocol distinguishes four capacity tiers and this module computes all four:

``THEORETICAL``  hardware roofline. Ignores framework, scheduling and kernel efficiency.
                 Always an upper bound; treat a measured/theoretical ratio above ~0.8 as
                 evidence of a measurement error, not of a fast server.
``MEASURED``     what a benchmark actually observed, SLO ignored.
``SUSTAINABLE``  measured capacity restricted to the operating points where every SLO gate
                 passed for the full sustained window.
``RECOMMENDED``  sustainable divided by a headroom factor, for production sizing.

Units are explicit in every name: ``_bytes``, ``_tokens``, ``_s``, ``_tok_s``. There are no
implicit GiB/GB conversions — use :data:`GIB` and :data:`GB`.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum

GB = 1_000_000_000
GIB = 1_073_741_824

#: Bytes per element. ``fp8``/``int8`` and the 4-bit formats refer to the *stored* weight
#: element; most runtimes keep scales and zero-points alongside, which
#: :func:`weight_bytes` accounts for with ``overhead_frac``.
DTYPE_BYTES: Mapping[str, float] = {
    "fp32": 4.0,
    "tf32": 4.0,
    "fp16": 2.0,
    "bf16": 2.0,
    "fp8": 1.0,
    "int8": 1.0,
    "fp6": 0.75,
    "nvfp4": 0.5,
    "mxfp4": 0.5,
    "int4": 0.5,
    "fp4": 0.5,
}


class Tier(str, Enum):
    THEORETICAL = "theoretical"
    MEASURED = "measured"
    SUSTAINABLE = "sustainable"
    RECOMMENDED = "recommended"


class Provenance(str, Enum):
    """How a reported number came to exist (C2). One definition, here, because the tag is
    what separates a measurement from a guess — and a benchmark harness, a validator and a
    renderer that each carry their own copy will eventually disagree about what ``T`` means
    and publish an inferred number as a measured one."""

    MEASURED = "M"  # observed on the hardware under test
    INFERRED = "I"  # computed from (M) values by a formula in this module
    THEORETICAL = "T"  # from a datasheet or a roofline; never observed
    UNMEASURED = "U"  # not established; requires a stated reason (C1)


class Constraint(str, Enum):
    """Which floor binds. Reporting this is mandatory — a capacity number without its
    binding constraint cannot be acted on, because it does not say what to buy."""

    WEIGHTS = "weights"  # the model does not fit at all
    KV = "kv"  # memory-bound: not enough KV pool for the concurrency
    THROUGHPUT = "throughput"  # compute-bound: not enough tokens/s for the demand
    PREFILL = "prefill"  # compute-bound: not enough input tokens/s for the prompts
    SLO = "slo"  # fits and is fast enough on average, but misses a latency gate


def dtype_bytes(precision: str) -> float:
    try:
        return DTYPE_BYTES[precision.lower()]
    except KeyError:
        raise ValueError(f"unknown precision {precision!r}; known: {sorted(DTYPE_BYTES)}") from None


# --------------------------------------------------------------------------- memory model


def weight_bytes(total_params: float, precision: str, overhead_frac: float = 0.0) -> float:
    """Resident weight bytes.

    ``total_params`` is *total* parameters, not active. For a Mixture-of-Experts model every
    expert must be resident even though only ``top_k`` of them run per token — this is the
    single most common MoE sizing error. Use active params only in the *compute* models
    (:func:`roofline_decode_tok_s`), never here.

    ``overhead_frac`` covers quantization scales/zero-points and any format padding; for
    bf16/fp16 it is 0.
    """
    return total_params * dtype_bytes(precision) * (1.0 + overhead_frac)


def kv_heads_per_rank(n_kv_heads: int, tensor_parallel: int) -> int:
    """KV heads materialized on each rank.

    When ``tensor_parallel > n_kv_heads`` the heads cannot be split further, so runtimes
    **replicate** them. Total KV memory then grows with TP instead of staying flat, and a
    wider deployment can hold *fewer* tokens than a narrower one. This is not a hypothetical:
    it is why grouped-query models with 2 KV heads lose capacity at TP=4.
    """
    if n_kv_heads < 1 or tensor_parallel < 1:
        raise ValueError("n_kv_heads and tensor_parallel must be >= 1")
    return max(1, math.ceil(n_kv_heads / tensor_parallel))


def effective_layer_frac(
    global_layer_frac: float,
    sliding_window_tokens: float,
    avg_context_tokens: float,
) -> float:
    """The share of full-length-equivalent KV a hybrid stack actually holds at this context.

    ``global_layer_frac`` on its own is the **asymptote**, not the answer. A local layer is
    capped at ``sliding_window_tokens``, so it holds ``min(context, window)`` tokens — which is
    the *whole* context until the context outgrows the window, and only then starts saving
    anything. Averaged over a sequence of ``avg_context_tokens``:

        frac = global_layer_frac + (1 - global_layer_frac) × min(context, window) / context

    It equals 1.0 while the context fits the window, decays as the context grows past it, and
    approaches ``global_layer_frac`` only as context goes to infinity.

    **The failure this prevents.** Declaring the asymptote at every context under-reports KV by
    up to ``1 / global_layer_frac``. On the hybrid 26B MoE in ``examples/gb200-moe-26b-tp1`` —
    30 layers, 5 of them global, a 1,024-token window, and an average context of 903 tokens —
    the asymptote 1/6 says 20,480 bytes per token when every one of the 25 local layers is
    holding the full context and the truth is 122,880. That is a **6× understatement**, and it
    is the dangerous direction: it inflates KV capacity, so it over-promises concurrent
    sessions on hardware that cannot hold them. Chapter 2 warns that the 1.0 default is
    pessimistic and therefore harmless; the asymptote is the opposite of both.

    It is also visible before deployment, which is the point of computing it. Against that
    model's engine-reported 492,000 KV tokens, the asymptote predicts 6,210,713 — 12.6× the
    engine — while the context-aware fraction predicts 1,035,119, or 2.1×, which is ordinary
    workspace-and-fragmentation territory that :func:`calibrate_memory_utilization` absorbs.
    A 12.6× disagreement is the protocol's own signal that ``global_layer_frac`` is wrong.
    """
    if not 0.0 < global_layer_frac <= 1.0:
        raise ValueError("global_layer_frac must be in (0, 1]")
    if sliding_window_tokens <= 0:
        raise ValueError("sliding_window_tokens must be > 0")
    # A zero-length context has no per-token cost to average over, and dividing by it would
    # return inf -- a KV capacity of zero tokens, reported as though it had been computed.
    if avg_context_tokens <= 0:
        raise ValueError("avg_context_tokens must be > 0 to average a windowed layer over it")
    local_share = min(avg_context_tokens, sliding_window_tokens) / avg_context_tokens
    return global_layer_frac + (1.0 - global_layer_frac) * local_share


def kv_bytes_per_token(
    n_layers: int,
    n_kv_heads: int,
    head_dim: int,
    kv_precision: str = "bf16",
    tensor_parallel: int = 1,
    global_layer_frac: float = 1.0,
    sliding_window_tokens: float | None = None,
    avg_context_tokens: float | None = None,
) -> float:
    """Cluster-wide KV bytes for one token of context, for **standard MHA/GQA/MQA** attention.

    Counts K and V (hence the factor 2) and includes the TP replication penalty above.

    ``global_layer_frac`` is the fraction of layers holding full-length KV. Models with
    sliding-window or hybrid attention only keep full context on their *global* layers; the
    local layers are capped at the window. Set it to ``n_global_layers / n_layers`` for those,
    or leave 1.0 for uniform full attention. Report the value you used — it moves KV capacity
    by several times.

    For a windowed or hybrid model, also pass ``sliding_window_tokens`` and
    ``avg_context_tokens``. ``global_layer_frac`` alone is the limit the cost approaches as
    context grows without bound; below and around the window the local layers are holding the
    full context and saving nothing, so the bare fraction under-reports KV by up to
    ``1 / global_layer_frac``. :func:`effective_layer_frac` computes what the stack holds at a
    stated context. Omit both and the fraction is used as given, which is exactly right for
    uniform attention and is what every full-attention report published so far assumed.

    **Do not use this for MLA** (DeepSeek-style multi-head latent attention), which caches a
    compressed latent instead of per-head K and V — this function overstates its KV by an
    order of magnitude. Use :func:`kv_bytes_per_token_mla`. For linear/SSM/hybrid-recurrent
    attention, per-token KV is not the right model at all: state is constant per sequence, so
    declare it via ``kv_bytes_per_sequence`` on :func:`kv_capacity_sessions` instead.
    """
    if not 0.0 < global_layer_frac <= 1.0:
        raise ValueError("global_layer_frac must be in (0, 1]")
    window_geometry = (sliding_window_tokens, avg_context_tokens)
    # One of the two alone cannot be honoured, and honouring neither is the silent failure:
    # the caller said the model is windowed, got the asymptote anyway, and the inflated KV
    # capacity it returns over-promises sessions on hardware that cannot hold them.
    if any(v is not None for v in window_geometry) and any(v is None for v in window_geometry):
        raise ValueError(
            "the windowed KV cost needs sliding_window_tokens and avg_context_tokens "
            "together: the saving is the ratio between them. Pass both, or neither to use "
            f"global_layer_frac as given. Got sliding_window_tokens={sliding_window_tokens}, "
            f"avg_context_tokens={avg_context_tokens}"
        )
    if sliding_window_tokens is not None:
        assert avg_context_tokens is not None  # narrowed by the check above
        global_layer_frac = effective_layer_frac(
            global_layer_frac, sliding_window_tokens, avg_context_tokens
        )
    effective_kv_heads = kv_heads_per_rank(n_kv_heads, tensor_parallel) * tensor_parallel
    return (
        2.0
        * n_layers
        * global_layer_frac
        * effective_kv_heads
        * head_dim
        * dtype_bytes(kv_precision)
    )


def kv_bytes_per_token_mla(
    n_layers: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    kv_precision: str = "bf16",
    global_layer_frac: float = 1.0,
) -> float:
    """Cluster-wide KV bytes per token for **multi-head latent attention** (MLA).

    MLA caches a single low-rank latent vector per layer plus a small decoupled RoPE key,
    rather than K and V for every head. Two consequences that break the standard model:

    * There is no factor of 2 and no head count. Per-token cost is
      ``(kv_lora_rank + qk_rope_head_dim)`` elements per layer — for DeepSeek-V3 that is
      ``512 + 64 = 576`` versus the tens of thousands a naive GQA calculation would give.
    * **The latent is not sharded by tensor parallelism**; every rank holds a full copy, so
      there is no ``tensor_parallel`` argument here. Cluster-wide KV bytes per token are
      *unchanged* by TP width, which means widening TP buys you compute and aggregate HBM but
      **not** proportionally more sessions. Sizing MLA with the GQA formula overstates KV
      demand by one to two orders of magnitude — 57x for DeepSeek-V3 geometry (61 layers,
      128 heads, head_dim 128), the difference between predicting 3 concurrent sessions and
      the real 153 on the same eight GPUs — and will lead you to buy GPUs you do not need.

    Take ``kv_lora_rank`` and ``qk_rope_head_dim`` verbatim from the model's ``config.json``.
    """
    if not 0.0 < global_layer_frac <= 1.0:
        raise ValueError("global_layer_frac must be in (0, 1]")
    if kv_lora_rank < 1 or qk_rope_head_dim < 0:
        raise ValueError("kv_lora_rank must be >= 1 and qk_rope_head_dim >= 0")
    return (
        n_layers * global_layer_frac * (kv_lora_rank + qk_rope_head_dim) * dtype_bytes(kv_precision)
    )


def kv_pool_bytes(
    n_gpus: int,
    vram_bytes_per_gpu: float,
    weights_bytes: float,
    memory_utilization: float = 0.90,
    activation_bytes: float = 0.0,
) -> float:
    """Bytes left for KV after weights and workspace.

    ``memory_utilization`` is the framework's own cap (vLLM ``gpu_memory_utilization``,
    SGLang ``mem_fraction_static``, ...). **Record it.** It is the most frequently omitted
    number in published benchmarks and without it a KV measurement cannot be reproduced or
    compared against another run.
    """
    pool = n_gpus * vram_bytes_per_gpu * memory_utilization - weights_bytes - activation_bytes
    return max(0.0, pool)


def kv_capacity_tokens(kv_pool: float, kv_per_token: float) -> float:
    """Total context tokens the cluster can hold at once."""
    if kv_per_token <= 0:
        raise ValueError("kv_bytes_per_token must be > 0")
    return kv_pool / kv_per_token


def kv_capacity_sessions(
    kv_pool: float,
    kv_bytes_per_token: float = 0.0,
    avg_context_tokens: float = 0.0,
    kv_bytes_per_sequence: float = 0.0,
) -> float:
    """Concurrent sequences the KV pool can hold, for attention that is *not* purely quadratic.

    Attention families differ in how state scales with context, and the difference decides
    which floor binds:

    * **Attention-based** (MHA/GQA/MQA/MLA) — state grows with context. Pass
      ``kv_bytes_per_token`` and ``avg_context_tokens``. Sessions fall as conversations
      lengthen, so long-context workloads become KV-bound.
    * **Linear attention, SSM/Mamba, and recurrent hybrids** — state is a fixed-size
      recurrent tensor per sequence, independent of context length. Pass
      ``kv_bytes_per_sequence`` only. Session capacity is then **flat in context length**, and
      such a model is essentially never KV-bound at long context — it is throughput-bound
      everywhere. Applying the per-token model to one of these predicts a long-context cliff
      that does not exist.
    * **Hybrid stacks** (some layers attention, some recurrent) — pass **both**; the per-layer
      costs add. Split the per-token term across only the attention layers using
      ``global_layer_frac``.

    A report MUST state which of these it assumed; the three give materially different answers
    for the same hardware.
    """
    per_seq = kv_bytes_per_sequence + kv_bytes_per_token * avg_context_tokens
    if per_seq <= 0:
        raise ValueError(
            "supply kv_bytes_per_token with avg_context_tokens, or kv_bytes_per_sequence, "
            "or both for a hybrid stack"
        )
    return kv_pool / per_seq


def calibrate_memory_utilization(
    measured_kv_tokens: float,
    n_gpus: int,
    vram_bytes_per_gpu: float,
    weights_bytes: float,
    kv_per_token: float,
) -> float:
    """Back-solve the effective utilization from an engine-reported KV cache size.

    **Prefer this over the analytic path whenever the engine prints its KV cache size.** The
    analytic model cannot see workspace, fragmentation, CUDA graph capture, or the framework's
    own reservations; a measured token count already contains all of them. Feeding the result
    back in makes subsequent projections at other context lengths consistent with reality.

    A returned value above 1.0 means the inputs disagree with the measurement — most often a
    wrong ``global_layer_frac``, a wrong KV precision, or KV offloading being active.
    """
    denom = n_gpus * vram_bytes_per_gpu
    if denom <= 0:
        raise ValueError("n_gpus and vram_bytes_per_gpu must be > 0")
    return (measured_kv_tokens * kv_per_token + weights_bytes) / denom


def fits(
    weights_bytes_: float,
    n_gpus: int,
    vram_bytes_per_gpu: float,
    memory_utilization: float = 0.90,
    min_kv_tokens: float = 0.0,
    kv_per_token: float = 0.0,
) -> bool:
    """Whether the model loads *and* leaves a usable KV pool.

    A deployment that loads but has near-zero KV is not viable — it serializes to batch-size-1
    and its latency collapses. Pass ``min_kv_tokens`` to assert a floor; asserting one requires
    ``kv_per_token``, since the floor is in tokens and the pool is in bytes.
    """
    # Returning True here would be the worst available answer: the caller asked for the floor
    # to be enforced, the floor was silently skipped, and a "fits" verdict came back for a
    # configuration nobody checked. The two arguments are one assertion and must arrive together.
    if min_kv_tokens > 0 and kv_per_token <= 0:
        raise ValueError(
            "min_kv_tokens needs kv_per_token to be checkable: the floor is a token count and "
            "the pool is bytes. Pass kv_bytes_per_token(...) for the deployed attention "
            "geometry, or drop min_kv_tokens if you only mean 'the weights load'."
        )
    pool = kv_pool_bytes(n_gpus, vram_bytes_per_gpu, weights_bytes_, memory_utilization)
    if pool <= 0:
        return False
    if min_kv_tokens > 0:
        return kv_capacity_tokens(pool, kv_per_token) >= min_kv_tokens
    return True


# ------------------------------------------------------------------------ roofline (compute)


def moe_decode_weight_params(
    active_params: float,
    total_params: float,
    moe_experts: int,
    moe_top_k: int,
    batch_size: int,
) -> float:
    """Parameters an MoE reads from HBM in one decode step at ``batch_size``.

    ``active_params`` is a *per-token* figure, and a decode step does not process one token —
    it processes ``batch_size`` of them, each routed to its own ``top_k`` experts. The step must
    read the union of everything they picked, not one token's share of it. Each token routes
    independently, so an expert escapes being read only if all ``batch_size`` tokens miss it:

        expected_experts = moe_experts * (1 - (1 - moe_top_k / moe_experts) ** batch_size)

    The union saturates fast. At 128 experts and top-8, batch 1 touches 8, batch 32 touches
    112, and by batch 192 essentially all 128 are read every step — so a large-batch MoE streams
    close to its *total* parameters per step, not its active ones.

    **The failure this prevents.** Pricing every step at ``active_params`` treats batching as
    free for the weight term, which is true for a dense model and false for this one. On a
    26B/3.8B MoE at batch 476 it understates weight traffic 6.8x, and since the roofline sits in
    the denominator of ``roofline_efficiency`` the error lands there inverted: a server running
    at a real 0.16 of its bound is published at 0.04, and the C6 rule that reads an efficiency
    near 1.0 as evidence of measurement error is left checking a number too small to trip it.

    Assumes tokens route independently and uniformly. Trained routers are pushed toward balance,
    which spreads tokens across *more* distinct experts than chance would, so this is a floor on
    the parameters read and therefore keeps the enclosing throughput figure an upper bound.
    """
    if not 0 < moe_top_k <= moe_experts:
        raise ValueError(
            f"moe_top_k must be in (0, moe_experts]; got top_k={moe_top_k}, experts={moe_experts}"
        )
    # Below the trunk there is nothing to route: a model whose total equals its active count has
    # no expert bank, and the per-expert size solved for below would come out zero or negative
    # and quietly shrink the step's weight read as the batch grew.
    if total_params < active_params:
        raise ValueError(
            f"total_params ({total_params:,.0f}) is below active_params ({active_params:,.0f}); "
            "an MoE stores every expert and runs top_k of them, so total is never the smaller"
        )
    if moe_experts == moe_top_k:
        # Every token already routes to every expert, so there is no union to compute and
        # active_params is exact. Solving for the per-expert size here would divide by zero.
        return active_params
    per_expert = (total_params - active_params) / (moe_experts - moe_top_k)
    trunk = active_params - moe_top_k * per_expert
    if batch_size < 1:
        return active_params
    touched = moe_experts * (1.0 - (1.0 - moe_top_k / moe_experts) ** batch_size)
    return trunk + touched * per_expert


def roofline_decode_tok_s(
    active_params: float,
    precision: str,
    hbm_bandwidth_bytes_s: float,
    batch_size: int = 1,
    avg_context_tokens: float = 0.0,
    kv_per_token: float = 0.0,
    efficiency: float = 1.0,
    total_params: float | None = None,
    moe_experts: int | None = None,
    moe_top_k: int | None = None,
) -> float:
    """Upper bound on decode throughput, from memory bandwidth.

    Autoregressive decode is bandwidth-bound, not FLOP-bound: each step streams the weights it
    needs once and re-reads the KV cache for every sequence in the batch. Batching amortizes
    the weight read across ``batch_size`` tokens, which is why throughput rises with
    concurrency until KV traffic takes over.

    ``active_params`` is per-token active parameters — for MoE that is the shared trunk plus
    ``top_k`` experts, **not** total params. The gap between this bound and measurement is the
    protocol's ``roofline_efficiency`` and is a required reporting field: real servers land far
    below 1.0, and a value near or above 1.0 indicates the measurement is wrong.

    For an MoE, also pass ``total_params``, ``moe_experts`` and ``moe_top_k``. Per-token active
    params price the weight read correctly only at ``batch_size`` 1; above it the step reads the
    union of the experts the whole batch selected, which
    :func:`moe_decode_weight_params` computes and which approaches total params well before a
    production batch size. Omit all three and the weight read is ``active_params`` flat, which
    is exactly right for a dense model and is what every dense report published so far assumed.
    """
    if hbm_bandwidth_bytes_s <= 0:
        raise ValueError("hbm_bandwidth_bytes_s must be > 0")
    moe_geometry = (total_params, moe_experts, moe_top_k)
    # Silently falling back to the dense path on a partial declaration is the dangerous
    # outcome: the caller asked for the expert-union model, got the flat one, and the
    # understated roofline it returns looks like an ordinary number rather than a mistake.
    if any(v is not None for v in moe_geometry) and any(v is None for v in moe_geometry):
        raise ValueError(
            "the MoE weight read needs total_params, moe_experts and moe_top_k together: the "
            "expert-union size is solved from all three. Pass all of them, or none for a dense "
            f"model. Got total_params={total_params}, moe_experts={moe_experts}, "
            f"moe_top_k={moe_top_k}"
        )
    if total_params is None:
        weight_params = active_params
    else:
        assert moe_experts is not None and moe_top_k is not None  # narrowed by the check above
        weight_params = moe_decode_weight_params(
            active_params, total_params, moe_experts, moe_top_k, batch_size
        )
    weight_read = weight_params * dtype_bytes(precision)
    kv_read = batch_size * avg_context_tokens * kv_per_token
    bytes_per_step = weight_read + kv_read
    if bytes_per_step <= 0:
        raise ValueError("degenerate roofline inputs")
    steps_per_s = hbm_bandwidth_bytes_s / bytes_per_step
    return steps_per_s * batch_size * efficiency


def roofline_prefill_ttft_s(
    active_params: float,
    prompt_tokens: float,
    flops_per_s: float,
    mfu: float = 0.4,
) -> float:
    """Lower bound on time-to-first-token, from dense compute.

    Prefill is FLOP-bound at roughly ``2 * active_params`` FLOPs per prompt token. ``mfu`` is
    model-FLOPs-utilization; 0.3-0.5 is typical for a well-tuned server. Excludes queueing,
    tokenization, scheduling and network — so real TTFT under load is always larger, often by
    an order of magnitude once requests queue behind each other.
    """
    if flops_per_s <= 0 or not 0 < mfu <= 1:
        raise ValueError("flops_per_s must be > 0 and mfu in (0, 1]")
    return (2.0 * active_params * prompt_tokens) / (flops_per_s * mfu)


# ---------------------------------------------------------------------- media token model


def image_tokens(
    width_px: int,
    height_px: int,
    *,
    policy: str,
    patch_px: int | None = None,
    spatial_merge: int = 1,
    fixed_tokens: int | None = None,
    table: Sequence[Mapping[str, int]] | None = None,
    longest_edge_px: int | None = None,
    tokens_min: int | None = None,
    tokens_max: int | None = None,
) -> int:
    """Visual tokens for one image, under the model's declared ``image_token_policy``.

    ``policy`` must be one of ``fixed-grid``, ``dynamic-resolution`` or ``declared-table``.
    An unknown policy, or a policy whose required argument is missing, raises
    :class:`ValueError` naming the field - a silent default here becomes a wrong token
    count in every downstream floor.

    ``fixed-grid`` returns ``fixed_tokens`` verbatim: every image costs the same.
    ``declared-table`` returns the ``tokens`` of the smallest table row whose
    ``max_width``/``max_height`` both cover the input, and raises if no row does.

    ``dynamic-resolution`` is the real one, and follows the preprocessor in order:

    1. If ``longest_edge_px`` is set and ``width_px * height_px`` exceeds it, both
       dimensions are scaled by ``sqrt(longest_edge_px / (width_px * height_px))``,
       preserving aspect ratio. Despite the name it carries upstream, this is an **area**
       budget, not a per-side limit.
    2. Each dimension is rounded UP to a multiple of ``patch_px * spatial_merge`` - a
       partial patch is still a patch.
    3. ``tokens = (w / patch_px) * (h / patch_px) / spatial_merge ** 2``, as an integer.
    4. The result is clamped into ``[tokens_min, tokens_max]`` when either is given.

    Sanity check, from a real config: ``patch_px=16`` with ``spatial_merge=2`` gives one
    visual token per 32x32 px, so the default ``longest_edge_px`` of 25,165,824 caps any
    single input at 24,576 visual tokens.
    """
    if width_px < 1 or height_px < 1:
        raise ValueError("width_px and height_px must be >= 1")
    if policy == "fixed-grid":
        if fixed_tokens is None:
            raise ValueError("fixed_tokens is required for policy 'fixed-grid'")
        return fixed_tokens
    if policy == "declared-table":
        if table is None:
            raise ValueError("table is required for policy 'declared-table'")
        covering = [
            row for row in table if row["max_width"] >= width_px and row["max_height"] >= height_px
        ]
        if not covering:
            raise ValueError(
                f"no declared-table row covers {width_px}x{height_px}; the image would be "
                "rejected or rescaled by the preprocessor, and either way this token count "
                "would be wrong"
            )
        return min(covering, key=lambda row: row["max_width"] * row["max_height"])["tokens"]
    if policy == "dynamic-resolution":
        if patch_px is None:
            raise ValueError("patch_px is required for policy 'dynamic-resolution'")
        if patch_px < 1 or spatial_merge < 1:
            raise ValueError("patch_px and spatial_merge must be >= 1")
        unit = patch_px * spatial_merge
        w = float(width_px)
        h = float(height_px)
        if longest_edge_px is not None and w * h > longest_edge_px:
            scale = math.sqrt(longest_edge_px / (w * h))
            w *= scale
            h *= scale
        w_padded = math.ceil(w / unit) * unit
        h_padded = math.ceil(h / unit) * unit
        tokens = (w_padded // patch_px) * (h_padded // patch_px) // (spatial_merge**2)
        if tokens_min is not None:
            tokens = max(tokens, tokens_min)
        if tokens_max is not None:
            tokens = min(tokens, tokens_max)
        return tokens
    raise ValueError(
        f"unknown image token policy {policy!r}; "
        "known: fixed-grid, dynamic-resolution, declared-table"
    )


def video_frames(
    duration_s: float,
    *,
    policy: str,
    sampling_fps: float | None = None,
    frame_count: int | None = None,
    max_frames: int | None = None,
) -> int:
    """Frames sampled from a clip, under the model's declared ``video_frame_policy``.

    ``uniform-fps`` uses ``ceil(duration_s * sampling_fps)``; ``uniform-count`` returns
    ``frame_count`` regardless of duration; ``native-timestamped`` requires ``sampling_fps``
    and behaves like ``uniform-fps``. A positive duration always yields at least one frame.

    ``max_frames`` is applied as a hard clamp last. That clamp is the reason a long clip and
    a short one can cost the same - which is exactly the failure the calibration check in
    :func:`media_token_cap_check` exists to catch, so never treat this function's output as
    evidence that cost scales with duration.
    """
    if policy in ("uniform-fps", "native-timestamped"):
        if sampling_fps is None:
            raise ValueError(f"sampling_fps is required for policy {policy!r}")
        frames = math.ceil(duration_s * sampling_fps)
    elif policy == "uniform-count":
        if frame_count is None:
            raise ValueError("frame_count is required for policy 'uniform-count'")
        frames = frame_count
    else:
        raise ValueError(
            f"unknown video frame policy {policy!r}; "
            "known: uniform-fps, uniform-count, native-timestamped"
        )
    if duration_s > 0:
        frames = max(1, frames)
    if max_frames is not None:
        frames = min(frames, max_frames)
    return frames


def video_tokens(
    duration_s: float,
    width_px: int,
    height_px: int,
    *,
    frame_policy: str,
    sampling_fps: float | None = None,
    frame_count: int | None = None,
    max_frames: int | None = None,
    temporal_merge: int = 1,
    image_policy: str = "dynamic-resolution",
    patch_px: int | None = None,
    spatial_merge: int = 1,
    fixed_tokens: int | None = None,
    longest_edge_px: int | None = None,
    tokens_max: int | None = None,
) -> int:
    """Visual tokens for a whole clip.

    Frames come from :func:`video_frames`, are grouped by ``temporal_merge``
    (``groups = ceil(frames / temporal_merge)``), and each group costs the per-frame count
    from :func:`image_tokens` computed WITHOUT the area budget - the real preprocessor does
    not apply it per frame.

    ``longest_edge_px`` and ``tokens_max`` clamp the **whole clip**, because that is where
    the preprocessor applies them: measured on one server, three clips of 94.8 s, 47.4 s and
    39.1 s all landed within 2% of each other at ~12.3k tokens under the default budget, and
    the 94.8 s one rose to 32,853 tokens when the budget was raised.

    The budget is counted in pixels of the **raw** frame set, so it binds before frames are
    merged in time and the conversion carries ``temporal_merge``::

        ceiling = longest_edge_px / ((patch_px * spatial_merge) ** 2 * temporal_merge)

    Dropping that last factor is the easy mistake and it overstates the ceiling by exactly
    ``temporal_merge``: the default 25,165,824 px budget then predicts 24,576 tokens for a
    clip the server actually truncates at 12,288, so a capped run reads as an uncapped one
    and the cap is discovered after the cluster is sized. With the factor, the prediction is
    12,288 against measurements of 12,090, 12,333 and 12,345.
    """
    if temporal_merge < 1:
        raise ValueError("temporal_merge must be >= 1")
    frames = video_frames(
        duration_s,
        policy=frame_policy,
        sampling_fps=sampling_fps,
        frame_count=frame_count,
        max_frames=max_frames,
    )
    per_frame = image_tokens(
        width_px,
        height_px,
        policy=image_policy,
        patch_px=patch_px,
        spatial_merge=spatial_merge,
        fixed_tokens=fixed_tokens,
    )
    groups = math.ceil(frames / temporal_merge)
    tokens = groups * per_frame
    if longest_edge_px is not None:
        if patch_px is None:
            raise ValueError(
                "patch_px is required to apply longest_edge_px: the area budget is converted "
                "to a token ceiling as longest_edge_px / ((patch_px * spatial_merge) ** 2 "
                "* temporal_merge)"
            )
        ceiling = int(longest_edge_px / ((patch_px * spatial_merge) ** 2 * temporal_merge))
        tokens = min(tokens, ceiling)
    if tokens_max is not None:
        tokens = min(tokens, tokens_max)
    return tokens


def media_tokens_per_request(
    *,
    images: int = 0,
    tokens_per_image: float = 0.0,
    videos: int = 0,
    tokens_per_video: float = 0.0,
) -> float:
    """Total media tokens attached to one request.

    A trivial sum; it exists so the arithmetic has one name in the report and the reader can
    see the image and video contributions separately.
    """
    return images * tokens_per_image + videos * tokens_per_video


def vision_encoder_bytes(
    params: int,
    precision: str,
    *,
    tensor_parallel: int = 1,
    replicated_per_rank: bool = True,
) -> int:
    """Resident bytes of the vision encoder, per GPU.

    When ``replicated_per_rank`` is true the encoder is NOT sharded: every rank pays the
    full ``params * bytes`` and tensor parallelism buys nothing. When false, the cost
    divides by ``tensor_parallel``. Assuming the encoder shards with the language model is
    how a vision tower goes missing from a memory budget - and on a small model it is a
    double-digit percentage of the weight floor.
    """
    if tensor_parallel < 1:
        raise ValueError("tensor_parallel must be >= 1")
    total = params * dtype_bytes(precision)
    if replicated_per_rank:
        return int(total)
    return int(total / tensor_parallel)


@dataclass(frozen=True)
class CapCheck:
    """Outcome of a calibration check: whether a hidden cap (or a missing input) was found,
    plus a human-readable explanation that names the evidence. The explanation is not
    optional decoration - a bare boolean gets quoted without its caveat."""

    capped: bool
    explanation: str


def media_token_cap_check(
    samples: Sequence[tuple[float, float]],
    *,
    size_ratio: float = 2.0,
    token_tolerance: float = 0.10,
) -> CapCheck:
    """Detect a hidden media-token cap from ``(input_magnitude, measured_tokens)`` samples.

    ``samples`` are pixels for images, seconds for video. They are sorted by magnitude; if
    the largest and smallest magnitudes differ by at least ``size_ratio`` while their
    measured token counts differ by less than ``token_tolerance`` in relative terms, the
    preprocessor is capping and ``capped=True`` is returned with an explanation naming both
    points.

    Raises :class:`ValueError` on fewer than two samples, or when the magnitudes do not span
    ``size_ratio``: an inconclusive check must not be reported as a pass, because "we looked
    and found no cap" and "we could not look" size a cluster differently.
    """
    if len(samples) < 2:
        raise ValueError(
            "need at least two samples to compare; an inconclusive check must not be "
            "reported as a pass"
        )
    pts = sorted(samples)
    lo_mag, lo_tok = pts[0]
    hi_mag, hi_tok = pts[-1]
    if lo_mag <= 0:
        raise ValueError("sample magnitudes must be positive for a size ratio to be meaningful")
    if hi_mag < lo_mag * size_ratio:
        raise ValueError(
            f"samples span only {hi_mag / lo_mag:.2f}x (< required {size_ratio}x); the check "
            "is inconclusive and must not be reported as a pass"
        )
    denom = max(hi_tok, lo_tok)
    rel = abs(hi_tok - lo_tok) / denom if denom > 0 else 0.0
    if rel < token_tolerance:
        return CapCheck(
            capped=True,
            explanation=(
                f"token count is flat across a {hi_mag / lo_mag:.1f}x size range: "
                f"{lo_mag} -> {lo_tok} tokens vs {hi_mag} -> {hi_tok} tokens (relative "
                f"difference {rel:.1%} < tolerance {token_tolerance:.1%}); the preprocessor "
                "is capping media tokens"
            ),
        )
    return CapCheck(
        capped=False,
        explanation=(
            f"token count scales with input size: {lo_mag} -> {lo_tok} tokens vs {hi_mag} "
            f"-> {hi_tok} tokens (relative difference {rel:.1%} >= tolerance "
            f"{token_tolerance:.1%}); no cap detected"
        ),
    )


def media_arrival_check(
    measured_prompt_tokens: float,
    text_only_prompt_tokens: float,
    *,
    tolerance: float = 0.05,
) -> CapCheck:
    """Detect media that never reached the model.

    Returns ``capped=True`` - "the media did not arrive" - when ``measured_prompt_tokens``
    is within ``tolerance`` of the text-only prediction. On one real run the corpus was AV1,
    the container's decoder produced zero frames, and every request went through as text
    with no error at all: the run published a media capacity figure measured on no media.
    """
    if text_only_prompt_tokens <= 0:
        raise ValueError("text_only_prompt_tokens must be > 0 to serve as a baseline")
    rel = abs(measured_prompt_tokens - text_only_prompt_tokens) / text_only_prompt_tokens
    if rel <= tolerance:
        return CapCheck(
            capped=True,
            explanation=(
                f"the media did not arrive: measured prompt {measured_prompt_tokens} tokens "
                f"is within {tolerance:.1%} of the text-only prediction "
                f"{text_only_prompt_tokens}"
            ),
        )
    return CapCheck(
        capped=False,
        explanation=(
            f"media arrived: measured prompt {measured_prompt_tokens} tokens differs from "
            f"the text-only prediction {text_only_prompt_tokens} by {rel:.1%} "
            f"(tolerance {tolerance:.1%})"
        ),
    )


# ------------------------------------------------------------------- workload -> demand


@dataclass(frozen=True)
class Workload:
    """An application-level description of demand.

    This is the layer that turns a product question ("10,000 daily users") into a serving
    question ("N concurrent sessions, M tokens/s"). Every field is something a product owner
    can answer without knowing anything about GPUs.
    """

    daily_active_users: float | None = None
    sessions_per_user_per_day: float = 1.0
    avg_session_seconds: float = 0.0
    #: Ratio of peak-hour rate to daily-average rate. 1.0 is a flat load and is almost never
    #: real; 3-6 is typical for consumer apps in a single timezone.
    peak_to_mean: float = 4.0
    #: Fraction of a concurrent session that is actually generating tokens. A chat user
    #: reading a reply is idle: duty cycle is what separates "logged in" from "in flight".
    duty_cycle: float = 1.0
    #: Fraction of a concurrent session whose KV blocks stay resident -- NOT the same
    #: question as ``duty_cycle``. An agent paused on a tool call is not generating but IS
    #: still holding its context, so reusing the duty cycle here divides the pool by a
    #: fraction that describes decode, not memory. None means ``duty_cycle``, which keeps
    #: every v0.4.0 artifact recomputing unchanged and is why C10 warns on the default.
    kv_residency: float | None = None
    concurrent_users: float | None = None
    input_tokens_per_request: float = 0.0
    output_tokens_per_request: float = 0.0
    #: Visual tokens attached to the prompt, from :func:`media_tokens_per_request`. They sit
    #: in the prompt for the whole request, so unlike output they are NOT halved in
    #: :meth:`avg_context_tokens`.
    media_tokens_per_request: float = 0.0
    #: Reasoning ("thinking") tokens generated per request. They accumulate exactly like
    #: output tokens, so they are halved in :meth:`avg_context_tokens` and counted in
    #: :meth:`demand_tok_s`; omitting them understates the throughput floor by the
    #: reasoning-to-visible ratio, which can be two orders of magnitude.
    reasoning_tokens_per_request: float = 0.0
    requests_per_session: float = 1.0
    #: New context tokens each turn appends and every later turn must re-read. Tool traces
    #: accumulate across an agent session in a way chat replies do not. Default 0 reproduces
    #: the chat estimator bit-for-bit, so non-agent workloads need no change.
    context_growth_tokens_per_turn: float = 0.0
    #: Per-stream generation speed the product requires (e.g. faster than reading speed, or
    #: fast enough to feed a TTS pipeline without underrun).
    target_tok_s_per_user: float = 0.0

    def peak_concurrent_users(self) -> float:
        """Concurrent users at peak.

        Uses Little's law on the daily figures when ``concurrent_users`` is not given
        directly: ``L = lambda * W``, scaled by the peak-to-mean ratio.
        """
        if self.concurrent_users is not None:
            return self.concurrent_users
        if not self.daily_active_users or self.avg_session_seconds <= 0:
            raise ValueError(
                "provide concurrent_users, or daily_active_users with avg_session_seconds"
            )
        arrivals_per_s = (self.daily_active_users * self.sessions_per_user_per_day) / 86_400.0
        return arrivals_per_s * self.avg_session_seconds * self.peak_to_mean

    def active_sessions(self) -> float:
        """Sessions actually occupying a KV slot and generating at peak."""
        return self.peak_concurrent_users() * self.duty_cycle

    def avg_context_tokens(self) -> float:
        """Mean KV footprint per active session.

        Approximates a session mid-generation as its full input plus half its generated
        tokens. Two terms are new: ``media_tokens_per_request`` sits in the prompt for the
        whole request, so it is added in full and NOT halved;
        ``reasoning_tokens_per_request`` is generated and accumulates exactly like output,
        so it is halved together with ``output_tokens_per_request``. If you have real
        conversation-length data, override this - it is the highest-leverage single input
        in the whole protocol.

        ``context_growth_tokens_per_turn`` corrects that estimate for a tool-calling loop:
        every turn re-reads each earlier turn's context, so the mean over N turns is
        ``input + media + g*(N-1)/2 + (reasoning + output)/2``. When the growth is 0 the
        branch below is skipped and the chat expression is returned verbatim, so existing
        artifacts recompute bit-identically.
        """
        if self.context_growth_tokens_per_turn > 0 and self.requests_per_session > 1:
            # Pricing an accumulating transcript with the chat estimator drops the
            # g*(N-1)/2 re-read term, which on a many-turn code agent exceeds the whole
            # chat estimate -- the KV floor would pass a workload the pool cannot hold.
            return (
                self.input_tokens_per_request
                + self.media_tokens_per_request
                + self.context_growth_tokens_per_turn * (self.requests_per_session - 1.0) / 2.0
                + (self.reasoning_tokens_per_request + self.output_tokens_per_request) / 2.0
            )
        return (
            self.input_tokens_per_request
            + self.media_tokens_per_request
            + (self.reasoning_tokens_per_request + self.output_tokens_per_request) / 2.0
        )

    def demand_tok_s(self) -> float:
        """Aggregate output tokens/s required at peak.

        ``reasoning_tokens_per_request`` counts as generated output: thinking mode took one
        checkpoint from 120 to 33,829 completion tokens per request, so omitting reasoning
        tokens understates the throughput floor by exactly that factor.
        """
        if self.target_tok_s_per_user > 0:
            return self.active_sessions() * self.target_tok_s_per_user
        if self.avg_session_seconds <= 0:
            raise ValueError("need target_tok_s_per_user or avg_session_seconds")
        tokens_per_session = (
            self.output_tokens_per_request + self.reasoning_tokens_per_request
        ) * self.requests_per_session
        return self.peak_concurrent_users() * tokens_per_session / self.avg_session_seconds

    def demand_prefill_tok_s(self) -> float:
        """Aggregate input tokens/s required at peak -- the mirror of :meth:`demand_tok_s`.

        Prompt tokens cost GPU time to read, but ``demand_tok_s`` counts only generated
        tokens, so the throughput floor prices a workload as if its prompts were free. The
        two rates together are what make the benchmark's input:output mix comparable to the
        declared one; a workload more input-heavy than the run it was sized against is
        overstated by exactly the ratio of those two mixes.

        The ``target_tok_s_per_user`` branch converts through the declared per-request
        shape: pinning generation at ``r`` tokens/s implies ``r / generated`` requests/s,
        each carrying ``input + media`` prompt tokens. That ratio is declared, not assumed.
        """
        prompt_tokens = self.input_tokens_per_request + self.media_tokens_per_request
        if self.target_tok_s_per_user > 0:
            generated = self.output_tokens_per_request + self.reasoning_tokens_per_request
            if generated <= 0:
                # A per-stream generation rate cannot describe a workload that generates
                # nothing, and silently returning 0 here would delete the prefill floor for
                # exactly the embedding and reranking services it was added to price.
                raise ValueError(
                    "target_tok_s_per_user needs output_tokens_per_request or "
                    "reasoning_tokens_per_request to imply a request rate"
                )
            return self.active_sessions() * self.target_tok_s_per_user * prompt_tokens / generated
        if self.avg_session_seconds <= 0:
            raise ValueError("need target_tok_s_per_user or avg_session_seconds")
        tokens_per_session = prompt_tokens * self.requests_per_session
        return self.peak_concurrent_users() * tokens_per_session / self.avg_session_seconds


# ------------------------------------------------------------------------- capacity result


@dataclass
class Capacity:
    """The protocol's standard answer, with its binding constraint always attached."""

    tier: Tier
    max_concurrent_users: float
    max_tokens_per_s: float
    max_requests_per_s: float
    binding_constraint: Constraint
    n_gpus: int
    detail: dict = field(default_factory=dict)

    def daily_requests(self) -> float:
        """Requests per day *if peak load were sustained around the clock*.

        Deliberately not de-rated by the peak-to-mean ratio: this is a headroom figure, not a
        forecast. Divide by ``Workload.peak_to_mean`` for an expected-volume estimate.
        """
        return self.max_requests_per_s * 86_400.0

    def monthly_requests(self, days: float = 30.0) -> float:
        """The same headroom figure over a billing period, whose length you must choose.

        Not a reported field, and deliberately so: it is ``daily_requests`` times a constant, so
        giving it its own slot in the schema would create a second place for one number to be
        wrong, and C2 would have to tag an **(I)** derived from an **(I)**. Quote it in a
        commercial conversation if that is the unit the conversation uses, but state the day
        count with it — "30 days" and "a calendar month" differ by up to 3.3%, which is larger
        than the headroom margin some deployments are sized on.
        """
        return self.daily_requests() * days


def capacity_at(
    n_gpus: int,
    kv_tokens: float,
    throughput_tok_s: float,
    workload: Workload,
    tier: Tier = Tier.MEASURED,
    headroom: float = 1.0,
    slo_pass: bool = True,
    prefill_tok_s: float | None = None,
) -> Capacity:
    """Concurrent users supportable on ``n_gpus``, as ``min`` of the available floors.

    ``throughput_tok_s`` must be the aggregate tokens/s measured *at this context length and
    this GPU count* — throughput falls steeply with input length, so a single headline number
    from a short-prompt benchmark will overstate capacity by 2-4x at document lengths.

    ``headroom`` divides the usable throughput and KV (use >1.0 for the RECOMMENDED tier).

    ``prefill_tok_s`` is the input-token rate from the *same run and same window* as
    ``throughput_tok_s``. Passing it adds a fourth floor: the benchmark's input:output mix
    is rarely the declared workload's, and when the declared workload is the more
    input-heavy one, the output-only floor overstates capacity by the ratio of the two
    mixes. Left None, no fourth floor is added and this returns the v0.4.0 answer, which
    C11 then flags as publishing on an unpriced axis.

    ``Workload.kv_residency``, when set, replaces ``duty_cycle`` as the KV divisor. Sessions
    that hold KV while not generating still consume the pool, and pricing them as if they
    released it triples the claimed user count on a long-session agent workload.
    """
    if headroom < 1.0:
        raise ValueError("headroom must be >= 1.0")
    ctx = workload.avg_context_tokens()
    if ctx <= 0:
        raise ValueError("workload context length must be > 0")

    sessions_kv = (kv_tokens / headroom) / ctx

    if workload.kv_residency is not None and workload.kv_residency < workload.duty_cycle:
        # A residency below the generation fraction claims sessions hand back context they
        # are still generating from. No engine does that, so the number can only be a mixup,
        # and it would deflate the per-session footprint until the KV floor passes anything.
        raise ValueError("kv_residency must be >= duty_cycle")
    residency = workload.kv_residency if workload.kv_residency is not None else workload.duty_cycle
    users_kv = sessions_kv / residency if residency > 0 else math.inf

    # Demand per *concurrent user*, not per active session: a user who is idle between turns
    # still occupies a seat but generates nothing. Deriving this from Workload.demand_tok_s()
    # rather than from target_tok_s_per_user directly is what keeps the duty cycle applied —
    # using the raw per-stream target here silently overstates demand by 1/duty_cycle.
    base_users = workload.peak_concurrent_users()
    per_user = workload.demand_tok_s() / base_users if base_users > 0 else 0.0
    users_thr = (throughput_tok_s / headroom) / per_user if per_user > 0 else math.inf

    # Listed KV first so min()'s first-wins tie rule reproduces the v0.4.0 "KV wins ties"
    # pick and extends the same precedence to the new floor, rather than letting list order
    # silently re-rank equal candidates.
    floors: list[tuple[float, Constraint]] = [
        (users_kv, Constraint.KV),
        (users_thr, Constraint.THROUGHPUT),
    ]
    per_user_prefill = 0.0
    users_prefill = math.inf
    if prefill_tok_s is not None:
        per_user_prefill = workload.demand_prefill_tok_s() / base_users if base_users > 0 else 0.0
        users_prefill = (
            (prefill_tok_s / headroom) / per_user_prefill if per_user_prefill > 0 else math.inf
        )
        # A reranker declares output_tokens_per_request = 0, so per_user is 0 and users_thr
        # is infinite. Without this floor the min lands on KV and the report brands a service
        # holding no persistent KV as memory-bound. Pricing prefill makes the axis that
        # actually limits it bind instead.
        floors.append((users_prefill, Constraint.PREFILL))

    users, constraint = min(floors, key=lambda floor: floor[0])
    if not slo_pass:
        constraint = Constraint.SLO

    # Reasoning tokens belong in this denominator because they are already in the numerator:
    # per_user comes from Workload.demand_tok_s(), which counts them as generated output. A
    # visible-output-only divisor therefore prices each request at a fraction of the tokens it
    # actually cost, inflating requests/s -- and daily_requests() with it -- by exactly
    # (output + reasoning) / output. That is 282x on the thinking-mode checkpoint demand_tok_s()
    # already cites, in the direction that makes a cluster look like it can serve more.
    out_tokens = (workload.output_tokens_per_request + workload.reasoning_tokens_per_request) or 1.0
    if per_user > 0:
        requests_per_s = (users * per_user) / out_tokens
    elif per_user_prefill > 0:
        # Deriving requests/s from generated tokens reports 0 req/s for an embedding or
        # reranking service, which serves requests at full rate and simply returns no
        # tokens. When nothing is generated, the prompt side is the only honest denominator.
        prompt_tokens = workload.input_tokens_per_request + workload.media_tokens_per_request
        requests_per_s = (users * per_user_prefill) / prompt_tokens
    else:
        requests_per_s = 0.0

    detail = {
        "avg_context_tokens": ctx,
        "users_kv_floor": users_kv,
        "users_throughput_floor": users_thr,
        "per_user_tok_s": per_user,
        "headroom": headroom,
        "kv_tokens": kv_tokens,
        "throughput_tok_s": throughput_tok_s,
    }
    if prefill_tok_s is not None:
        # The keys stay absent when no rate was measured, so a v0.4.0 report's detail
        # serialises byte-identically. A present key holding a null would read as "we looked
        # and found nothing" rather than "we did not look".
        detail["users_prefill_floor"] = users_prefill
        detail["per_user_prefill_tok_s"] = per_user_prefill
        detail["prefill_tok_s"] = prefill_tok_s
    return Capacity(
        tier=tier,
        max_concurrent_users=users,
        max_tokens_per_s=min(throughput_tok_s / headroom, users * per_user),
        max_requests_per_s=requests_per_s,
        binding_constraint=constraint,
        n_gpus=n_gpus,
        detail=detail,
    )


def gpus_required(
    workload: Workload,
    kv_tokens_per_gpu: float,
    throughput_tok_s_per_gpu: float,
    headroom: float = 1.15,
    gpus_per_replica: int = 1,
    max_gpus: int = 4096,
) -> Capacity:
    """Smallest GPU count meeting ``workload``, rounded up to whole replicas.

    ``gpus_per_replica`` is the tensor-parallel width: capacity is bought in whole replicas,
    so a deployment needing 3 GPUs at TP=2 must provision 4. Both per-GPU inputs must come
    from a measurement **at this same TP width** — per-GPU KV is not a constant across
    topologies (see :func:`kv_heads_per_rank`).
    """
    if gpus_per_replica < 1:
        raise ValueError("gpus_per_replica must be >= 1")
    for replicas in range(1, max_gpus // gpus_per_replica + 1):
        n = replicas * gpus_per_replica
        cap = capacity_at(
            n_gpus=n,
            kv_tokens=kv_tokens_per_gpu * n,
            throughput_tok_s=throughput_tok_s_per_gpu * n,
            workload=workload,
            tier=Tier.RECOMMENDED,
            headroom=headroom,
        )
        if cap.max_concurrent_users >= workload.peak_concurrent_users():
            return cap
    raise ValueError(f"workload not satisfiable below {max_gpus} GPUs")


def scaling_efficiency(
    baseline_tok_s: float, scaled_tok_s: float, baseline_gpus: int, scaled_gpus: int
) -> float:
    """Fraction of linear scaling achieved, in ``[0, >1]``.

    Values above 1.0 are real but always mean the *baseline* was degraded — typically the
    narrow configuration was KV-starved — not that the wide one is superlinear. The protocol
    requires flagging any efficiency above 1.0 rather than reporting it as a win.
    """
    if baseline_gpus <= 0 or baseline_tok_s <= 0:
        raise ValueError("baseline must be positive")
    ideal = baseline_tok_s * (scaled_gpus / baseline_gpus)
    return scaled_tok_s / ideal


def interpolate_throughput(curve: Sequence[tuple[float, float]], context_tokens: float) -> float:
    """Piecewise-linear throughput at ``context_tokens`` from measured ``(ctx, tok_s)`` points.

    Clamps to the endpoints rather than extrapolating: throughput past the longest measured
    prompt is genuinely unknown, and linear extrapolation of a convex-decreasing curve can go
    negative. Callers projecting beyond the measured range must mark the result unmeasured.
    """
    pts = sorted(curve)
    if not pts:
        raise ValueError("curve must not be empty")
    if context_tokens <= pts[0][0]:
        return pts[0][1]
    if context_tokens >= pts[-1][0]:
        return pts[-1][1]
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x0 <= context_tokens <= x1:
            return y0 + (context_tokens - x0) / (x1 - x0) * (y1 - y0)
    raise AssertionError("unreachable")
