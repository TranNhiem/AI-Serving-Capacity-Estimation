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


def kv_bytes_per_token(
    n_layers: int,
    n_kv_heads: int,
    head_dim: int,
    kv_precision: str = "bf16",
    tensor_parallel: int = 1,
    global_layer_frac: float = 1.0,
) -> float:
    """Cluster-wide KV bytes for one token of context, for **standard MHA/GQA/MQA** attention.

    Counts K and V (hence the factor 2) and includes the TP replication penalty above.

    ``global_layer_frac`` is the fraction of layers holding full-length KV. Models with
    sliding-window or hybrid attention only keep full context on their *global* layers; the
    local layers are capped at the window and contribute far less. Set it to
    ``n_global_layers / n_layers`` for those, or leave 1.0 for uniform full attention.
    Report the value you used — it moves KV capacity by several times.

    **Do not use this for MLA** (DeepSeek-style multi-head latent attention), which caches a
    compressed latent instead of per-head K and V — this function overstates its KV by an
    order of magnitude. Use :func:`kv_bytes_per_token_mla`. For linear/SSM/hybrid-recurrent
    attention, per-token KV is not the right model at all: state is constant per sequence, so
    declare it via ``kv_bytes_per_sequence`` on :func:`kv_capacity_sessions` instead.
    """
    if not 0.0 < global_layer_frac <= 1.0:
        raise ValueError("global_layer_frac must be in (0, 1]")
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


def roofline_decode_tok_s(
    active_params: float,
    precision: str,
    hbm_bandwidth_bytes_s: float,
    batch_size: int = 1,
    avg_context_tokens: float = 0.0,
    kv_per_token: float = 0.0,
    efficiency: float = 1.0,
) -> float:
    """Upper bound on decode throughput, from memory bandwidth.

    Autoregressive decode is bandwidth-bound, not FLOP-bound: each step streams the active
    weights once and re-reads the KV cache for every sequence in the batch. Batching amortizes
    the weight read across ``batch_size`` tokens, which is why throughput rises with
    concurrency until KV traffic takes over.

    ``active_params`` is per-token active parameters — for MoE that is the shared trunk plus
    ``top_k`` experts, **not** total params. The gap between this bound and measurement is the
    protocol's ``roofline_efficiency`` and is a required reporting field: real servers land far
    below 1.0, and a value near or above 1.0 indicates the measurement is wrong.
    """
    if hbm_bandwidth_bytes_s <= 0:
        raise ValueError("hbm_bandwidth_bytes_s must be > 0")
    weight_read = active_params * dtype_bytes(precision)
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
        """
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
) -> Capacity:
    """Concurrent users supportable on ``n_gpus``, as ``min`` of the KV and throughput floors.

    ``throughput_tok_s`` must be the aggregate tokens/s measured *at this context length and
    this GPU count* — throughput falls steeply with input length, so a single headline number
    from a short-prompt benchmark will overstate capacity by 2-4x at document lengths.

    ``headroom`` divides the usable throughput and KV (use >1.0 for the RECOMMENDED tier).
    """
    if headroom < 1.0:
        raise ValueError("headroom must be >= 1.0")
    ctx = workload.avg_context_tokens()
    if ctx <= 0:
        raise ValueError("workload context length must be > 0")

    sessions_kv = (kv_tokens / headroom) / ctx
    users_kv = sessions_kv / workload.duty_cycle if workload.duty_cycle > 0 else math.inf

    # Demand per *concurrent user*, not per active session: a user who is idle between turns
    # still occupies a seat but generates nothing. Deriving this from Workload.demand_tok_s()
    # rather than from target_tok_s_per_user directly is what keeps the duty cycle applied —
    # using the raw per-stream target here silently overstates demand by 1/duty_cycle.
    base_users = workload.peak_concurrent_users()
    per_user = workload.demand_tok_s() / base_users if base_users > 0 else 0.0
    users_thr = (throughput_tok_s / headroom) / per_user if per_user > 0 else math.inf

    users = min(users_kv, users_thr)
    constraint = Constraint.KV if users_kv <= users_thr else Constraint.THROUGHPUT
    if not slo_pass:
        constraint = Constraint.SLO

    out_tokens = workload.output_tokens_per_request or 1.0
    return Capacity(
        tier=tier,
        max_concurrent_users=users,
        max_tokens_per_s=min(throughput_tok_s / headroom, users * per_user),
        max_requests_per_s=(users * per_user) / out_tokens,
        binding_constraint=constraint,
        n_gpus=n_gpus,
        detail={
            "avg_context_tokens": ctx,
            "users_kv_floor": users_kv,
            "users_throughput_floor": users_thr,
            "per_user_tok_s": per_user,
            "headroom": headroom,
            "kv_tokens": kv_tokens,
            "throughput_tok_s": throughput_tok_s,
        },
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
