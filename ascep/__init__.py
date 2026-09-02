"""ASCEP — AI Serving Capacity Estimation Protocol.

``ascep.capacity`` is stdlib-only by design: the analytic half of the protocol must import
and run on any machine, including air-gapped benchmark clusters with no package index. The
heavier modules — ``ascep.validation`` (which needs the optional ``jsonschema`` and
``referencing`` packages), ``ascep.conformance`` and ``ascep.render`` — are therefore not
re-exported here, so that a bare ``import ascep`` never requires third-party packages.
"""

from __future__ import annotations

from ascep.capacity import (
    DTYPE_BYTES,
    Capacity,
    Constraint,
    Tier,
    Workload,
    calibrate_memory_utilization,
    capacity_at,
    fits,
    gpus_required,
    interpolate_throughput,
    kv_bytes_per_token,
    kv_bytes_per_token_mla,
    kv_capacity_sessions,
    kv_capacity_tokens,
    kv_heads_per_rank,
    kv_pool_bytes,
    roofline_decode_tok_s,
    roofline_prefill_ttft_s,
    scaling_efficiency,
    weight_bytes,
)

__version__ = "0.4.0"
ASCEP_VERSION = "0.4.0"

__all__ = [
    "ASCEP_VERSION",
    "Capacity",
    "Constraint",
    "DTYPE_BYTES",
    "Tier",
    "Workload",
    "__version__",
    "calibrate_memory_utilization",
    "capacity_at",
    "fits",
    "gpus_required",
    "interpolate_throughput",
    "kv_bytes_per_token",
    "kv_bytes_per_token_mla",
    "kv_capacity_sessions",
    "kv_capacity_tokens",
    "kv_heads_per_rank",
    "kv_pool_bytes",
    "roofline_decode_tok_s",
    "roofline_prefill_ttft_s",
    "scaling_efficiency",
    "weight_bytes",
]
