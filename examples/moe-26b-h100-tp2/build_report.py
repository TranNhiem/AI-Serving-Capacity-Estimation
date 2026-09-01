#!/usr/bin/env python3
"""Assemble the schema-conforming report from this example's measured facts.

``run-summary.json`` holds what was actually observed, in a shape a human can read.
This script derives every ``(I)`` value from it using ``ascep.capacity`` only, and emits
``report.json`` validating against ``schemas/capacity-report.schema.json``.

Nothing here invents a number. Where a value cannot be derived from what was measured, it is
emitted as ``null`` alongside a ``*_u_reason`` saying why — which is what rule C1 requires and
what makes this report honestly *partial* rather than dishonestly complete.

    python examples/moe-26b-h100-tp2/build_report.py
"""

from __future__ import annotations

import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))

from ascep.capacity import Tier, Workload, capacity_at, scaling_efficiency  # noqa: E402
from ascep.validation import validate  # noqa: E402

VERSION = "0.1.0"
SRC = json.loads((HERE / "run-summary.json").read_text())

# The KV geometry was never captured, so kv_bytes_per_token cannot be computed. A
# weight-bandwidth-only roofline would omit the KV-read term, which dominates at this batch
# size, so it would be misleading rather than merely loose. We therefore decline to publish a
# theoretical tier at all. This is the honest outcome and the reason the report is partial.
NO_ROOFLINE = (
    "(U) model.n_layers / n_kv_heads / head_dim were not captured, so kv_bytes_per_token "
    "cannot be computed. A weight-bandwidth-only roofline would omit the KV-read term that "
    "dominates at this batch size, so publishing it would understate the bound's error rather "
    "than bound the measurement. Closing model KV geometry closes this too."
)


def _workload() -> Workload:
    w = SRC["workload"]
    return Workload(
        daily_active_users=0,
        concurrent_users=w["concurrent_users"],
        duty_cycle=w["duty_cycle"],
        input_tokens_per_request=w["input_tokens_per_request"],
        output_tokens_per_request=w["output_tokens_per_request"],
        target_tok_s_per_user=w["target_tok_s_per_user"],
    )


def _scaling_row(src: dict, efficiency: float, explanation: str | None = None) -> dict:
    row = {
        "tensor_parallel": src["tensor_parallel"],
        "pipeline_parallel": 1,
        "gpu_count": src["n_gpus"],
        "output_tok_s": src["tok_s"],
        "context_tokens": None,
        "context_tokens_u_reason": (
            "(U) the scaling comparison was logged without a context length, and it cannot be "
            "recovered: neither 287 nor 1117 tok/s matches any point on the TP=2 "
            "throughput-vs-context curve, so it was a different shape. This row is therefore "
            "C4-incomplete and is published only to demonstrate the above-1.0 check, never as "
            "a scaling result to plan against."
        ),
        "scaling_efficiency": round(efficiency, 2),
        "provenance": src["provenance"],
    }
    if explanation:
        row["efficiency_over_one_explanation"] = explanation
    return row


def _row(tier: Tier, cap, note: str | None = None) -> dict:
    """One capacity_tiers row. `cap` is a Capacity, or None for a tier we decline to state."""
    if cap is None:
        return {
            "tier": tier.value,
            "max_concurrent_users": None,
            "max_concurrent_users_u_reason": note,
            "max_tokens_per_s": None,
            "max_tokens_per_s_u_reason": note,
            "max_requests_per_s": None,
            "max_requests_per_s_u_reason": note,
            "daily_requests": None,
            "daily_requests_u_reason": note,
            "binding_constraint": None,
            "binding_constraint_u_reason": note,
            "n_gpus": SRC["serving"]["tensor_parallel"],
            # 'U', not null: the row is present and every cell in it is an unmeasured
            # assumption. That is exactly what the (U) tag means, and C2 wants a tag.
            "provenance": "U",
        }
    return {
        "tier": tier.value,
        "max_concurrent_users": round(cap.max_concurrent_users, 1),
        "max_tokens_per_s": round(cap.max_tokens_per_s, 1),
        "max_requests_per_s": round(cap.max_requests_per_s, 4),
        "daily_requests": round(cap.daily_requests()),
        "binding_constraint": cap.binding_constraint.value,
        "n_gpus": cap.n_gpus,
        "provenance": "I",
    }


NO_SPLIT = (
    "(U) the campaign logged context length only, not the input/output split. Recorded as "
    "null rather than back-filled: the split changes the prefill/decode mix and inventing it "
    "would make the point look more reproducible than it is."
)


def _results(m: dict, n_gpus: int) -> list[dict]:
    """One row per measured context length, throughput and latency joined on context.

    The two were reported as separate tables in the source campaign, which is how a curve and
    its SLO gates drift apart. Joining them here means every throughput figure carries the
    latency verdict measured at the same shape, or an explicit null saying it does not.
    """
    lat = {r["context_tokens"]: r for r in m["latency_gates"]["results"]}
    rows = []
    for p in m["throughput_vs_context_tp2"]["points"]:
        ctx = p["context_tokens"]
        g = lat.get(ctx)
        rows.append(
            {
                "context_tokens": ctx,
                "input_tokens": None,
                "input_tokens_u_reason": NO_SPLIT,
                "output_tokens": None,
                "output_tokens_u_reason": NO_SPLIT,
                "concurrency": None,
                "concurrency_u_reason": (
                    "(U) the point is described as 'at saturation'; the in-flight request count "
                    "was not recorded. Without it the point cannot be reproduced exactly, only "
                    "approached by ramping concurrency until throughput plateaus."
                ),
                # Aggregate, not per-GPU: capacity_at consumes cluster throughput. The source
                # table is per-GPU, so this multiplication is the (I) step (C3 -- the figure is
                # only meaningful bound to TP=2).
                "output_tok_s": p["tok_s_per_gpu"] * n_gpus,
                "requests_per_s": None,
                "requests_per_s_u_reason": (
                    "(U) not derivable: requests/s needs output tokens per request, which was "
                    "not recorded for this point."
                ),
                "ttft_p50_s": None,
                "ttft_p50_s_u_reason": "(U) only p95 was retained",
                "ttft_p95_s": g["ttft_p95_s"] if g else None,
                "ttft_p99_s": None,
                "ttft_p99_s_u_reason": "(U) only p95 was retained",
                "itl_p50_s": None,
                "itl_p50_s_u_reason": "(U) inter-token latency was not recorded",
                "itl_p95_s": None,
                "itl_p95_s_u_reason": "(U) inter-token latency was not recorded",
                # Null, and honestly so: with no ITL figure there is no population to name.
                # A campaign that recorded ITL but not its population would have to pick one
                # after the fact, which is the guess section 4.1 forbids.
                "itl_population": None,
                "itl_population_u_reason": (
                    "(U) no ITL percentile was recorded, so there is no population to declare"
                ),
                "e2e_p95_s": None,
                "e2e_p95_s_u_reason": "(U) end-to-end latency was not recorded",
                "e2e_p99_s": None,
                "e2e_p99_s_u_reason": "(U) end-to-end latency was not recorded",
                "gpu_util_pct": None,
                "gpu_util_pct_u_reason": "(U) device telemetry was not sampled during the run",
                "gpu_mem_util_pct": None,
                "gpu_mem_util_pct_u_reason": "(U) device telemetry was not sampled during the run",
                "error_rate_pct": None,
                "error_rate_pct_u_reason": (
                    "(U) not recorded. A run with an unrecorded error rate cannot distinguish "
                    "high throughput from high throughput of failures."
                ),
                "slo_pass": g["slo_pass"] if g else None,
                # Not back-filled from slo_pass. Chapter 7 §7 reserves `complete` for a rung
                # whose telemetry was evaluable and whose abort signals were demonstrably
                # observable before timing; neither was recorded here, so calling a passing
                # gate `complete` would be precisely the pass-by-omission the vocabulary
                # exists to stop. Retroactively, `complete` and `invalid` are indistinguishable.
                "outcome": None,
                "outcome_u_reason": (
                    "(U) the campaign recorded no rung outcome, and it cannot be reconstructed: "
                    "with no telemetry-health record there is no evidence separating a rung that "
                    "completed cleanly from one whose measurement was defective"
                ),
                "provenance": "M",
            }
        )
        if g is None:
            # No gate was run at this shape. Both the latency figure and the pass/fail verdict
            # are therefore absent for the same reason, stated once.
            why = "(U) the SLO gate was not exercised at this context length"
            rows[-1]["ttft_p95_s_u_reason"] = why
            rows[-1]["slo_pass_u_reason"] = why
    return rows


def build() -> dict:
    work = _workload()
    c = SRC["capacity_result"]
    m = SRC["measurements"]
    n_gpus = SRC["serving"]["tensor_parallel"]
    kv, thr = c["kv_tokens"], c["throughput_tok_s"]

    measured = capacity_at(n_gpus, kv, thr, work, tier=Tier.MEASURED, headroom=1.0)
    # Every declared SLO gate holds at this workload's 2,000-token average context: TTFT p95
    # was measured at 1.3311 s at 4k against a 2.0 s gate, and TTFT rises monotonically with
    # prompt length, so the shorter shape passes a fortiori. Sustainable therefore equals
    # measured here. At 8k the same gate fails, which is why 8k is excluded.
    sustainable = capacity_at(n_gpus, kv, thr, work, tier=Tier.SUSTAINABLE, headroom=1.0)
    recommended = capacity_at(n_gpus, kv, thr, work, tier=Tier.RECOMMENDED, headroom=c["headroom"])

    hw, mo, sv, wl = SRC["hardware"], SRC["model"], SRC["serving"], SRC["workload"]
    s = m["scaling"]
    # How far the binding floor sits below the slack one. Large means the non-binding floor is
    # irrelevant at this context; near 1.0 means the crossover is close and worth locating.
    floor_margin = measured.detail["users_kv_floor"] / measured.detail["users_throughput_floor"]

    return {
        "ascep_version": VERSION,
        # A literal, not `datetime.now()`: CI regenerates this file and diffs it against the
        # committed copy, so a wall-clock timestamp would fail the build every run. It is a
        # real date rather than the epoch because a "generated at 1970" reads as data, not as
        # the absence of it, and `ascep conformance` now rejects it as the placeholder it is.
        "report_generated_utc": SRC.get("report_generated_utc", "2026-08-31T00:00:00Z"),
        "conformance": SRC["conformance"],
        "conformance_note": SRC["conformance_note"],
        "hardware": {
            "ascep_version": VERSION,
            "gpu_model": hw["gpu_model"],
            "gpu_count": n_gpus,
            "nodes": hw["nodes"],
            "gpus_per_node": hw["gpu_count_per_node"],
            "vram_bytes_per_gpu": hw["vram_bytes_per_gpu"],
            "interconnect_intra_node": hw["interconnect_intra_node"],
            "interconnect_inter_node": None,
            "interconnect_inter_node_u_reason": hw["inter_node_note"],
            "node_exclusivity": hw["node_exclusivity"],
            "driver_version": None,
            "driver_version_u_reason": "(U) not captured at run time",
            "compute_runtime_version": None,
            "compute_runtime_version_u_reason": "(U) not captured at run time",
            "hbm_bandwidth_bytes_s": hw["hbm_bandwidth_bytes_s"],
            "dense_bf16_flops_per_s": hw["dense_bf16_flops_per_s"],
            "cpu_model": None,
            "cpu_model_u_reason": "(U) not captured at run time",
            "cpu_cores": None,
            "cpu_cores_u_reason": "(U) not captured at run time",
            "system_ram_bytes": None,
            "system_ram_bytes_u_reason": "(U) not captured at run time",
            "storage_class": None,
            "storage_class_u_reason": "(U) not captured at run time",
            "model_load_path": None,
            "model_load_path_u_reason": "(U) not captured at run time",
        },
        "model": {
            "ascep_version": VERSION,
            "model_id": None,
            "model_id_u_reason": mo["model_id_note"],
            "revision": None,
            "revision_u_reason": "(U) not pinned at run time",
            "total_params": mo["total_params"],
            "active_params": mo["active_params"],
            "architecture": mo["architecture"],
            "moe_experts": None,
            "moe_experts_u_reason": "(U) expert layout not captured",
            "moe_top_k": None,
            "moe_top_k_u_reason": "(U) expert layout not captured",
            "weight_precision": mo["weight_precision"],
            "kv_precision": mo["kv_precision"],
            "n_layers": None,
            "n_layers_u_reason": mo["kv_geometry_note"],
            "n_kv_heads": None,
            "n_kv_heads_u_reason": (
                "(U) not captured. This is the field that decides whether per-GPU KV capacity "
                "FALLS as TP grows: once tensor_parallel exceeds it, kv_heads_per_rank floors "
                "at 1 and effective cluster KV heads grow with TP. Without it the observed "
                "TP=1/TP=2 flatness must not be extended to TP=4."
            ),
            "head_dim": None,
            "head_dim_u_reason": (
                "(U) not captured. Third of the three fields kv_bytes_per_token needs; see "
                "n_layers."
            ),
            "attention_type": None,
            "attention_type_u_reason": (
                "(U) not captured. Without it the applicable KV formula cannot even be "
                "selected, let alone evaluated — see chapter 2, Attention families."
            ),
            "global_layer_frac": None,
            "global_layer_frac_u_reason": "(U) attention type unknown, so not applicable",
            "native_max_context_tokens": None,
            "native_max_context_tokens_u_reason": "(U) not captured; served length was 32768",
            "weight_bytes_on_disk": None,
            "weight_bytes_on_disk_u_reason": "(U) not recorded",
            "licence": None,
            "licence_u_reason": "(U) pre-release checkpoint; licence not public",
        },
        "serving": {
            "ascep_version": VERSION,
            "framework": sv["framework"],
            "framework_version": None,
            "framework_version_u_reason": sv["framework_version_note"],
            "container_digest": None,
            "container_digest_u_reason": "(U) not recorded at run time",
            "tensor_parallel": sv["tensor_parallel"],
            "pipeline_parallel": sv["pipeline_parallel"],
            "gpu_count": n_gpus,
            "max_model_len": sv["max_model_len"],
            "memory_utilization": None,
            "memory_utilization_u_reason": sv["memory_utilization_note"],
            "batching_mode": sv["batching_mode"],
            "prefix_caching": None,
            "prefix_caching_u_reason": (
                "(U) not recorded. The most dangerous gap in this report: if it was on and the "
                "benchmark reused prompt prefixes, measured throughput overstates production."
            ),
            "kv_cache_offload": None,
            "kv_cache_offload_u_reason": "(U) not recorded",
            "kv_cache_quantized": "off",
            "chunked_prefill": None,
            "chunked_prefill_u_reason": "(U) not recorded",
            "speculative_decoding": sv["speculative_decoding"],
            "engine_reported_kv_cache_tokens": kv,
            "cold_start_to_ready_s": None,
            "cold_start_to_ready_s_u_reason": "(U) not timed",
        },
        "run": {
            "ascep_version": VERSION,
            "engine_version": None,
            "engine_version_u_reason": sv["framework_version_note"],
            "container_digest": None,
            "container_digest_u_reason": "(U) not recorded at run time",
            "warmup_seconds": None,
            "warmup_seconds_u_reason": "(U) warm-up was performed but not recorded",
            "repeats": None,
            "repeats_u_reason": "(U) not recorded",
            "sustained_window_seconds": None,
            "sustained_window_seconds_u_reason": "(U) not recorded",
            "concurrency_ladder": None,
            "concurrency_ladder_u_reason": "(U) not recorded",
            "outlier_method": None,
            "outlier_method_u_reason": "(U) no outlier policy was declared before the run",
            # Chapter 4.3 and 4.7.1 postdate this campaign. Declaring them null with a reason
            # is what C1 asks for; back-filling "type-7" and the tokenizer we *probably* used
            # would be a guess dressed as a measurement, and the whole point of the (U) tag is
            # that a reader can tell those apart.
            "percentile_method": None,
            "percentile_method_u_reason": (
                "(U) the harness's percentile convention was not recorded and the raw records "
                "were not kept, so it cannot be recovered; the published p50/p95/p99 figures "
                "are therefore comparable within this report but not across sites"
            ),
            "tokenizer": None,
            "tokenizer_u_reason": (
                "(U) token counts come from the engine's usage accounting only; no independent "
                "local count was taken, so the 4.7.1 reconciliation check never ran"
            ),
            # The three chapter-7 "declare it before timing" rules. Nothing about them can be
            # recovered after the fact by definition -- a threshold chosen once the results are
            # in is not a predeclaration, so the only honest value is null.
            "drain_deadline_seconds": None,
            "drain_deadline_seconds_u_reason": (
                "(U) no drain deadline was declared, so it is unknown whether a request "
                "straddling the window close was counted as a late latency sample, as an error, "
                "or dropped from both"
            ),
            "throughput_collapse_ratio": None,
            "throughput_collapse_ratio_u_reason": (
                "(U) no collapse threshold was declared before the run; the campaign stopped "
                "climbing by operator judgement, which is the practice chapter 7 §7 replaces"
            ),
            "monotonic_across_ladder": None,
            "monotonic_across_ladder_u_reason": (
                "(U) no concurrency ladder was recorded, so monotonicity was never assessed; "
                "each point was measured 'at saturation' rather than swept"
            ),
            "open_loop": m["throughput_vs_context_tp2"]["mode"] == "open-loop",
            "ignore_eos": True,
            "slo_gates": {
                "ttft_p95_max_s": m["latency_gates"]["ttft_p95_s_max"],
                "itl_p95_max_s": None,
                "itl_p95_max_s_u_reason": "(U) no ITL gate was declared for this campaign",
                "e2e_p95_max_s": None,
                "e2e_p95_max_s_u_reason": "(U) no end-to-end gate was declared for this campaign",
                "error_rate_max_pct": None,
                "error_rate_max_pct_u_reason": "(U) no error-rate gate was declared",
                "declared_before_run": m["latency_gates"]["gates_declared_before_run"],
            },
            "environment_capture_path": None,
            "environment_capture_path_u_reason": "(U) predates the protocol; not published",
            "raw_records_path": None,
            "raw_records_path_u_reason": "(U) predates the protocol; not published",
            "engine_logs_path": None,
            "engine_logs_path_u_reason": "(U) predates the protocol; not published",
            "results": _results(m, n_gpus),
        },
        "workload": {
            "ascep_version": VERSION,
            "application_type": wl["application_type"],
            "daily_active_users": None,
            "daily_active_users_u_reason": (
                "(U) this campaign was specified directly in concurrent users, not DAU"
            ),
            "concurrent_users": wl["concurrent_users"],
            "sessions_per_user_per_day": None,
            "sessions_per_user_per_day_u_reason": "(U) not applicable; specified as CCU",
            "avg_session_seconds": None,
            "avg_session_seconds_u_reason": "(U) not applicable; specified as CCU",
            "peak_to_mean": None,
            "peak_to_mean_u_reason": "(U) not applicable; CCU is already the peak figure",
            "duty_cycle": wl["duty_cycle"],
            "input_tokens_per_request": wl["input_tokens_per_request"],
            "output_tokens_per_request": wl["output_tokens_per_request"],
            "requests_per_session": None,
            "requests_per_session_u_reason": "(U) not recorded",
            "target_tok_s_per_user": wl["target_tok_s_per_user"],
            # Peak concurrency was declared directly rather than derived from DAU, so
            # peak_concurrent_users passes concurrent_users through unchanged; active_sessions
            # is where the duty cycle actually lands.
            "peak_concurrent_users": work.peak_concurrent_users(),
            "active_sessions": work.active_sessions(),
            "avg_context_tokens": work.avg_context_tokens(),
            "demand_tok_s": work.demand_tok_s(),
        },
        "capacity_tiers": {
            "theoretical": _row(Tier.THEORETICAL, None, NO_ROOFLINE),
            "measured": _row(Tier.MEASURED, measured),
            "sustainable": _row(Tier.SUSTAINABLE, sustainable),
            "recommended": _row(Tier.RECOMMENDED, recommended),
        },
        "roofline_comparison": {
            "decode_tok_s_theoretical": None,
            "decode_tok_s_theoretical_u_reason": NO_ROOFLINE,
            "decode_tok_s_measured": thr,
            "roofline_efficiency": None,
            "roofline_efficiency_u_reason": NO_ROOFLINE,
            "prefill_ttft_s_theoretical": None,
            "prefill_ttft_s_theoretical_u_reason": (
                "(U) prefill is FLOP-bound and its roofline needs an assumed MFU. No MFU was "
                "measured for this campaign, and picking a conventional 0.4-0.5 would produce "
                "a bound whose error is the assumption rather than the hardware."
            ),
            "prefill_ttft_s_measured": m["latency_gates"]["results"][0]["ttft_p95_s"],
        },
        "scaling": [
            _scaling_row(s["baseline"], 1.0),
            _scaling_row(
                s["scaled"],
                scaling_efficiency(
                    s["baseline"]["tok_s"],
                    s["scaled"]["tok_s"],
                    s["baseline"]["n_gpus"],
                    s["scaled"]["n_gpus"],
                ),
                s["efficiency_note"],
            ),
        ],
        "sizing_result": {
            "gpus_required": n_gpus,
            "replica_topology": c["topology"],
            "binding_constraint": c["binding_constraint"],
            "headroom_factor": c["headroom"],
            "utilization_at_target_pct": round(100 * work.demand_tok_s() / thr, 1),
            "headroom_remaining": round(1 - work.demand_tok_s() / thr, 3),
            "floor_crossover_context_tokens": None,
            "floor_crossover_context_tokens_u_reason": (
                "(U) not determined. The floors cross where kv_tokens/ctx/duty_cycle equals "
                "the throughput floor; solving it needs the throughput-vs-context curve beyond "
                "8k, which was never measured. At 2,000 tokens the throughput floor binds by "
                f"{floor_margin:.1f}x."
            ),
            "provenance": "I",
        },
        "unmeasured_assumptions": [
            {
                "field": u["field"],
                "value_used": "null",
                "impact_if_wrong": u["impact_if_wrong"],
                "cost_to_measure": u["cost_to_measure"],
            }
            for u in SRC["unmeasured"]
        ],
        "reproduction": {
            "run_configs_path": None,
            "run_configs_path_u_reason": "(U) predates the protocol; not published",
            "raw_records_path": None,
            "raw_records_path_u_reason": "(U) predates the protocol; not published",
            "engine_logs_path": None,
            "engine_logs_path_u_reason": "(U) predates the protocol; not published",
            "environment_capture_path": None,
            "environment_capture_path_u_reason": "(U) predates the protocol; not published",
            "analysis_script_path": "examples/moe-26b-h100-tp2/build_report.py",
            "container_digest": None,
            "container_digest_u_reason": "(U) not recorded at run time",
        },
    }


def main() -> int:
    report = build()
    errors = validate("capacity-report", report)
    out = HERE / "report.json"
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out.relative_to(HERE.parent.parent)}")
    if errors:
        print(f"\n{len(errors)} schema error(s):")
        for e in errors:
            print("  ", e)
        return 1
    tiers = report["capacity_tiers"]
    print("\ntier            users   binding")
    for name in ("theoretical", "measured", "sustainable", "recommended"):
        r = tiers[name]
        u = r["max_concurrent_users"]
        users = "      —" if u is None else f"{u:>7.1f}"
        print(f"  {name:13} {users}   {r['binding_constraint'] or '(not stated)'}")
    print("\nvalid against capacity-report.schema.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
