"""Render validated ASCEP report dictionaries as standalone Markdown reports."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

_MISSING = "unmeasured; no reason recorded"
_TIERS = ("theoretical", "measured", "sustainable", "recommended")


def render(report: dict) -> str:
    """Render a conforming report dict as the standard Markdown capacity report."""
    lines: list[str] = []
    _front_matter(lines, report)
    _section_hardware(lines, _mapping(report.get("hardware")))
    _section_model(lines, _mapping(report.get("model")))
    _section_serving(lines, _mapping(report.get("serving")))
    _section_benchmark(lines, _mapping(report.get("run")))
    _section_capacity(lines, report)
    _section_workload(lines, report)
    _section_assumptions(lines, report.get("unmeasured_assumptions"))
    _section_reproduction(lines, report)
    return "\n".join(lines).rstrip() + "\n"


def render_file(path, out=None) -> str:
    """Read a report.json, render it, optionally write to `out`. Returns the Markdown."""
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    markdown = render(report)
    if out is not None:
        Path(out).write_text(markdown, encoding="utf-8")
    return markdown


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _clean(value: Any) -> str:
    text = str(value).replace("\n", " ").replace("|", "\\|")
    return " ".join(text.split())


def _reason(owner: Mapping[str, Any], field: str) -> str:
    raw = owner.get(f"{field}_u_reason")
    text = _clean(raw) if raw is not None else _MISSING
    if text.startswith("(U)"):
        text = text[3:].strip()
    return text


def _dash(owner: Mapping[str, Any], field: str) -> str:
    return f"— *(U) {_reason(owner, field)}*"


def _dash_text(text: str) -> str:
    reason = _clean(text) if text else _MISSING
    if reason.startswith("(U)"):
        reason = reason[3:].strip()
    return f"— *(U) {reason}*"


def _combined_missing(owner: Mapping[str, Any], fields: tuple[str, ...]) -> str:
    parts = [f"{field}: {_reason(owner, field)}" for field in fields]
    return f"— *(U) {'; '.join(parts)}*"


def _format_number(value: int | float) -> str:
    if isinstance(value, int):
        return f"{value:,}"
    if math.isfinite(value) and value.is_integer():
        return f"{int(value):,}"
    rendered = f"{value:,.6f}".rstrip("0").rstrip(".")
    return rendered if rendered else "0"


def _number(value: Any, tag: str, suffix: str = "") -> str:
    return f"{_format_number(value)} ({tag}){suffix}"


def _tag(owner: Mapping[str, Any], default: str) -> str:
    tag = owner.get("provenance", default)
    text = str(tag).upper()
    return text if text in {"M", "I", "T", "U"} else default


def _field(
    owner: Mapping[str, Any],
    field: str,
    default_tag: str = "M",
    suffix: str = "",
) -> str:
    value = owner.get(field)
    if value is None:
        return _dash(owner, field)
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return _number(value, _tag(owner, default_tag), suffix)
    if isinstance(value, (list, tuple)):
        return ", ".join(_number(item, default_tag) for item in value)
    return _clean(value)


def _text_field(owner: Mapping[str, Any], field: str) -> str:
    value = owner.get(field)
    return _dash(owner, field) if value is None else _clean(value)


def _code(value: Any) -> str:
    return f"`{_clean(value)}`"


def _bytes_field(owner: Mapping[str, Any], field: str, default_tag: str = "M") -> str:
    value = owner.get(field)
    if value is None:
        return _dash(owner, field)
    return _number(value / 1_073_741_824, _tag(owner, default_tag), " GiB")


def _large_number_field(
    owner: Mapping[str, Any],
    field: str,
    suffix: str,
    default_tag: str = "M",
) -> str:
    value = owner.get(field)
    if value is None:
        return _dash(owner, field)
    if value >= 1_000_000_000_000:
        return _number(value / 1_000_000_000_000, _tag(owner, default_tag), f" T{suffix}")
    if value >= 1_000_000_000:
        return _number(value / 1_000_000_000, _tag(owner, default_tag), f" G{suffix}")
    return _number(value, _tag(owner, default_tag), suffix)


def _table(lines: list[str], rows: list[tuple[str, str]]) -> None:
    lines.extend(["| field | value |", "|---|---|"])
    for label, value in rows:
        lines.append(f"| {label} | {value} |")


def _append_heading(lines: list[str], heading: str) -> None:
    lines.extend(["", heading, ""])


def _front_matter(lines: list[str], report: Mapping[str, Any]) -> None:
    model = _mapping(report.get("model")).get("model_id") or "unpublished model"
    hardware = _mapping(report.get("hardware")).get("gpu_model") or "unreported hardware"
    generated = _clean(report.get("report_generated_utc") or "not reported")
    conformance = _clean(report.get("conformance") or "not reported")

    lines.append(f"# ASCEP Capacity Report — `{_clean(model)}` on `{_clean(hardware)}`")
    lines.append("")
    lines.append(
        f"> ASCEP v0.1 · report generated `{generated}` · protocol conformance: **`{conformance}`**"
    )
    lines.append("")
    lines.append(_headline(report))
    lines.append("")
    # Required by the schema and by the template, because a level claimed without a stated
    # reason is unreviewable — the reader cannot tell a careful `partial` from a careless one.
    # Dropping it from the rendered output would hide exactly the caveat the level exists for.
    note = _clean(report.get("conformance_note") or "")
    lines.append(f"**Conformance note.** {note or _MISSING}")
    lines.append("")
    lines.extend(
        [
            "Every numeric claim below carries a provenance tag. Reports that omit tags are",
            "**non-conforming** and must not be compared against conforming reports.",
            "",
            "| tag | meaning |",
            "|---|---|",
            "| **(M)** | measured in this campaign, raw record available under `runs/` |",
            "| **(I)** | computed from an (M) value by a formula in `ascep.capacity`, "
            "formula named |",
            "| **(T)** | theoretical roofline — an upper bound, not an expectation |",
            "| **(U)** | unmeasured assumption; the report is only as good as this number |",
            "",
            "---",
        ]
    )


def _headline(report: Mapping[str, Any]) -> str:
    tiers = _mapping(report.get("capacity_tiers"))
    chosen: Mapping[str, Any] = {}
    tier_name = ""
    for candidate in ("recommended", "sustainable", "measured"):
        entry = _mapping(tiers.get(candidate))
        if entry.get("max_concurrent_users") is not None:
            chosen = entry
            tier_name = candidate
            break
    if not chosen:
        return "**Headline:** not reported"

    serving = _mapping(report.get("serving"))
    workload = _mapping(report.get("workload"))
    tag = _tag(chosen, "I")
    users = _number(chosen["max_concurrent_users"], tag)
    gpu_count = chosen.get("n_gpus", serving.get("gpu_count"))
    if gpu_count is None:
        gpu_count = _mapping(report.get("hardware")).get("gpu_count")
    gpus = _number(gpu_count, tag) if gpu_count is not None else "not reported"
    tp = _field(serving, "tensor_parallel")
    pp = _field(serving, "pipeline_parallel")
    constraint = _text_field(chosen, "binding_constraint")

    context = workload.get("avg_context_tokens")
    context_tag = "I"
    if context is None:
        input_tokens = workload.get("input_tokens_per_request")
        output_tokens = workload.get("output_tokens_per_request")
        if isinstance(input_tokens, (int, float)) and isinstance(output_tokens, (int, float)):
            context = input_tokens + output_tokens / 2.0
    context_text = (
        _number(context, context_tag) if isinstance(context, (int, float)) else "not reported"
    )

    return (
        f"**Headline:** {users} concurrent users on {gpus} GPUs "
        f"(TP={tp}, PP={pp}), bound by {constraint}, at {context_text} token average "
        f"context -- {tier_name} tier"
    )


def _section_hardware(lines: list[str], hardware: Mapping[str, Any]) -> None:
    _append_heading(lines, "## 1. Hardware")
    cpu_fields = ("cpu_model", "cpu_cores")
    if hardware.get("cpu_model") is None and hardware.get("cpu_cores") is None:
        cpu = _combined_missing(hardware, cpu_fields)
    else:
        cpu = f"{_text_field(hardware, 'cpu_model')} / {_field(hardware, 'cpu_cores')}"

    storage_fields = ("storage_class", "model_load_path")
    if all(hardware.get(field) is None for field in storage_fields):
        storage = _combined_missing(hardware, storage_fields)
    else:
        storage = (
            f"{_text_field(hardware, 'storage_class')} / {_text_field(hardware, 'model_load_path')}"
        )

    runtime_fields = ("driver_version", "compute_runtime_version")
    if all(hardware.get(field) is None for field in runtime_fields):
        runtime = _combined_missing(hardware, runtime_fields)
    else:
        runtime = (
            f"{_text_field(hardware, 'driver_version')} / "
            f"{_text_field(hardware, 'compute_runtime_version')}"
        )

    rows = [
        (
            "GPU model / count",
            f"{_text_field(hardware, 'gpu_model')} / {_field(hardware, 'gpu_count')}",
        ),
        ("VRAM per GPU", _bytes_field(hardware, "vram_bytes_per_gpu")),
        ("Interconnect (intra-node)", _text_field(hardware, "interconnect_intra_node")),
        ("Interconnect (inter-node)", _text_field(hardware, "interconnect_inter_node")),
        (
            "Nodes × GPUs per node",
            f"{_field(hardware, 'nodes')} × {_field(hardware, 'gpus_per_node')}",
        ),
        ("CPU model / cores", cpu),
        ("System RAM", _bytes_field(hardware, "system_ram_bytes")),
        ("Storage class + model load path", storage),
        (
            "HBM bandwidth per GPU (spec)",
            _large_number_field(hardware, "hbm_bandwidth_bytes_s", "B/s", "T"),
        ),
        (
            "Dense BF16 TFLOP/s per GPU (spec)",
            _large_number_field(hardware, "dense_bf16_flops_per_s", "FLOP/s", "T"),
        ),
        ("Driver / CUDA / ROCm version", runtime),
        ("Node exclusivity during benchmark", _text_field(hardware, "node_exclusivity")),
    ]
    _table(lines, rows)
    lines.extend(["", "> Shared nodes invalidate cross-report comparison. State it."])


def _section_model(lines: list[str], model: Mapping[str, Any]) -> None:
    _append_heading(lines, "## 2. Model")

    identity_fields = ("model_id", "revision")
    if all(model.get(field) is None for field in identity_fields):
        identity = _combined_missing(model, identity_fields)
    else:
        identity = f"{_text_field(model, 'model_id')} / {_text_field(model, 'revision')}"

    architecture = _text_field(model, "architecture")
    if architecture.lower() == "moe":
        experts = _field(model, "moe_experts")
        top_k = _field(model, "moe_top_k")
        architecture = f"MoE ({experts} experts, top_k {top_k})"

    geometry_fields = ("n_layers", "n_kv_heads", "head_dim")
    if all(model.get(field) is None for field in geometry_fields):
        geometry = _combined_missing(model, geometry_fields)
    else:
        geometry = (
            f"{_field(model, 'n_layers')} / {_field(model, 'n_kv_heads')} / "
            f"{_field(model, 'head_dim')}"
        )

    weight_bytes = model.get("weight_bytes_on_disk")
    if weight_bytes is None:
        weight_value = _dash(model, "weight_bytes_on_disk")
    else:
        weight_value = _number(weight_bytes / 1_000_000_000, _tag(model, "M"), " GB")

    rows = [
        ("Model ID + revision/commit", identity),
        ("Total parameters", _large_number_field(model, "total_params", " parameters")),
        (
            "Active parameters per token",
            _large_number_field(model, "active_params", " parameters per token"),
        ),
        ("Architecture", architecture),
        ("Weight precision", _text_field(model, "weight_precision")),
        ("KV precision", _text_field(model, "kv_precision")),
        ("Layers / KV heads / head dim", geometry),
        ("Attention type", _text_field(model, "attention_type")),
        ("Global-layer fraction", _field(model, "global_layer_frac")),
        ("Native max context", _field(model, "native_max_context_tokens", suffix=" tokens")),
        ("Weight bytes on disk", weight_value),
        ("Licence", _text_field(model, "licence")),
    ]
    _table(lines, rows)


def _section_serving(lines: list[str], serving: Mapping[str, Any]) -> None:
    _append_heading(lines, "## 3. Serving configuration")
    offload_fields = ("kv_cache_offload", "kv_cache_quantized")
    if all(serving.get(field) is None for field in offload_fields):
        kv_cache = _combined_missing(serving, offload_fields)
    else:
        kv_cache = (
            f"{_text_field(serving, 'kv_cache_offload')} / "
            f"{_text_field(serving, 'kv_cache_quantized')}"
        )

    rows = [
        (
            "Framework + exact version",
            f"{_text_field(serving, 'framework')} / {_text_field(serving, 'framework_version')}",
        ),
        ("Container image digest", _text_field(serving, "container_digest")),
        (
            "Tensor parallel / pipeline parallel",
            f"{_field(serving, 'tensor_parallel')} / {_field(serving, 'pipeline_parallel')}",
        ),
        ("`max_model_len`", _field(serving, "max_model_len")),
        ("`gpu_memory_utilization` (or equivalent)", _field(serving, "memory_utilization")),
        ("Batching mode", _text_field(serving, "batching_mode")),
        ("Max num seqs / max num batched tokens", "not reported"),
        ("Prefix caching", _text_field(serving, "prefix_caching")),
        ("KV cache offload / quantized KV", kv_cache),
        ("Chunked prefill", _text_field(serving, "chunked_prefill")),
        ("Speculative decoding", _text_field(serving, "speculative_decoding")),
        (
            "Engine-reported KV cache size",
            _field(serving, "engine_reported_kv_cache_tokens", suffix=" tokens"),
        ),
        ("Cold-start time to ready", _field(serving, "cold_start_to_ready_s", suffix=" s")),
    ]
    _table(lines, rows)


def _section_benchmark(lines: list[str], run: Mapping[str, Any]) -> None:
    _append_heading(lines, "## 4. Benchmark results")
    if not run:
        lines.append("not reported")
        return

    warmup = _field(run, "warmup_seconds", suffix=" s")
    repeats = _field(run, "repeats")
    window = _field(run, "sustained_window_seconds", suffix=" s")
    ladder = _field(run, "concurrency_ladder")
    outliers = _text_field(run, "outlier_method")
    lines.append(
        f"Per §7 of the protocol: warm-up discarded: {warmup}; repeats: {repeats}; "
        f"sustained window: {window}; concurrency ladder: {ladder}; outliers handled by "
        f"{outliers}."
    )
    lines.append("")

    results = run.get("results")
    if not isinstance(results, list) or not results:
        lines.append("not reported")
    else:
        _benchmark_table(lines, results)

    lines.append("")
    _slo_gates(lines, _mapping(run.get("slo_gates")))


def _benchmark_table(lines: list[str], results: list[Any]) -> None:
    lines.append(
        "| shape (in/out) | concurrency | TTFT p50 / p95 / p99 (s) | "
        "ITL p50 / p95 (s) | e2e p95 / p99 (s) | output tok/s | req/s | GPU util % | "
        "GPU mem util % | SLO |"
    )
    lines.append("|---|---:|---|---|---|---:|---:|---:|---:|:--:|")
    for item in results:
        row = _mapping(item)
        if row.get("input_tokens") is None and row.get("output_tokens") is None:
            shape_detail = _combined_missing(row, ("input_tokens", "output_tokens"))
        else:
            shape_detail = f"{_field(row, 'input_tokens')} / {_field(row, 'output_tokens')}"
        shape = f"{_field(row, 'context_tokens')} context; {shape_detail}"
        ttft = (
            f"{_field(row, 'ttft_p50_s')} / {_field(row, 'ttft_p95_s')} / "
            f"{_field(row, 'ttft_p99_s')}"
        )
        itl = f"{_field(row, 'itl_p50_s')} / {_field(row, 'itl_p95_s')}"
        e2e = f"{_field(row, 'e2e_p95_s')} / {_field(row, 'e2e_p99_s')}"
        slo = row.get("slo_pass")
        if slo is None:
            slo_cell = _dash(row, "slo_pass")
        else:
            slo_cell = "pass" if slo else "**fail**"
        cells = [
            shape,
            _field(row, "concurrency"),
            ttft,
            itl,
            e2e,
            _field(row, "output_tok_s"),
            _field(row, "requests_per_s"),
            _field(row, "gpu_util_pct"),
            _field(row, "gpu_mem_util_pct"),
            slo_cell,
        ]
        lines.append("| " + " | ".join(cells) + " |")


def _slo_gates(lines: list[str], gates: Mapping[str, Any]) -> None:
    if not gates:
        lines.append("**SLO gates applied:** not reported")
        return
    declared = _text_field(gates, "declared_before_run")
    lines.append(
        f"**SLO gates applied:** TTFT p95 ≤ {_field(gates, 'ttft_p95_max_s')} s · "
        f"ITL p95 ≤ {_field(gates, 'itl_p95_max_s')} s · "
        f"e2e p95 ≤ {_field(gates, 'e2e_p95_max_s')} s · "
        f"error rate ≤ {_field(gates, 'error_rate_max_pct')} % · "
        "all must hold for the full window."
    )
    lines.append(f"Declared before run: {declared}.")


def _section_scaling(lines: list[str], scaling: Any) -> None:
    lines.extend(["", "**Scaling efficiency**", ""])
    if not isinstance(scaling, list) or not scaling:
        lines.append("not reported")
        lines.append("")
        return
    lines.extend(
        [
            "| topology | GPUs | tok/s | efficiency vs baseline |",
            "|---|---:|---:|---:|",
        ]
    )
    notes: list[str] = []
    for item in scaling:
        row = _mapping(item)
        topology = f"TP={_field(row, 'tensor_parallel')}, PP={_field(row, 'pipeline_parallel')}"
        cells = [
            topology,
            _field(row, "gpu_count"),
            _field(row, "output_tok_s"),
            _field(row, "scaling_efficiency"),
        ]
        lines.append("| " + " | ".join(cells) + " |")
        context = _field(row, "context_tokens")
        notes.append(f"- {topology}: measured at {context} token context.")
        explanation = row.get("efficiency_over_one_explanation")
        if explanation:
            notes.append(f"> {_clean(explanation)}")
    if notes:
        lines.append("")
        lines.extend(notes)
    lines.append("")
    lines.append(
        "> Any efficiency > 1.0 must be explained, not celebrated — it means the baseline "
        "was degraded."
    )


def _section_roofline(lines: list[str], roofline: Mapping[str, Any]) -> None:
    lines.extend(["", "**Roofline comparison**", ""])
    if not roofline:
        lines.append("not reported")
        return
    lines.extend(
        [
            "| metric | theoretical (T) | measured (M) | efficiency |",
            "|---|---:|---:|---:|",
            (
                f"| decode tok/s | {_field(roofline, 'decode_tok_s_theoretical', 'T')} | "
                f"{_field(roofline, 'decode_tok_s_measured', 'M')} | "
                f"{_field(roofline, 'roofline_efficiency', 'I')} |"
            ),
            (
                f"| prefill TTFT | {_field(roofline, 'prefill_ttft_s_theoretical', 'T')} | "
                f"{_field(roofline, 'prefill_ttft_s_measured', 'M')} s | not reported |"
            ),
        ]
    )


def _section_capacity(lines: list[str], report: Mapping[str, Any]) -> None:
    _section_scaling(lines, report.get("scaling"))
    _section_roofline(lines, _mapping(report.get("roofline_comparison")))
    _append_heading(lines, "## 5. Capacity")

    serving = _mapping(report.get("serving"))
    lines.append(
        f"Topology for all rows: {_field(serving, 'gpu_count')} GPUs "
        f"(TP={_field(serving, 'tensor_parallel')}, "
        f"PP={_field(serving, 'pipeline_parallel')})."
    )
    lines.append("")
    lines.extend(
        [
            "| tier | max concurrent users | max tok/s | max req/s | daily requests | "
            "binding constraint |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    tiers = _mapping(report.get("capacity_tiers"))
    labels = {
        "theoretical": "Theoretical (T)",
        "measured": "Measured (M)",
        "sustainable": "Sustainable (M, SLO-gated)",
        "recommended": "**Recommended (I, headroom "
        + _field(_mapping(report.get("sizing_result")), "headroom_factor", "I")
        + ")**",
    }
    for tier in _TIERS:
        entry = _mapping(tiers.get(tier))
        tag = _tag(entry, "I")
        cells = [
            labels[tier],
            _field(entry, "max_concurrent_users", tag),
            _field(entry, "max_tokens_per_s", tag),
            _field(entry, "max_requests_per_s", tag),
            _field(entry, "daily_requests", tag),
            _text_field(entry, "binding_constraint"),
        ]
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    lines.append(
        "> A capacity figure without its binding constraint is not actionable — it does not "
        "say what to buy more of. Name it in every row."
    )
    if not _mapping(report.get("capacity_tiers")):
        lines.append("")
        lines.append("Capacity tiers beyond this table: not reported.")


def _section_workload(lines: list[str], report: Mapping[str, Any]) -> None:
    workload = _mapping(report.get("workload"))
    sizing = _mapping(report.get("sizing_result"))
    _append_heading(lines, "## 6. Application workload → required infrastructure")

    user_fields = ("daily_active_users", "concurrent_users")
    if all(workload.get(field) is None for field in user_fields):
        users = _combined_missing(workload, user_fields)
    else:
        users = f"{_field(workload, 'daily_active_users')} / {_field(workload, 'concurrent_users')}"

    session_fields = ("sessions_per_user_per_day", "avg_session_seconds")
    if all(workload.get(field) is None for field in session_fields):
        sessions = _combined_missing(workload, session_fields)
    else:
        sessions = (
            f"{_field(workload, 'sessions_per_user_per_day')} / "
            f"{_field(workload, 'avg_session_seconds', suffix=' s')}"
        )

    token_fields = ("input_tokens_per_request", "output_tokens_per_request")
    if all(workload.get(field) is None for field in token_fields):
        tokens = _combined_missing(workload, token_fields)
    else:
        tokens = (
            f"{_field(workload, 'input_tokens_per_request')} / "
            f"{_field(workload, 'output_tokens_per_request')}"
        )

    rows = [
        ("Application type", _text_field(workload, "application_type")),
        ("Daily active users / concurrent users", users),
        ("Sessions per user per day · avg session length", sessions),
        ("Peak-to-mean ratio", _field(workload, "peak_to_mean")),
        ("Duty cycle", _field(workload, "duty_cycle")),
        ("Input / output tokens per request", tokens),
        ("Avg KV context per active session", _field(workload, "avg_context_tokens", "I")),
        (
            "Required per-stream token rate",
            _field(workload, "target_tok_s_per_user", suffix=" tok/s"),
        ),
        ("Aggregate demand at peak", _field(workload, "demand_tok_s", "I", " tok/s")),
    ]
    _table(lines, rows)

    lines.extend(["", "**Result**", ""])
    if not sizing:
        lines.append("not reported")
        return
    topology = sizing.get("replica_topology")
    topology_cell = f"{_code(topology)} ({_tag(sizing, 'I')})" if topology else "not reported"
    result_rows = [
        ("GPUs required", _field(sizing, "gpus_required", "I")),
        ("Replica topology", topology_cell),
        ("Binding constraint", _text_field(sizing, "binding_constraint")),
        (
            "Utilization at target load",
            _field(sizing, "utilization_at_target_pct", "I", " %"),
        ),
        ("Headroom remaining", _field(sizing, "headroom_remaining", "I")),
    ]
    _table(lines, result_rows)


def _section_assumptions(lines: list[str], assumptions: Any) -> None:
    _append_heading(lines, "## 7. Unmeasured assumptions and known limits")
    lines.extend(
        [
            "List every **(U)** in the report and what it would take to close it. A report "
            "whose conclusion flips on an unmeasured number must say so here, in the same "
            "words a reviewer would use against it.",
            "",
        ]
    )
    if not isinstance(assumptions, list) or not assumptions:
        lines.append("not reported")
        return
    lines.extend(
        [
            "| field | impact if wrong | cost to measure |",
            "|---|---|---|",
        ]
    )
    for item in assumptions:
        entry = _mapping(item)
        cells = [
            _text_field(entry, "field"),
            _text_field(entry, "impact_if_wrong"),
            _text_field(entry, "cost_to_measure"),
        ]
        lines.append("| " + " | ".join(cells) + " |")


def _section_reproduction(lines: list[str], report: Mapping[str, Any]) -> None:
    reproduction = _mapping(report.get("reproduction"))
    run = _mapping(report.get("run"))
    _append_heading(lines, "## 8. Reproduction")
    lines.append("```")
    config_path = reproduction.get("run_configs_path")
    if config_path:
        lines.append("git clone <repository not reported> && cd <repository not reported>")
        lines.append(f"ascep run --config {_clean(config_path)}")
    else:
        lines.append("not reported")
    lines.extend(["```", ""])

    rows = [
        ("Run configs", _artefact(reproduction, run, "run_configs_path", None)),
        ("Raw per-request records", _artefact(reproduction, run, "raw_records_path", None)),
        ("Engine logs", _artefact(reproduction, run, "engine_logs_path", None)),
        (
            "Environment capture",
            _artefact(reproduction, run, "environment_capture_path", None),
        ),
        (
            "Analysis notebook / script",
            _artefact(reproduction, run, "analysis_script_path", None),
        ),
    ]
    lines.extend(["| artefact | path |", "|---|---|"])
    for label, value in rows:
        lines.append(f"| {label} | {value} |")


def _artefact(
    primary: Mapping[str, Any],
    secondary: Mapping[str, Any],
    field: str,
    fallback_field: str | None,
) -> str:
    for owner in (primary, secondary):
        if owner.get(field) is not None:
            return _code(owner[field])
    if primary:
        return _dash(primary, field)
    if secondary:
        return _dash(secondary, field)
    return _dash_text(_MISSING)
