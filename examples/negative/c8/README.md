# A report whose per-request records were never published

**Rule broken:** C8
**The one edit:** `reproduction.raw_records_path` is set to `null`, `reproduction.raw_records_path_u_reason` is added

The per-request records are the sole artifact from which anyone can recompute your p95s and error rates; with raw_records_path nulled, every latency figure in the report can only be taken on trust. An SLO-gated tier rests entirely on those percentiles, so someone reproducing the result has no way to check that the gate actually held. The honest _u_reason keeps this at partial rather than dropped keys territory, but it does not substitute for the artifact. The run configs, engine logs, environment capture, and container digest are all still declared, and that changes nothing -- one missing path breaks the bundle. The grade here is partial, publishable but not reproducible at the latency level.

## Reproduce

```bash
ascep conformance examples/negative/c8/report.json
```

Every other byte of this report is identical to `examples/negative/baseline.json`, which grades
`conforming` with no findings. Diff the two to see the edit on its own:

```bash
diff <(jq -S . examples/negative/baseline.json) \
     <(jq -S . examples/negative/c8/report.json)
```
