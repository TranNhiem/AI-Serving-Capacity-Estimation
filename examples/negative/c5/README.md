# A capacity figure that does not say which floor bound it

**Rule broken:** C5
**The one edit:** `capacity_tiers.measured.binding_constraint` is set to `null`, `capacity_tiers.measured.binding_constraint_u_reason` is added

The measured tier reports 16.384 concurrent users, but `binding_constraint` is null with an honest `(U)` reason, so nobody can say whether weights, KV, or throughput set that ceiling. Each floor has a different remedy: a weights-bound system needs a smaller model or more VRAM, a KV-bound one needs shorter contexts or KV quantisation, and a throughput-bound one needs more GPUs. Without the named constraint, an operator reading 16.384 knows the ceiling but not which purchase raises it, and can spend on the wrong one -- adding GPUs to a KV-bound system changes nothing. C5 exists precisely because a capacity figure without its floor is a plan with no next step, so this case grades non-conforming.

## Reproduce

```bash
ascep conformance examples/negative/c5/report.json
```

Every other byte of this report is identical to `examples/negative/baseline.json`, which grades
`conforming` with no findings. Diff the two to see the edit on its own:

```bash
diff <(jq -S . examples/negative/baseline.json) \
     <(jq -S . examples/negative/c5/report.json)
```
