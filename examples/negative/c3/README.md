# A topology that does not multiply out

**Rule broken:** C3
**The one edit:** `serving.gpu_count` is set to `8`

The declared topology is tensor_parallel 4 times pipeline_parallel 1, which is 4 GPUs, but the report says gpu_count 8. The two numbers no longer multiply out, so the capacity figures describe a machine that cannot be built. This is worse than a typo because throughput per GPU is the figure you divide by when you size a cluster, and the denominator here is wrong by a factor of two.

A reader who trusts gpu_count divides 245.76 tok/s by 8, concludes each GPU yields 30.7, and orders twice the hardware the run actually needed; a reader who trusts the parallelism fields divides by 4 and orders the right amount. Either way the procurement decision is made from a figure the run never produced. The grade is non-conforming because every capacity number in the report is bound to a topology that did not exist during the measurement.

## Reproduce

```bash
ascep conformance examples/negative/c3/report.json
```

Every other byte of this report is identical to `examples/negative/baseline.json`, which grades
`conforming` with no findings. Diff the two to see the edit on its own:

```bash
diff <(jq -S . examples/negative/baseline.json) \
     <(jq -S . examples/negative/c3/report.json)
```
