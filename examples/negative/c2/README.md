# A number with no provenance

**Rule broken:** C2
**The one edit:** `model.weight_bytes_tag` is deleted

The model declares 26 GB of weights on disk, but the tag that says where that number came from is gone. If a reader assumes (M) and it was actually (I), the weights floor -- whether the model fits on the 4x A100s at all -- rests on a figure computed from the parameter count rather than read off the disk, and the two diverge by whatever the sharding and format overheads add. With no tag, there is no way to check the claim against the evidence without re-deriving it. This gap is not hypothetical. C2 was enforced for `provenance` fields but not for the three numbers whose provenance travels in a sibling `*_tag`, and the mutation search that built this corpus is what found the hole: the published moe-26b example had been shipping an untagged `avg_context_tokens` -- the number the KV floor divides by -- the whole time. The case grades non-conforming because an untested provenance claim on a binding number is exactly what C2 exists to prevent.

## Reproduce

```bash
ascep conformance examples/negative/c2/report.json
```

Every other byte of this report is identical to `examples/negative/baseline.json`, which grades
`conforming` with no findings. Diff the two to see the edit on its own:

```bash
diff <(jq -S . examples/negative/baseline.json) \
     <(jq -S . examples/negative/c2/report.json)
```
