"""One-shot v0.1 -> v0.2 migration backfilling multimodal and reasoning-mode declarations.

Every example document in the tree describes a text-only, non-thinking deployment, so the
new v0.2 fields can be filled in mechanically. The script is kept in the tree so the
change is auditable; it is not a tool anyone is expected to run twice. It is idempotent,
so an accidental second run is harmless and adds nothing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

MODEL_ADDITIONS: dict[str, object] = {
    "input_modalities": ["text"],
    "reasoning_modes": ["non-thinking"],
}

SERVING_ADDITIONS: dict[str, object] = {
    "image_input_transport": "n-a",
}

# 0 means measured and genuinely none; null means not reported. images_per_request and
# media_tokens_per_request are 0 because a text-only workload demonstrably sends no media,
# while video_sampling_fps is null because there is no sampling rate to state at all.
# Using one sentinel for both would make "not reported" indistinguishable from "measured
# zero" for every downstream consumer.
WORKLOAD_ADDITIONS: dict[str, object] = {
    "images_per_request": 0,
    "image_resolution_mix": None,
    "image_resolution_mix_u_reason": (
        "(U) no images are sent, so there is no resolution mix to declare"
    ),
    "videos_per_request": 0,
    "video_seconds_per_request": None,
    "video_seconds_per_request_u_reason": "(U) no video is sent",
    "video_sampling_fps": None,
    "video_sampling_fps_u_reason": "(U) no video is sent",
    "video_max_frames": None,
    "video_max_frames_u_reason": "(U) no video is sent",
    "media_tokens_per_request": 0,
    "reasoning_mode": "non-thinking",
    "reasoning_share": None,
    "reasoning_share_u_reason": "(U) reasoning_share is meaningful only for a mixed workload",
    "reasoning_tokens_per_request": 0,
    "max_output_tokens": None,
    "max_output_tokens_u_reason": (
        "(U) no output cap was declared for this run; output length was governed by the "
        "model's own stop condition"
    ),
}

LAYER_ADDITIONS = {
    "model": MODEL_ADDITIONS,
    "serving": SERVING_ADDITIONS,
    "workload": WORKLOAD_ADDITIONS,
}

STANDALONE_LAYERS = {
    "examples/bench-config/model.json": "model",
    "examples/bench-config/serving.json": "serving",
    "examples/bench-config/workload.json": "workload",
    "examples/chatbot-10k-dau/workload.json": "workload",
}

REPORT_PATHS = [
    "examples/moe-26b-h100-tp2/report.json",
    "examples/negative/baseline.json",
    *(f"examples/negative/c{i}/report.json" for i in range(1, 9)),
]

ALL_PATHS = [*STANDALONE_LAYERS, *REPORT_PATHS]

LAYER_NAMES = ("model", "serving", "workload")


class MigrationError(Exception):
    """A file could not be migrated; treating it as skippable would leave it invalid."""


def backfill(layer: dict[str, object], additions: dict[str, object]) -> int:
    # Overwriting an existing key would destroy a hand-edited value; only absent keys are added.
    added = 0
    for key, value in additions.items():
        if key not in layer:
            layer[key] = value
            added += 1
    return added


def process_file(rel_path: str, check: bool) -> str:
    path = REPO_ROOT / rel_path
    if not path.is_file():
        raise MigrationError(f"{rel_path}: file does not exist")
    raw = path.read_text(encoding="utf-8")
    trailing_newline = raw.endswith("\n")
    try:
        doc = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise MigrationError(f"{rel_path}: failed to parse JSON: {exc}") from exc
    if not isinstance(doc, dict):
        raise MigrationError(f"{rel_path}: top level is not an object")

    # Guessing the shape from the path alone can write a model layer's fields into a
    # report's root, which validates as nothing and is tedious to unpick.
    nested_model = doc.get("model")
    if isinstance(nested_model, dict) and "attention_type" in nested_model:
        shape = "B"
        missing = [name for name in LAYER_NAMES if not isinstance(doc.get(name), dict)]
        if missing:
            # Skipping a missing layer would leave the report invalid; the failure then
            # surfaces much later as a confusing schema error.
            raise MigrationError(f"{rel_path}: report is missing layer keys: {', '.join(missing)}")
        targets = {name: doc[name] for name in LAYER_NAMES}
    else:
        shape = "A"
        layer_name = STANDALONE_LAYERS.get(rel_path)
        if layer_name is None:
            raise MigrationError(f"{rel_path}: not a report and not in the standalone table")
        targets = {layer_name: doc}

    added = {name: backfill(targets[name], LAYER_ADDITIONS[name]) for name in targets}
    total = sum(added.values())

    if total and not check:
        with path.open("w", encoding="utf-8") as fh:
            json.dump(doc, fh, indent=2, ensure_ascii=False)
            # Adding or dropping the trailing newline shows up as noise in later diffs.
            if trailing_newline:
                fh.write("\n")

    verb = "would add" if check else "added"
    per_layer = " ".join(f"{name}={count}" for name, count in added.items())
    return f"{rel_path}: shape {shape}, {verb} {per_layer} ({total} keys total)"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Backfill v0.2 multimodal and reasoning-mode fields into example documents."
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="report what would change and write nothing",
    )
    args = parser.parse_args(argv)

    errors = []
    for rel_path in ALL_PATHS:
        try:
            print(process_file(rel_path, args.check))
        except MigrationError as exc:
            errors.append(str(exc))

    for error in errors:
        print(f"error: {error}", file=sys.stderr)
    if errors:
        # A zero exit here would let a silently skipped file surface much later as a
        # confusing schema error.
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
