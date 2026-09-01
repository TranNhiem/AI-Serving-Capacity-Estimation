#!/usr/bin/env python3
"""Generate `workload.json` for the 10,000-DAU chatbot from chapter 6's worked example (a).

The point of this file is that the derived fields are not typed in by hand. Every (I) figure
in the artifact — peak concurrency, active sessions, average context, aggregate demand — is
computed here by `ascep.capacity.Workload`, so the published JSON cannot drift away from the
formulas the protocol says produced it. CI regenerates it and diffs.

Run:  python examples/chatbot-10k-dau/build_workload.py
"""

from __future__ import annotations

import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))

from ascep import ASCEP_VERSION  # noqa: E402
from ascep.capacity import Workload  # noqa: E402
from ascep.validation import validate  # noqa: E402

# Imported rather than pinned here: a builder that hardcodes the version keeps stamping the
# release it was written under, so the artifact it regenerates claims a protocol the code no
# longer implements, and the version-consistency test does not look inside examples.

#: Chapter 6, worked example (a). These are product forecasts, not measurements: a workload
#: declaration is (U) by construction until product telemetry replaces it, which is exactly
#: why `daily_users_tag` exists and why the sensitivity note below is not optional.
WORK = Workload(
    daily_active_users=10_000,
    sessions_per_user_per_day=2,
    avg_session_seconds=600,
    peak_to_mean=4.0,
    duty_cycle=0.4,
    input_tokens_per_request=1_000,
    output_tokens_per_request=400,
    requests_per_session=5,
    target_tok_s_per_user=0.0,  # no per-stream anchor; demand comes from session volume
)


def build() -> dict:
    return {
        "ascep_version": ASCEP_VERSION,
        "application_type": (
            "conversational assistant (chat), short context, single timezone, "
            "human-in-the-loop turns"
        ),
        # --- product forecast (U) -------------------------------------------------------
        "daily_active_users": WORK.daily_active_users,
        "daily_users_tag": "U",
        "sessions_per_user_per_day": WORK.sessions_per_user_per_day,
        "avg_session_seconds": WORK.avg_session_seconds,
        "peak_to_mean": WORK.peak_to_mean,
        "duty_cycle": WORK.duty_cycle,
        "input_tokens_per_request": WORK.input_tokens_per_request,
        "output_tokens_per_request": WORK.output_tokens_per_request,
        "requests_per_session": WORK.requests_per_session,
        # Peak concurrency is derived here rather than observed, so the direct-override slot
        # is empty. C1: recorded as null with a reason, never omitted.
        "concurrent_users": None,
        "concurrent_users_u_reason": (
            "(U) no production traffic exists yet to observe peak concurrency directly, so "
            "the Little's-law derivation in peak_concurrent_users is used instead. Once the "
            "service is live this field SHOULD be replaced with the observed peak and tagged "
            "(M); it then overrides the derivation."
        ),
        "target_tok_s_per_user": None,
        "target_tok_s_per_user_u_reason": (
            "(U) the product has not fixed a per-stream rate anchor, so demand_tok_s is "
            "derived from session volume instead (output_tokens_per_request x "
            "requests_per_session / avg_session_seconds). Note what that figure is and is "
            "not: it is the average rate a concurrent user consumes across a whole session, "
            "including the pauses, and it is the correct input to the throughput floor. It "
            "is NOT a per-stream user-experience guarantee -- while a reply is actually "
            "streaming the user needs reading speed or better, and that is governed by the "
            "inter-token-latency SLO gate (chapter 4), not by this field."
        ),
        # --- derived by ascep.capacity.Workload (I) -------------------------------------
        "peak_concurrent_users": round(WORK.peak_concurrent_users(), 2),
        "active_sessions": round(WORK.active_sessions(), 2),
        "avg_context_tokens": round(WORK.avg_context_tokens(), 2),
        "avg_context_tokens_tag": "I",
        "demand_tok_s": round(WORK.demand_tok_s(), 2),
        # --- media and reasoning shape ---------------------------------------------------
        # 0 and null are different answers and the schema keeps them apart: 0 is measured and
        # genuinely none, null is not reported. This workload demonstrably sends no media, so
        # the counts are 0; there is no sampling rate to state at all, so those are null.
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
        # A capacity figure that does not name its reasoning mode is not comparable to one that
        # does: on the same weights and the same server, turning thinking on has been measured
        # to move completion tokens per request from 120 to 33,829.
        "reasoning_mode": "non-thinking",
        "reasoning_share": None,
        "reasoning_share_u_reason": ("(U) reasoning_share is meaningful only for a mixed workload"),
        "reasoning_tokens_per_request": 0,
        "max_output_tokens": None,
        "max_output_tokens_u_reason": (
            "(U) no output cap was declared for this run; output length was governed by the "
            "model's own stop condition"
        ),
    }


def main() -> int:
    doc = build()
    errors = validate("workload", doc)
    if errors:
        print("workload.json does not validate:", file=sys.stderr)
        for e in errors:
            print(f"  {e}", file=sys.stderr)
        return 1

    (HERE / "workload.json").write_text(json.dumps(doc, indent=2) + "\n")
    print(f"wrote {HERE / 'workload.json'}")
    print(
        f"  peak {doc['peak_concurrent_users']:.0f} concurrent"
        f" -> {doc['active_sessions']:.0f} active"
        f" @ {doc['avg_context_tokens']:.0f} tok context"
        f" = {doc['demand_tok_s']:.0f} tok/s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
