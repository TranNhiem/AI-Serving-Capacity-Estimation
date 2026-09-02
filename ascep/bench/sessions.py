"""Replay captured agent-session shapes as benchmark traffic.

A coding-agent session is a closed loop: the model is called, it asks for a tool, the
client runs the tool while sending nothing, and then calls again with the conversation so
far plus the tool output. The prompt grows monotonically until a compaction truncates it.
An independent-request workload cannot measure that shape: it makes every request a cold
prefill, ignoring that turn N's prompt literally begins with turn N-1's, and it never
holds KV through a tool gap. This module REPLAYS the shape captured by
ascep/agent_profile.py -- prompt sizes come from the capture, not from any model output,
so nothing the server returns feeds back into the traffic. The wrong number it prevents
is a capacity figure measured on cold prefill alone, understated by the whole
prefix-cache hit rate and reported as though hardware were insufficient, plus its mirror
image: rung 8 replaying rung 1's session state and being answered out of a warm cache --
the flattering failure.
"""

from __future__ import annotations

import abc
import hashlib
import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

from ascep.bench.adapters.base import RequestSpec

#: The only shapes-file version this reader can interpret. Anything else is not a weaker
#: capture, it is a schema whose field meanings may differ, so it is refused at load
#: rather than coerced mid-ladder.
SHAPES_VERSION = 1

#: The keys every captured step must carry. All five are required: a step missing one is
#: a hand-assembled file guessing at the schema, and a guessed shape is a wrong workload.
_REQUIRED_STEP_KEYS = ("turn_index", "prompt_tokens", "output_tokens", "gap_s", "resets_prefix")


@dataclass(frozen=True)
class StepShape:
    """One API call inside a captured session: prompt size and output size in tokens, the
    client-side gap that followed it in seconds, and whether a compaction immediately
    preceded it."""

    turn_index: int
    prompt_tokens: int
    output_tokens: int
    gap_s: float
    resets_prefix: bool


@dataclass(frozen=True)
class SessionShape:
    """One captured session: its id and its steps in chronological order.

    Validated at construction for the failures a hand-assembled file actually produces --
    a decreasing turn_index, a negative gap, a zero-token prompt -- because a silent
    zero-token prompt renders as a rung of empty requests that looks like a very fast
    server.
    """

    session_id: str
    steps: tuple[StepShape, ...]

    def __post_init__(self) -> None:
        if not self.steps:
            raise ValueError(f"session {self.session_id}: no steps")
        previous = self.steps[0].turn_index
        for i, step in enumerate(self.steps):
            if step.turn_index < previous:
                raise ValueError(
                    f"session {self.session_id}: turn_index decreases at step {i} "
                    f"({previous} then {step.turn_index}); steps must be chronological"
                )
            previous = step.turn_index
            if step.gap_s < 0:
                raise ValueError(
                    f"session {self.session_id}: negative gap_s {step.gap_s} at step {i}; "
                    "a gap is client-side idle time in seconds and cannot run backwards"
                )
            if step.prompt_tokens < 1:
                raise ValueError(
                    f"session {self.session_id}: prompt_tokens {step.prompt_tokens} at step {i}; "
                    "a zero-token prompt renders as an empty request and reads as a very "
                    "fast server"
                )

    @property
    def turns(self) -> int:
        """Distinct application turns. A tool-calling turn issues several API calls that
        share one index, so this is not len(steps): reporting API calls as turns would
        overstate how much the agent got done per session."""
        return len({step.turn_index for step in self.steps})

    @property
    def requests(self) -> int:
        """The number of API calls the session makes -- simply len(steps)."""
        return len(self.steps)

    @property
    def wall_clock_s(self) -> float:
        """Client-side idle time in seconds: the sum of the tool gaps. Generation time is
        measured by the harness, not declared here."""
        return sum(step.gap_s for step in self.steps)


class SessionPlan(abc.ABC):
    """A source of session scripts for the closed-loop driver."""

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Stable plan identifier shipped in the run manifest."""

    @property
    @abc.abstractmethod
    def digest(self) -> str:
        """Content digest, so 'same traffic' is checkable rather than asserted."""

    @property
    @abc.abstractmethod
    def size(self) -> int:
        """Number of distinct captured shapes."""

    @property
    @abc.abstractmethod
    def sampler_rule(self) -> str:
        """Human-readable draw rule for the manifest."""

    @abc.abstractmethod
    def shape(self, session_index: int) -> SessionShape:
        """Which captured shape session number session_index replays."""

    @abc.abstractmethod
    def spec(self, *, session_index: int, step_index: int, request_id: str) -> RequestSpec:
        """The request spec for one step of one session, pure in its arguments."""

    def manifest(self) -> dict:
        """The reproduction bundle's statement of what traffic was replayed. A run whose
        manifest cannot say which shapes it used is not reproducible."""
        return {
            "kind": type(self).__name__,
            "name": self.name,
            "digest": self.digest,
            "size": self.size,
            "sampler_rule": self.sampler_rule,
            # Concrete on the base class for the same reason field_path is in workloads:
            # a third-party plan must keep working, and 0 is the conservative value.
            "shared_prefix_tokens": getattr(self, "shared_prefix_tokens", 0),
        }


@dataclass
class ReplaySessionPlan(SessionPlan):
    """Replays captured SessionShapes as prompt text whose prefix relationships match the
    capture: step k begins with step k-1's text unless the step is marked as a
    compaction, in which case it begins with the shared prefix and fresh filler.

    Every render is a pure function of (seed, session_index, step_index). One plan
    instance serves every rung of the concurrency ladder and every repetition, and it
    holds no cursor, counter or cache that could let a later rung inherit an earlier
    rung's state -- the flattering failure this whole module bends away from.
    """

    shapes: tuple[SessionShape, ...]
    seed: int
    tokenizer: Callable[[str], int]
    #: System prompt and tool schemas every session sends identically, in tokens. It is
    #: declared, not captured, because a transcript records only the total prompt size.
    shared_prefix_tokens: int = 0
    label: str = "replay"
    #: The most recently rendered step of each session, as {session_index: (step, text)}.
    #:
    #: A memo, not state: it is keyed by the arguments of a pure function and returns
    #: exactly what recomputing would, so it cannot let one rung inherit another's text --
    #: the invariant this module bends away from is about the *value*, and the value here
    #: is unchanged. What it buys is the difference between a harness that measures the
    #: server and one that measures itself. Without it, rendering step k rebuilds steps 0
    #: through k, so a 20-step session costs quadratic work in its own length, and the
    #: driver calls this inline on the event loop it is timing with: measured at 120,000
    #: prompt tokens it was half a second of blocking CPU per request, which every other
    #: virtual user waits out and every latency figure in the report absorbs.
    _recent: dict[int, tuple[int, str]] = field(default_factory=dict, repr=False, compare=False)
    #: The shared prefix, built on first use. Same reasoning as _recent: a memo over a pure
    #: function of the plan's own fields, not state that can vary a rendered prompt.
    _shared_text: str | None = field(default=None, repr=False, compare=False)

    #: How many sessions' texts to keep. Only the sessions currently in flight are ever
    #: read back -- a worker walks one session to its end and never returns to it -- so
    #: this needs to cover the widest rung, not the whole run. Bounded because the texts
    #: are the prompts themselves: at agent context lengths each one is megabytes.
    _RECENT_MAX = 256

    @property
    def name(self) -> str:
        return f"session-replay:{self.label}"

    @property
    def digest(self) -> str:
        """sha256 over the canonical JSON of the shapes, the shared prefix length and the
        seed. Two runs whose digests match replayed the same traffic; this is the value a
        reproduction bundle cites."""
        payload = {
            "seed": self.seed,
            "shared_prefix_tokens": self.shared_prefix_tokens,
            "shapes": [
                {
                    "session_id": shape.session_id,
                    "steps": [asdict(step) for step in shape.steps],
                }
                for shape in self.shapes
            ],
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(canonical).hexdigest()

    @property
    def size(self) -> int:
        """The number of captured shapes -- not the number of steps. Reporting steps here
        would understate cache reuse by the session length."""
        return len(self.shapes)

    @property
    def sampler_rule(self) -> str:
        """Round-robin, stated plainly. A random draw would give a 128-concurrency rung a
        different mix of session lengths than the same rung at 8, and the two rungs would
        stop being comparable; round-robin makes the mix a function of the request count
        alone."""
        return "round-robin by session index: shapes[session_index % len(shapes)]"

    def shape(self, session_index: int) -> SessionShape:
        return self.shapes[session_index % len(self.shapes)]

    def spec(self, *, session_index: int, step_index: int, request_id: str) -> RequestSpec:
        """Build the spec for one step.

        The prompt is one user message, not a reconstructed conversation array, on
        purpose: what the engine prices is prompt tokens and their prefix relationship,
        and one long message with the right prefix structure gives exactly that without
        pretending the replayed content is a real dialogue.

        max_tokens comes from the captured step -- a replay whose every turn generates
        the same number of tokens is not the workload that was captured. ignore_eos
        forces the captured length out of the model regardless of where it would stop, so
        the served text is meaningless. That is fine: the harness measures timing and
        token counts and discards text, and stopping early would make the output side of
        the shape whatever the served model happens to do, at which point the replay
        stops being a replay.
        """
        shape = self.shape(session_index)
        step = shape.steps[step_index]
        return RequestSpec(
            request_id=request_id,
            messages=[{"role": "user", "content": self._text(shape, session_index, step_index)}],
            max_tokens=step.output_tokens,
            extra={"ignore_eos": True},
        )

    def _seed_material(self, session_index: int, step_index: int) -> int:
        # Index-addressed determinism, the idiom workloads.py uses: the draw depends only
        # on these values, so a fresh process regenerates any step without replaying the
        # ones before it.
        key = f"{self.seed}:{session_index}:{step_index}".encode()
        return int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "big")

    def _shared_prefix_text(self) -> str:
        """The filler every session and every step begins with, identical across the whole
        plan. When shared_prefix_tokens is 0 there is no shared prefix and no
        shared-prefix cache hit -- the conservative default, because an invented one
        would hand every request a cache hit production may not have.

        Computed once. It is a constant of the plan, and every step of every session begins
        with it, so recomputing it per request put the cost of building a system-prompt-sized
        string inside the loop the driver times with.
        """
        if self.shared_prefix_tokens == 0:
            return ""
        if self._shared_text is None:
            key = f"{self.seed}:shared-prefix".encode()
            material = int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "big")
            self._shared_text = self._grow("", 0, self.shared_prefix_tokens, material)
        return self._shared_text

    def _text(self, shape: SessionShape, session_index: int, step_index: int) -> str:
        # Rebuild from the last compaction (or the session start) up to step_index, so
        # the returned text literally begins with the previous step's text -- the engine
        # then hits its prefix cache exactly as it would for a real agent.
        start = max(j for j in range(step_index + 1) if j == 0 or shape.steps[j].resets_prefix)
        # Resume from the memo when it holds a step of this same session inside the current
        # segment. Steps skipped this way were checked when they were built -- nothing is
        # cached that did not pass the shrink test below -- so resuming cannot smuggle an
        # unvalidated shape through.
        cached = self._recent.get(session_index)
        if cached is not None and start <= cached[0] < step_index:
            first, text = cached[0] + 1, cached[1]
            base_tokens = shape.steps[cached[0]].prompt_tokens
        else:
            first, text = start, self._shared_prefix_text()
            base_tokens = self.shared_prefix_tokens

        for j in range(first, step_index + 1):
            step = shape.steps[j]
            if j > start and step.prompt_tokens <= shape.steps[j - 1].prompt_tokens:
                # Appending cannot shrink. Clamping or truncating the prefix would
                # silently destroy the cache hit this module exists to reproduce.
                raise ValueError(
                    f"session {shape.session_id}, step {j}: prompt shrank from "
                    f"{shape.steps[j - 1].prompt_tokens} to {step.prompt_tokens} tokens "
                    "without resets_prefix; the capture recorded a compaction it did not "
                    "mark"
                )
            if j == start:
                # The first step of a segment starts from the shared prefix, not from the
                # step before it, which belongs to the conversation the compaction discarded.
                text, base_tokens = self._shared_prefix_text(), self.shared_prefix_tokens
            text = self._grow(
                text, base_tokens, step.prompt_tokens, self._seed_material(session_index, j)
            )
            base_tokens = step.prompt_tokens

        if len(self._recent) >= self._RECENT_MAX and session_index not in self._recent:
            # Oldest first: dicts keep insertion order, and a session already finished is
            # the one nobody will ask for again.
            del self._recent[next(iter(self._recent))]
        self._recent[session_index] = (step_index, text)
        return text

    def _grow(self, base: str, base_tokens: int, target: int, material: int) -> str:
        """Append deterministic filler to base until the tokenizer reports exactly target
        tokens. Refuses rather than approximating, as SyntheticCorpus does.

        The caller passes the base's exact token count, which it always knows -- it is the
        previous step's captured prompt_tokens, or the declared shared prefix -- so the
        first guess is the whole shortfall in one batch rather than a geometric ramp. For
        a one-word-per-token tokenizer that lands exactly and the target is confirmed in a
        single pass. A sub-word tokenizer lands near it and the loop corrects by the
        remaining difference, which is the same convergence the ramp had without paying to
        re-tokenize a megabyte of prompt seventeen times on the way up.
        """
        rng = random.Random(material)
        head = [base] if base else []
        filler: list[str] = []

        def count() -> int:
            return self.tokenizer(" ".join(head + filler))

        def add(k: int) -> None:
            for _ in range(k):
                filler.append(f"w{rng.getrandbits(32):08x}")

        add(max(0, target - base_tokens))
        n = count()
        # Bounded because a tokenizer can oscillate around a target it cannot hit -- some
        # targets are genuinely unreachable when one word is several tokens. Refusing after
        # a bounded search says so; looping would hang the run instead.
        for _ in range(64):
            if n == target:
                break
            if n < target:
                add(max(1, target - n))
            elif filler:
                del filler[max(0, len(filler) - (n - target)) :]
            else:
                break
            n = count()
        if n != target:
            # A shortfall of even a few tokens would be invisible in the report while
            # moving prefill cost; refuse rather than approximate.
            raise ValueError(f"cannot hit exactly {target} tokens (reached {n})")
        return " ".join(head + filler)


def load_shapes(path: str | Path) -> tuple[tuple[SessionShape, ...], int]:
    """Read a captured shapes file, returning (shapes, shared_prefix_tokens).

    Every schema failure is refused here, naming the path, because the alternative is
    discovering it eight hours into a ladder. shared_prefix_tokens defaults to 0: it is
    declared, not captured, since a transcript records the total prompt size and cannot
    decompose it.
    """
    p = Path(path)
    try:
        raw = p.read_text(encoding="utf-8")
        data = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{p}: unreadable shapes file ({exc})") from exc
    if not isinstance(data, dict) or data.get("ascep_shapes_version") != SHAPES_VERSION:
        raise ValueError(
            f"{p}: missing or unrecognised ascep_shapes_version "
            f"(expected {SHAPES_VERSION}); the schema version pins what the fields mean"
        )
    sessions = data.get("sessions")
    if not isinstance(sessions, list) or not sessions:
        raise ValueError(f"{p}: empty sessions list; a replay with no shapes is no traffic")
    shapes: list[SessionShape] = []
    for s, session in enumerate(sessions):
        if not isinstance(session, dict) or "session_id" not in session or "steps" not in session:
            raise ValueError(f"{p}: session {s} is missing session_id or steps")
        steps: list[StepShape] = []
        for k, step in enumerate(session["steps"]):
            missing = [key for key in _REQUIRED_STEP_KEYS if key not in step]
            if missing:
                raise ValueError(
                    f"{p}: step {k} of session {session['session_id']} is missing "
                    f"required key(s) {missing}; every field is required on every step"
                )
            steps.append(StepShape(**{key: step[key] for key in _REQUIRED_STEP_KEYS}))
        shapes.append(SessionShape(session_id=session["session_id"], steps=tuple(steps)))
    return tuple(shapes), data.get("shared_prefix_tokens", 0)
