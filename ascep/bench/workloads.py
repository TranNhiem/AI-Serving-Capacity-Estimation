"""Workload declarations for ASCEP chapter 7 sections 2 and 3.

Every published number is a reduction over request records, and the two cheapest ways to
corrupt those records live here: running a workload whose token basis was never declared,
and letting a prefix cache answer traffic no production user generates. This module makes
both a declaration the run cannot omit, and makes the prompt sequence regenerable from the
manifest alone -- index by index, in a fresh process, without any state carried between
calls.
"""

from __future__ import annotations

import abc
import hashlib
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from ascep.bench.adapters.base import RequestSpec

#: The closed vocabulary of section 7.3. Anything outside it is not a weaker policy, it is
#: a policy the protocol cannot interpret, so it is refused rather than coerced.
CACHE_POLICIES = frozenset({"disabled", "cleared", "unique-prefix", "declared-workload", "unknown"})

#: The markers a multimodal corpus leaves in the text where the image or clip belongs. They
#: survive every type check because the value really is a string, and sending one to a text
#: endpoint turns a fifteen-hundred-token prompt into a five-token one -- so input_tokens,
#: TTFT and every prefill figure in the report describe a workload nobody ran.
MEDIA_PLACEHOLDER = re.compile(r"<\s*/?\s*(image|img|video|audio|vision)[^>]*>", re.IGNORECASE)


@dataclass(frozen=True)
class FixedOutput:
    """A fixed output budget. ignore_eos has no default: without it the declared length is
    a ceiling the model rarely reaches, and the run publishes a decode cost it never paid."""

    output_tokens: int
    ignore_eos: bool

    def __post_init__(self) -> None:
        if not isinstance(self.output_tokens, int) or self.output_tokens <= 0:
            raise ValueError(f"output_tokens must be a positive int, got {self.output_tokens!r}")


@dataclass(frozen=True)
class ModelDecidedOutput:
    """The output length the model chose. Not a sampled distribution: declaring it as one
    would claim a distribution that does not exist in the reproduction bundle."""


class PromptSource(abc.ABC):
    """Where prompt text comes from. Implementations own the section 7.2 rule that token
    counts are meaningful only after tokenization, by declaring whether a unique prefix can
    be absorbed inside the declared budget or has to lengthen the prompt."""

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Stable corpus identifier shipped in the run manifest."""

    @property
    @abc.abstractmethod
    def digest(self) -> str:
        """Content digest, so 'same corpus' is checkable rather than asserted."""

    @property
    @abc.abstractmethod
    def size(self) -> int | None:
        """Number of distinct prompts; None when the source generates an unbounded set."""

    @property
    @abc.abstractmethod
    def absorbs_prefix(self) -> bool:
        """True when a unique prefix is budgeted inside the declared token count."""

    @property
    @abc.abstractmethod
    def sampler_rule(self) -> str:
        """Human-readable draw rule for the manifest."""

    @property
    def field_path(self) -> str | None:
        """Where inside each corpus record the prompt was read from; None if not from a file.

        Concrete rather than abstract, here and below, so that a third party's source keeps
        working when the manifest grows a field. An abstract property would make every
        addition here a breaking change to their code, and the pressure would then be to
        stop adding to the manifest.
        """
        return None

    @property
    def media_placeholders_stripped(self) -> bool:
        """True when media markers were removed from the text the server was sent."""
        return False

    @abc.abstractmethod
    def render(self, *, seed_material: int, prefix: str | None) -> str:
        """Produce the prompt text for one request, deterministically from seed_material."""


@dataclass
class SyntheticCorpus(PromptSource):
    """Prompts built by appending filler words until the supplied tokenizer reports exactly
    input_tokens. The tokenizer is mandatory: a generator sized by words or characters
    produces a token count no two tokenizers agree on (section 7.2)."""

    input_tokens: int
    tokenizer: Callable[[str], int]

    def __post_init__(self) -> None:
        if not isinstance(self.input_tokens, int) or self.input_tokens <= 0:
            raise ValueError(
                f"input_tokens must be a positive int of tokens, got {self.input_tokens!r}"
            )

    @property
    def name(self) -> str:
        return f"synthetic:{self.input_tokens}"

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.name.encode("utf-8")).hexdigest()

    @property
    def size(self) -> int | None:
        return None

    @property
    def absorbs_prefix(self) -> bool:
        # The prefix is part of the prompt, so it is part of the budget; prepending it after
        # padding would overshoot every declared input_tokens in the report.
        return True

    @property
    def sampler_rule(self) -> str:
        return "deterministic-generated"

    def render(self, *, seed_material: int, prefix: str | None) -> str:
        # Grow geometrically, then trim one word at a time. Appending a word and
        # re-tokenizing the whole prompt after each one costs a tokenizer call per token,
        # so a 4k-token prompt took four thousand of them per request -- which makes the
        # load generator the slowest thing in the measurement it is supposed to be taking.
        rng = random.Random(seed_material)
        head = [prefix] if prefix else []
        filler: list[str] = []

        def count(words: list[str]) -> int:
            return self.tokenizer(" ".join(head + words))

        n = count(filler)
        step = 1
        while n < self.input_tokens:
            grown = n
            for _ in range(step):
                filler.append(f"w{rng.getrandbits(32):08x}")
            n = count(filler)
            if n == grown:
                # Appending words stopped moving the count, so the target is unreachable.
                break
            step = max(1, min(step * 2, self.input_tokens - n))

        # Overshoot is expected after the last doubling; walk back to the exact target.
        while n > self.input_tokens and filler:
            filler.pop()
            n = count(filler)

        if n != self.input_tokens:
            # A shortfall of even a few tokens would be invisible in the report while moving
            # prefill cost; refuse rather than approximate.
            raise ValueError(
                f"cannot hit exactly {self.input_tokens} tokens (prefix={prefix!r}, reached {n})"
            )
        return " ".join(head + filler)


def _resolve(record: object, steps: list[str], lineno: int, field: str) -> object:
    """Walk a dotted path into one record, naming the component that failed.

    A bare "field missing" over ``conversations.0.value`` sends the operator to grep for the
    whole path, which appears nowhere in the file. Naming the step says whether the corpus
    has no conversations, too few turns, or a turn shaped differently than expected.
    """
    node = record
    for depth, step in enumerate(steps):
        if isinstance(node, list):
            if not step.isdigit() or int(step) >= len(node):
                raise ValueError(
                    f"line {lineno}: {field!r} -- index {step!r} is out of range for the "
                    f"{len(node)}-element list at {'.'.join(steps[:depth]) or 'the record'}"
                )
            node = node[int(step)]
        elif isinstance(node, dict):
            if step not in node:
                raise ValueError(
                    f"line {lineno}: {field!r} -- no key {step!r} at "
                    f"{'.'.join(steps[:depth]) or 'the top level'}; "
                    f"present keys are {sorted(node)[:8]}"
                )
            node = node[step]
        else:
            raise ValueError(
                f"line {lineno}: {field!r} -- cannot look up {step!r} inside a "
                f"{type(node).__name__}"
            )
    return node


@dataclass
class JsonlCorpus(PromptSource):
    """A JSONL file of text prompts, fully read and validated at construction. A corpus
    this reader cannot represent must stop the run here, not degrade silently mid-ladder."""

    path: str | Path
    #: Dotted path to the prompt inside each record, list indices included, as in
    #: ``conversations.0.value``. Post-training corpora nest the prompt inside a conversation
    #: turn, and without a path the only way to use one is to pre-flatten it into a second
    #: file -- at which point the digest in the manifest pins the copy, not the dataset.
    field: str
    #: Removing the media markers is a legitimate experiment and a different workload from
    #: the corpus it came from, so it is opted into and lands in the manifest.
    strip_media_placeholders: bool = False

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        data = self.path.read_bytes()
        self._digest = hashlib.sha256(data).hexdigest()
        steps = self.field.split(".")
        prompts: list[str] = []
        for lineno, line in enumerate(data.decode("utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            value = _resolve(json.loads(line), steps, lineno, self.field)
            if not isinstance(value, str):
                # A multimodal value flattened to text would publish an input_tokens the
                # server never saw.
                raise ValueError(
                    f"line {lineno}: field {self.field!r} must be text, got {type(value).__name__}"
                )
            if MEDIA_PLACEHOLDER.search(value):
                if not self.strip_media_placeholders:
                    raise ValueError(
                        f"line {lineno}: field {self.field!r} still contains a media "
                        "placeholder such as <image>, so the corpus is multimodal and this "
                        "reader would send the text alone. Pass "
                        "strip_media_placeholders=True to declare the text-only variant as "
                        "the workload, or use a text corpus."
                    )
                value = MEDIA_PLACEHOLDER.sub(" ", value).strip()
            prompts.append(value)
        if not prompts:
            raise ValueError(f"corpus {self.path} is empty")
        self._prompts = prompts

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def digest(self) -> str:
        return self._digest

    @property
    def size(self) -> int:
        return len(self._prompts)

    @property
    def absorbs_prefix(self) -> bool:
        # File text is fixed, so the prefix is added on top and genuinely lengthens the
        # prompt; the manifest reports that via prefix_adds_tokens.
        return False

    @property
    def sampler_rule(self) -> str:
        return "uniform-with-replacement"

    @property
    def field_path(self) -> str:
        return self.field

    @property
    def media_placeholders_stripped(self) -> bool:
        return self.strip_media_placeholders

    def render(self, *, seed_material: int, prefix: str | None) -> str:
        rng = random.Random(seed_material)
        text = self._prompts[rng.randrange(len(self._prompts))]
        return f"{prefix} {text}" if prefix else text


@dataclass(frozen=True)
class Workload:
    """Everything section 7.2 and 7.3 require a run to declare before it may generate a
    request: token basis, cache policy, seed, and think time -- none of them defaulted,
    because a default is how a conditional permission becomes the silent norm."""

    source: PromptSource
    output_plan: FixedOutput | ModelDecidedOutput
    cache_policy: str
    seed: int
    think_time_s: float
    run_label: str
    temperature: float | None = None
    unknown_cache_reason: str | None = None
    think_time_distribution: object | None = None

    def __post_init__(self) -> None:
        if self.cache_policy not in CACHE_POLICIES:
            raise ValueError(
                f"unknown cache policy {self.cache_policy!r}; expected one of "
                f"{sorted(CACHE_POLICIES)}"
            )
        if self.cache_policy == "unknown" and not self.unknown_cache_reason:
            # "unknown" is the null of section 7.3; without a reason it is a shrug, and an
            # unjustified null is not publishable.
            raise ValueError("cache policy 'unknown' requires unknown_cache_reason")
        if self.think_time_s < 0:
            raise ValueError(f"think_time_s must be >= 0, got {self.think_time_s!r}")
        if self.think_time_distribution is not None:
            raise ValueError(
                "think time distributions are not implemented; only a constant is "
                "supported, and silently running one would mislabel the workload"
            )
        if not isinstance(self.output_plan, (FixedOutput, ModelDecidedOutput)):
            raise ValueError(f"unrecognised output plan {self.output_plan!r}")

    def _window_tag(self, repetition: int, concurrency: int | None) -> str:
        # A ladder window is identified by its rung as well as its repetition, but a
        # single-window run has no rung. Omitting the rung entirely in that case keeps the
        # draws a standalone caller already has, rather than silently reseeding them.
        return f"r{repetition}" if concurrency is None else f"c{concurrency}-r{repetition}"

    def _seed_material(self, repetition: int, index: int, concurrency: int | None = None) -> int:
        # Index-addressed determinism: the draw depends only on these values, so a fresh
        # process regenerates any request without replaying the ones before it.
        key = f"{self.seed}:{self._window_tag(repetition, concurrency)}:{index}".encode()
        return int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "big")

    def for_repetition(
        self, repetition: int, *, concurrency: int | None = None
    ) -> Callable[[int], RequestSpec]:
        """Bind the generator to one window; pass ``concurrency`` when it is a ladder rung.

        A ladder puts every window of every rung in one bundle, so the rung has to enter both
        the request id and the seed material. Without it, rung 8 replays rung 1's prompts:
        the ids collide, and worse, the server answers the higher rung out of a prefix cache
        the lower rung filled, which reads as a throughput ceiling that is further away than
        it is. Rungs run in ascending order, so the error always flatters the result.
        """
        tag = self._window_tag(repetition, concurrency)

        def next_spec(index: int) -> RequestSpec:
            material = self._seed_material(repetition, index, concurrency)
            prefix = None
            if self.cache_policy == "unique-prefix":
                # Distinct per (seed, window, index): a repeated prompt hands the server
                # cache hits production never has, and an independent repetition inheriting
                # repetition 0's prompts measures cache state, not variance.
                token = random.Random(material).getrandbits(48)
                prefix = f"upx-{self.seed}-{tag}-{index}-{token:012x}"
            text = self.source.render(seed_material=material, prefix=prefix)
            if isinstance(self.output_plan, FixedOutput):
                max_tokens: int | None = self.output_plan.output_tokens
                extra = {"ignore_eos": self.output_plan.ignore_eos}
            else:
                # None means server default; coercing it here would change the workload
                # invisibly, since the record shows only what the server was asked for.
                max_tokens = None
                extra = {}
            return RequestSpec(
                # Encodes run, window and index because records from every window of every
                # rung share one bundle; a collision would let dedupe drop a real request.
                request_id=f"{self.run_label}-{tag}-i{index}",
                messages=[{"role": "user", "content": text}],
                max_tokens=max_tokens,
                temperature=self.temperature,
                extra=extra,
            )

        return next_spec

    def manifest(self) -> dict:
        """The reproduction bundle's half of the promise: everything needed to regenerate
        the exact prompt sequence, as a plain JSON-serialisable dict."""
        fixed = isinstance(self.output_plan, FixedOutput)
        m = {
            "run_label": self.run_label,
            "seed": self.seed,
            "sampler": self.source.sampler_rule,
            "cache_policy": self.cache_policy,
            "think_time_s": self.think_time_s,
            "output_basis": "fixed" if fixed else "model-decided",
            "ignore_eos": self.output_plan.ignore_eos if fixed else False,
            "corpus_name": self.source.name,
            "corpus_digest": self.source.digest,
            "corpus_size": self.source.size,
            "corpus_field": self.source.field_path,
            # Not conditional on being true. A reader scanning for "was anything removed
            # from these prompts" must find the answer, not the absence of one.
            "media_placeholders_stripped": self.source.media_placeholders_stripped,
            "prefix_adds_tokens": (
                self.cache_policy == "unique-prefix" and not self.source.absorbs_prefix
            ),
            "temperature": self.temperature,
        }
        if fixed:
            m["output_tokens"] = self.output_plan.output_tokens
        if self.cache_policy == "unknown":
            m["cache_policy_u_reason"] = self.unknown_cache_reason
        return m
