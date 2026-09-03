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
import base64
import hashlib
import json
import mimetypes
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from ascep.bench.adapters.base import RequestSpec

#: The closed vocabulary of section 7.3. Anything outside it is not a weaker policy, it is
#: a policy the protocol cannot interpret, so it is refused rather than coerced.
CACHE_POLICIES = frozenset({"disabled", "cleared", "unique-prefix", "declared-workload", "unknown"})

#: The filler vocabulary SyntheticCorpus pads with. Common English words, because a
#: subword tokenizer spends one token on each of them and eight on a random string: measured
#: on Gemma 4's tokenizer, 256 words of `w1a2b3c4d` filler came to 2,044 tokens and 256 words
#: drawn from this list came to exactly 256. A declared input_tokens of 1,500 therefore buys
#: about 1,500 tokens of prefill rather than about 12,000, and every context, KV and TTFT
#: figure derived from it describes the workload the config asked for.
#:
#: Short and closed on purpose. It is a padding vocabulary, not a corpus: the prompts are
#: filler either way, and a longer list would not make them mean anything. Two prompts must
#: share sixteen consecutive words before a vLLM prefix-cache block can hit, which at this
#: size is a probability with fifty zeros after the decimal point -- and under
#: 'unique-prefix' the unique head already rules it out.
_FILLER_WORDS = (
    "about above after again against all also although always among and another any are "
    "around because been before being below best better between both bring came can come "
    "could day did does done down during each early even ever every few find first found "
    "from give given going good great group had hand has have here high hold home house "
    "however important into just keep kind knew know large last later least left less life "
    "like little long look made make many may means might more most move much must name "
    "near need never new next night not now number often old once only open order other "
    "our over own part people perhaps place point possible present problem public put "
    "rather really right room said same saw say school second see seem seen set several "
    "shall she short should show side since small some something soon sound state still "
    "such take tell than that the their them then there these they thing think this those "
    "though thought three through time today together too took toward turn two under until "
    "upon use used using very want water way week well went were what when where whether "
    "which while who whole why will with within without word work world would year yet young"
).split()

#: The markers a multimodal corpus leaves in the text where the image or clip belongs. They
#: survive every type check because the value really is a string, and sending one to a text
#: endpoint turns a fifteen-hundred-token prompt into a five-token one -- so input_tokens,
#: TTFT and every prefill figure in the report describe a workload nobody ran.
MEDIA_PLACEHOLDER = re.compile(r"<\s*/?\s*(image|img|video|audio|vision)[^>]*>", re.IGNORECASE)

#: Per-modality marker patterns, so a multimodal reader can check that the text's markers
#: and the record's media references agree one modality at a time. A record with two
#: markers and one image sends a prompt the model cannot align.
_IMAGE_MARKER = re.compile(r"<\s*/?\s*(image|img)[^>]*>", re.IGNORECASE)
_VIDEO_MARKER = re.compile(r"<\s*/?\s*video[^>]*>", re.IGNORECASE)

#: How many resolutions media_shape lists before it stops and declares the remainder.
#: A natural image corpus measured here had 11,916 distinct resolutions across 13,644
#: images, so the untruncated list is longer than the rest of the manifest put together
#: and every row after the first few carries a share that rounds to zero.
_RESOLUTION_MIX_MAX = 32


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
class CappedOutput:
    """A ceiling on output length with EOS honoured: the production-realistic mode.

    Sits between FixedOutput, which suppresses EOS and makes every request pay exactly the
    declared decode cost, and ModelDecidedOutput, which sends no length at all. Here the
    model may stop early, so the number is not a decode commitment -- it is the ceiling
    that stops a single degenerate generation from monopolising a measurement window. The
    failure that ceiling exists to stop was measured, not hypothetical: on an H100 serving
    a 4B VLM against a real image corpus, a bench config declaring output_tokens 512 with
    ignore_eos false had the cap silently discarded, and one request that never emitted EOS
    generated alone for 90 consecutive seconds (engine log: one running request, zero
    prompt throughput the whole time, on the order of 14,000 tokens). The concurrency-1
    rung's three repetitions came out at 9 completions / 126.95 output tok/s, 12 / 113.23,
    and 2 / 13.87 -- a 9x collapse across repetitions caused by one runaway request, and it
    would have been published as throughput variance of the server.
    """

    output_tokens: int

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

    def render_content(self, *, seed_material: int, prefix: str | None) -> str | list[dict]:
        """Produce the message content for one request: plain text, or structured parts.

        Concrete rather than abstract for the same reason field_path is: a third party's
        text-only source must keep working now that the protocol can carry media, and for
        such a source the content of a request is exactly the text render, so the default
        is not a convenience but the correct answer.
        """
        return self.render(seed_material=seed_material, prefix=prefix)


@dataclass
class SyntheticCorpus(PromptSource):
    """Prompts built by appending filler words until the supplied tokenizer reports exactly
    input_tokens. The tokenizer is mandatory: a generator sized by words or characters
    produces a token count no two tokenizers agree on (section 7.2).

    The filler is drawn from _FILLER_WORDS rather than generated, because a caller that
    passes a word-count oracle -- as ``ascep bench`` does -- is relying on one word costing
    the served model about one token, and that is a property of the words, not of the
    counting."""

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
                filler.append(rng.choice(_FILLER_WORDS))
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


def _media_list(record: dict, key: str, lineno: int) -> list[str]:
    """Normalise a record's ``image`` or ``video`` entry to a list of relative paths.

    The corpus writes a lone string for the common one-image record and a list when there
    are several; anything else is a record this reader cannot represent.
    """
    value = record.get(key)
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list) and all(isinstance(v, str) for v in value):
        return list(value)
    raise ValueError(
        f"line {lineno}: {key!r} must be a relative path or a list of them, "
        f"got {type(value).__name__}"
    )


def _image_sizes(record: dict, n_images: int) -> list[tuple[int | None, int | None]]:
    """Pair each image in a record with the resolution the corpus declares for it.

    Single-image corpora write scalar ``width`` and ``height``; multi-image corpora write
    parallel lists, one entry per image. Reading only the scalar form throws the geometry
    of every multi-image record away, and media_shape then reports an empty resolution
    histogram for a corpus that stated every resolution -- which reads as "the corpus does
    not say" when the truth is "the reader did not look". A list whose length disagrees
    with the image count is left unsized rather than aligned by position, because guessing
    which image a stray entry belongs to would attribute a real resolution to the wrong
    image.
    """
    width = record.get("width")
    height = record.get("height")
    if _is_size(width) and _is_size(height):
        return [(width, height)] * n_images
    if (
        isinstance(width, list)
        and isinstance(height, list)
        and len(width) == len(height) == n_images
        and all(_is_size(w) for w in width)
        and all(_is_size(h) for h in height)
    ):
        return list(zip(width, height, strict=True))
    # Unsized media is a reporting gap, not a corpus error: media_shape declares the
    # share of images it could size, so the run says so rather than refusing to start.
    return [(None, None)] * n_images


def _is_size(value: object) -> bool:
    # bool is an int in Python, and a width of True would sail into the resolution
    # histogram as the pixel count 1.
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


@dataclass(frozen=True)
class _MediaRef:
    """One media reference with the geometry the corpus declares inline for it. The token
    cost of an image request is predictable from width and height without opening the
    file, so they are kept rather than re-derived."""

    kind: str
    rel_path: str
    width: int | None
    height: int | None


@dataclass(frozen=True)
class _MmRecord:
    """One parsed corpus line: the prompt text with its markers removed, the media those
    markers stood for, and whether the gpt turn marked the record as thinking-mode."""

    text: str
    media: tuple[_MediaRef, ...]
    has_reasoning: bool


@dataclass
class MultimodalJsonlCorpus(PromptSource):
    """A LLaVA/ShareGPT-shaped JSONL corpus whose media markers are honoured, not stripped.

    Each record names its media by relative path and carries the prompt in a conversation
    list, with an ``<image>`` or ``<video>`` marker where the content belongs. This reader
    removes the marker from the text and emits a real content part in its place: sending
    the marker itself is the five-token prompt MEDIA_PLACEHOLDER exists to refuse, and
    dropping the media is a text benchmark wearing a media benchmark's name.

    Under transport="base64" every referenced file is read and encoded exactly once, at
    construction, and render_content only looks the finished data URL up. Reading and
    encoding per request instead costs 1.93 ms of blocking work per request on the
    corpus this reader is run against, and the driver calls render_content from a
    single-threaded event loop between issuing requests. That is a client-side ceiling
    of roughly 518 requests per second that has nothing to do with the server, so a
    ladder climbing into it would report the load generator's limit as the model's
    throughput collapse -- and every blocking call also stalls the other in-flight
    requests, so the same 1.93 ms lands in the ITL and TTFT samples of requests that
    did no reading at all, arriving in the report as server latency.
    """

    path: Path
    #: The corpus references its media relative to this root; without one the paths in
    #: the file resolve against whatever directory the driver happened to start in. It is
    #: also a boundary, not just a prefix: a record whose path resolves outside it is
    #: refused, because an absolute path would discard the root and pin the corpus to the
    #: one machine whose filesystem it happens to describe.
    media_root: Path
    #: How the media reaches the server, mirroring the serving layer's
    #: image_input_transport: the bytes inline as a data URL, or a URL the server fetches.
    transport: str
    #: Base URL the server fetches from when transport is "url". Forbidden otherwise:
    #: with base64 the bytes travel in the request body and a prefix would be dead
    #: configuration that misdescribes the run.
    url_prefix: str | None = None
    #: Dotted path to the conversation list; the prompt is the first turn from "human".
    prompt_field: str = "conversations"
    #: Cap for a smoke run. Truncation changes the corpus, so it changes the digest and
    #: is named in the sampler rule. Under transport="base64" it also bounds memory:
    #: every record kept holds its media as a resident data URL, and media_shape
    #: reports the total as media_bytes_resident.
    max_records: int | None = None

    def __post_init__(self) -> None:
        if self.transport not in ("base64", "url"):
            raise ValueError(
                f"transport must be 'base64' or 'url', got {self.transport!r}; the serving "
                "layer declares image_input_transport from the same vocabulary and "
                "anything else is a transport the report cannot name"
            )
        if self.transport == "url" and not self.url_prefix:
            raise ValueError(
                "transport 'url' requires url_prefix: the server fetches the media from "
                "that base URL, and without it the corpus's relative paths resolve to nothing"
            )
        if self.transport != "url" and self.url_prefix is not None:
            raise ValueError(
                "url_prefix is only meaningful with transport 'url'; with 'base64' the "
                "bytes travel in the request body and the prefix would never be used"
            )
        if self.max_records is not None and (
            not isinstance(self.max_records, int) or self.max_records <= 0
        ):
            raise ValueError(
                f"max_records must be a positive int or None, got {self.max_records!r}"
            )
        self.path = Path(self.path)
        self.media_root = Path(self.media_root)
        # Resolved once, because the containment check in _parse_record runs per media file
        # and resolve() touches the filesystem.
        self._media_root_resolved = self.media_root.resolve()
        data = self.path.read_bytes()
        # The digest covers what the server receives, not only what is on disk: the same
        # file over a different transport, or truncated to a different max_records, is a
        # different workload.
        digest = hashlib.sha256()
        digest.update(data)
        digest.update(f"transport={self.transport};max_records={self.max_records}".encode())
        self._digest = digest.hexdigest()
        steps = self.prompt_field.split(".")
        # The base64 encode happens once, here at construction, so its cost lands before
        # the first request rather than inside the window being measured. A pre-encode
        # that ran lazily on first touch would just move the same stall to a random
        # point in the ladder. Keyed on the resolved path so two records naming the
        # same file are encoded once, not twice.
        self._data_urls: dict[Path, str] = {}
        self._media_bytes_resident = 0
        records: list[_MmRecord] = []
        truncated = False
        for lineno, line in enumerate(data.decode("utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            if self.max_records is not None and len(records) >= self.max_records:
                truncated = True
                break
            records.append(self._parse_record(json.loads(line), lineno, steps))
        if not records:
            raise ValueError(f"corpus {self.path} is empty")
        self._records = records
        self._truncated = truncated

    def _encode_once(self, resolved: Path, lineno: int) -> None:
        """Read and base64-encode one media file into the data-URL cache.

        The cache is keyed on the resolved path, so two records naming the same file
        pay the read and the encode once. Doing this per request instead put 1.93 ms
        of blocking file I/O and encoding on the event loop for every request
        generated: a client-side ceiling of roughly 518 requests per second that a
        ladder would report as the model's throughput collapse, and a stall that
        inflates the ITL and TTFT samples of every other in-flight request as if it
        were server latency.
        """
        if resolved in self._data_urls:
            return
        mime, _ = mimetypes.guess_type(resolved.name)
        if mime is None:
            # Refusing at load rather than mid-ladder is consistent with the
            # missing-file and marker-mismatch checks above: the operator's next move
            # is to look at that record, so the message names its line.
            raise ValueError(
                f"line {lineno}: cannot guess a MIME type for {resolved}; refusing to "
                "default to jpeg, because a wrong type is a request the server may "
                "decode differently than the corpus intended"
            )
        encoded = base64.b64encode(resolved.read_bytes()).decode("ascii")
        url = f"data:{mime};base64,{encoded}"
        self._data_urls[resolved] = url
        self._media_bytes_resident += len(url)

    def _parse_record(self, record: dict, lineno: int, steps: list[str]) -> _MmRecord:
        turns = _resolve(record, steps, lineno, self.prompt_field)
        if not isinstance(turns, list):
            raise ValueError(
                f"line {lineno}: field {self.prompt_field!r} must be a list of "
                f"conversation turns, got {type(turns).__name__}"
            )
        text: str | None = None
        has_reasoning = False
        for turn in turns:
            if not isinstance(turn, dict):
                raise ValueError(
                    f"line {lineno}: field {self.prompt_field!r} -- a conversation turn "
                    f"must be an object, got {type(turn).__name__}"
                )
            speaker = turn.get("from")
            value = turn.get("value")
            if speaker == "human" and text is None:
                if not isinstance(value, str):
                    raise ValueError(
                        f"line {lineno}: the human turn's value must be text, "
                        f"got {type(value).__name__}"
                    )
                text = value
            # A dict-shaped gpt value carrying "reasoning" is the corpus declaring a
            # thinking-mode record; counted so the run's reasoning_mode is declared
            # from evidence rather than guessed.
            if speaker == "gpt" and isinstance(value, dict) and "reasoning" in value:
                has_reasoning = True
        if text is None:
            raise ValueError(
                f"line {lineno}: no human turn in {self.prompt_field!r}; the prompt has "
                "nowhere to come from"
            )
        images = _media_list(record, "image", lineno)
        videos = _media_list(record, "video", lineno)
        image_markers = len(_IMAGE_MARKER.findall(text))
        video_markers = len(_VIDEO_MARKER.findall(text))
        if image_markers != len(images) or video_markers != len(videos):
            raise ValueError(
                f"line {lineno}: {image_markers} <image> and {video_markers} <video> "
                f"marker(s) but {len(images)} image and {len(videos)} video "
                "reference(s); a record whose markers and media disagree sends a prompt "
                "the model cannot align, and the resulting token count is meaningless"
            )
        sizes = _image_sizes(record, len(images))
        media: list[_MediaRef] = []
        for kind, rel_paths in (("image", images), ("video", videos)):
            for index, rel in enumerate(rel_paths):
                resolved = self.media_root / rel
                if not resolved.resolve().is_relative_to(self._media_root_resolved):
                    # An absolute path in the corpus makes media_root dead configuration:
                    # Path("/root") / "/elsewhere/x.png" is "/elsewhere/x.png", so the
                    # declared root is silently discarded and the run reads whatever the
                    # corpus names. Two things break. The bundle stops being portable --
                    # moving it to another machine changes nothing that can be pointed at
                    # the media, so the corpus digest matches while the files do not exist
                    # -- and the root stops bounding what the load generator will open and
                    # base64 into a request body. The same applies to a ".." that climbs
                    # out. Refusing here is what makes media_root mean something.
                    raise ValueError(
                        f"line {lineno}: media path {rel!r} resolves to {resolved.resolve()}, "
                        f"outside media_root {self._media_root_resolved}. Corpus media paths "
                        "must be relative to media_root; an absolute path ignores the root "
                        "entirely and the corpus can then only be run on the machine that "
                        "built it"
                    )
                if not resolved.is_file():
                    # Refusing at load is cheaper than discovering the substitution in a
                    # report: a skipped record is a media benchmark quietly becoming a
                    # text one, the failure media_arrival_check exists to catch after
                    # the fact.
                    raise ValueError(f"line {lineno}: media file {resolved} does not exist")
                if self.transport == "base64":
                    # Under "url" nothing is read: no file bytes, no encoding, no
                    # resident cost -- the server fetches from url_prefix instead.
                    self._encode_once(resolved, lineno)
                size = sizes[index] if kind == "image" else (None, None)
                media.append(
                    _MediaRef(kind=kind, rel_path=rel, width=size[0], height=size[1])
                )
        return _MmRecord(
            text=MEDIA_PLACEHOLDER.sub(" ", text).strip(),
            media=tuple(media),
            has_reasoning=has_reasoning,
        )

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def digest(self) -> str:
        return self._digest

    @property
    def size(self) -> int:
        return len(self._records)

    @property
    def absorbs_prefix(self) -> bool:
        # File text is fixed, so the prefix lengthens the text part of every request.
        return False

    @property
    def sampler_rule(self) -> str:
        if self._truncated:
            # A corpus silently reduced to its first N records is a different corpus;
            # the draw rule has to say so.
            return (
                f"uniform-with-replacement over the first {len(self._records)} "
                "records (truncated by max_records)"
            )
        return "uniform-with-replacement"

    @property
    def field_path(self) -> str:
        return self.prompt_field

    @property
    def media_placeholders_stripped(self) -> bool:
        # The markers were honoured -- replaced by real content parts -- not stripped.
        # Reporting True here would misdescribe the run as the text-only variant.
        return False

    def render(self, *, seed_material: int, prefix: str | None) -> str:
        raise ValueError(
            "MultimodalJsonlCorpus produces structured content parts, not plain text; "
            "returning the text alone would quietly turn a media benchmark into a text "
            "benchmark. Use render_content, or JsonlCorpus with "
            "strip_media_placeholders=True for the declared text-only variant."
        )

    def _media_part(self, ref: _MediaRef) -> dict:
        if self.transport == "url":
            url = f"{self.url_prefix.rstrip('/')}/{ref.rel_path}"
        else:
            # A pure lookup: the bytes were read and encoded once at construction, so
            # the request path touches no disk. Blocking here instead would cap the
            # load generator near 518 requests per second and bill the stall to the
            # server's ITL and TTFT samples.
            url = self._data_urls[self.media_root / ref.rel_path]
        key = "image_url" if ref.kind == "image" else "video_url"
        return {"type": key, key: {"url": url}}

    def render_content(self, *, seed_material: int, prefix: str | None) -> list[dict]:
        rng = random.Random(seed_material)
        record = self._records[rng.randrange(len(self._records))]
        text = f"{prefix} {record.text}" if prefix else record.text
        parts: list[dict] = [{"type": "text", "text": text}]
        parts.extend(self._media_part(ref) for ref in record.media)
        return parts

    def media_shape(self) -> dict:
        """The media-shape declaration chapter 9 section 9.8 requires alongside any media
        throughput figure, computed from the corpus rather than recollected.

        Video duration is deliberately absent: reading it needs a demuxer, which would
        break this module's stdlib-only promise, and chapter 9 section 9.3 already
        requires the effective sampling rate to be measured from the corpus and declared
        rather than derived here. A 0 below means the modality was measured and is
        absent; it is never None, because section 9.6 makes null mean "not reported".

        media_bytes_resident is the total size of the encoded data URLs held in memory,
        and it is the number an operator sizes max_records against. It is 0 under url
        transport -- measured and genuinely none, never None. There is no hidden
        ceiling on it: asking for the whole corpus holds the whole corpus in RAM, and
        an unbounded corpus here is a declaration, not an accident. At this corpus's
        mean media size, 21,494 records is about 10 GB of resident base64.
        """
        n = len(self._records)
        total_images = sum(1 for r in self._records for m in r.media if m.kind == "image")
        total_videos = sum(1 for r in self._records for m in r.media if m.kind == "video")
        resolutions: dict[tuple[int, int], int] = {}
        for r in self._records:
            for m in r.media:
                if m.kind == "image" and m.width is not None and m.height is not None:
                    key = (m.width, m.height)
                    resolutions[key] = resolutions.get(key, 0) + 1
        total_sized = sum(resolutions.values())
        ranked = sorted(resolutions.items(), key=lambda kv: (-kv[1], kv[0]))
        mix = [
            {"width": w, "height": h, "share": round(count / total_sized, 4)}
            for (w, h), count in ranked[:_RESOLUTION_MIX_MAX]
        ]
        listed = sum(count for _, count in ranked[:_RESOLUTION_MIX_MAX])
        return {
            "images_per_request": total_images / n,
            "videos_per_request": total_videos / n,
            "image_resolution_mix": mix,
            # A curated corpus has a handful of resolutions and this is all of them. A
            # corpus of web images has thousands, and emitting one row each turns the
            # workload manifest into a listing nobody reads and no operator can
            # transcribe into a declaration. The list is the most common
            # _RESOLUTION_MIX_MAX; the two fields below are what stop that truncation
            # from reading as the whole distribution.
            "image_resolution_mix_distinct": len(resolutions),
            "image_resolution_mix_listed_share": (
                round(listed / total_sized, 4) if total_sized else 0.0
            ),
            # The shares above are computed over the images the corpus sized, so on a
            # corpus that sizes only some of them the mix describes a subset while
            # reading as if it described the whole. This is that subset's size: 1.0 when
            # every image is sized, 0.0 when none is and the mix is empty.
            "image_resolution_mix_coverage": (
                round(total_sized / total_images, 4) if total_images else 0.0
            ),
            "records": n,
            "records_with_reasoning": sum(1 for r in self._records if r.has_reasoning),
            "media_bytes_resident": self._media_bytes_resident,
        }


@dataclass(frozen=True)
class Workload:
    """Everything section 7.2 and 7.3 require a run to declare before it may generate a
    request: token basis, cache policy, seed, and think time -- none of them defaulted,
    because a default is how a conditional permission becomes the silent norm."""

    source: PromptSource
    output_plan: FixedOutput | CappedOutput | ModelDecidedOutput
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
        if not isinstance(self.output_plan, (FixedOutput, CappedOutput, ModelDecidedOutput)):
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
            content = self.source.render_content(seed_material=material, prefix=prefix)
            if isinstance(self.output_plan, FixedOutput):
                max_tokens: int | None = self.output_plan.output_tokens
                extra = {"ignore_eos": self.output_plan.ignore_eos}
            elif isinstance(self.output_plan, CappedOutput):
                # The ceiling goes on the wire by itself. An explicit ignore_eos: false
                # would be the same request at the server, but the record would then claim
                # a key nobody asked for.
                max_tokens = self.output_plan.output_tokens
                extra = {}
            else:
                # None means server default; coercing it here would change the workload
                # invisibly, since the record shows only what the server was asked for.
                max_tokens = None
                extra = {}
            return RequestSpec(
                # Encodes run, window and index because records from every window of every
                # rung share one bundle; a collision would let dedupe drop a real request.
                request_id=f"{self.run_label}-{tag}-i{index}",
                messages=[{"role": "user", "content": content}],
                max_tokens=max_tokens,
                temperature=self.temperature,
                extra=extra,
            )

        return next_spec

    def manifest(self) -> dict:
        """The reproduction bundle's half of the promise: everything needed to regenerate
        the exact prompt sequence, as a plain JSON-serialisable dict."""
        plan = self.output_plan
        if isinstance(plan, FixedOutput):
            basis, ignore_eos, output_tokens = "fixed", plan.ignore_eos, plan.output_tokens
        elif isinstance(plan, CappedOutput):
            basis, ignore_eos, output_tokens = "capped", False, plan.output_tokens
        else:
            basis, ignore_eos, output_tokens = "model-decided", False, None
        m = {
            "run_label": self.run_label,
            "seed": self.seed,
            "sampler": self.source.sampler_rule,
            "cache_policy": self.cache_policy,
            "think_time_s": self.think_time_s,
            # Three bases, and a bundle must keep them recognisable after publication:
            # "fixed" sent ignore_eos with the length and every request decoded exactly
            # output_tokens; "capped" sent the length alone, an anti-collapse ceiling with
            # EOS honoured; "model-decided" sent no length at all. ignore_eos and
            # output_tokens therefore record what went on the wire, and output_tokens is
            # present exactly when a length was sent -- if "capped at 512" and "uncapped"
            # were indistinguishable in this file, the run could not be reproduced from its
            # own bundle. output_basis gains a third value rather than a new key being
            # invented, because how the output length was decided is the axis that key
            # already names.
            "output_basis": basis,
            "ignore_eos": ignore_eos,
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
        if output_tokens is not None:
            m["output_tokens"] = output_tokens
        if self.cache_policy == "unknown":
            m["cache_policy_u_reason"] = self.unknown_cache_reason
        return m
