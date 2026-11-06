"""engine.py — everything that touches MLX, on one thread.

MLX ties GPU queues to the thread that created them, so a single GpuWorker thread
owns the vision-language model (behind a small adapter), every screen read, every
answer, and Kokoro text-to-speech. Other threads only hand it work.

Pieces:
  - GemmaAdapter: load / set_budget / read_screen(image) -> Snapshot / generate(cache, ids)
      * fixed screen/question split (review D2): the screen prefix is the chat template
        up to a fixed "Question:" marker after the image; the question is its own suffix.
      * spare cache copy prepared in the background (D12).
      * answers stop at the 3rd sentence or 120 tokens (D9); cancel checked every token (D6).
  - Tts: Kokoro, one sentence at a time, cancel checked per sentence (D6). Kokoro's built-in
    ~300 ms leading silence is trimmed; the first chunk can be an early phrase (first_chunk="phrase").
  - GpuWorker: answer jobs first; a new screen read never starts while a turn is active
    (the naive baseline); a read already running is not interrupted, and the wait is logged.
"""

from __future__ import annotations

import copy
import os
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator, Optional

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import mlx.core as mx  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent
GEMMA_PATH = ROOT / "models/gemma-4-e4b-it-4bit"
KOKORO_PATH = ROOT / "models/kokoro-82m-bf16"

SPLIT_MARKER = "Question:"
SPOKEN_INSTRUCTION = (
    "You are a voice assistant that can see the user's screen in the image above. "
    "Answer in one to three short spoken sentences. No lists, no markdown, no code blocks."
)
MAX_TOKENS = 120
MAX_SENTENCES = 3
_SENTENCE_END = re.compile(r"[.!?](?=\s|$)")


# ---------------------------------------------------------------- helpers

def clone_cache(cache):
    """Deep copy of a list of KV caches (keys/values copied; ring indices copied by value)."""
    out = []
    for c in cache:
        n = copy.copy(c)
        n.keys = None if c.keys is None else mx.array(c.keys)
        n.values = None if c.values is None else mx.array(c.values)
        out.append(n)
    mx.eval([x for n in out for x in (n.keys, n.values) if x is not None])
    return out


def count_sentences(text: str) -> int:
    """Sentence ends = . ! ? followed by whitespace or end of text (so 0.0825 is not an end)."""
    return len(_SENTENCE_END.findall(text.strip()))


_PHRASE_PUNCT = re.compile(r"[,;:\u2014\u2013](?=\s)")
BREAK_WORDS = {"and", "but", "because", "which", "that", "so", "with", "in", "on", "for", "to", "of",
               "from", "is", "are", "was", "were", "shows", "says", "means", "has", "have"}


def split_first_phrase(buffer: str, min_words: int = 4, max_words: int = 7) -> Optional[tuple[str, str]]:
    """Early first chunk for TTS: split an unfinished first sentence at a natural pause.

    1) punctuation (, ; : dash) after >= min_words words, or
    2) once >= max_words complete words have arrived, before the first break word
       ("is", "that", "in", ...) at word >= min_words.
    Returns (phrase, rest) or None to keep waiting (the sentence end then decides)."""
    for m in _PHRASE_PUNCT.finditer(buffer):
        if len(buffer[:m.end()].split()) >= min_words:
            return buffer[:m.end()].strip(), buffer[m.end():]
    words = list(re.finditer(r"\S+(?=\s)", buffer))          # complete words only
    if len(words) < max_words:
        return None
    for w in words[min_words:]:
        if w.group().lower() in BREAK_WORDS:
            return buffer[:w.start()].strip(), buffer[w.start():]
    return None


def trim_silence(audio: np.ndarray, sr: int, lead_keep_ms: float = 30, tail_keep_ms: float = 120,
                 thr_db: float = -50) -> np.ndarray:
    """Cut Kokoro's built-in leading (~300 ms) and trailing (~450 ms) silence, keeping a short pad."""
    idx = np.flatnonzero(np.abs(audio) > 10 ** (thr_db / 20))
    if len(idx) == 0:
        return audio
    a = max(0, idx[0] - int(sr * lead_keep_ms / 1000))
    b = min(len(audio), idx[-1] + 1 + int(sr * tail_keep_ms / 1000))
    return audio[a:b]


def split_complete_sentences(buffer: str) -> tuple[list[str], str]:
    """Split buffered text into complete sentences and the unfinished remainder."""
    out, start = [], 0
    for m in _SENTENCE_END.finditer(buffer):
        end = m.end()
        # only complete if followed by whitespace (not end of buffer: more text may come, e.g. "3.5")
        if end < len(buffer) and buffer[end].isspace():
            s = buffer[start:end].strip()
            if s:
                out.append(s)
            start = end
    return out, buffer[start:]


# ---------------------------------------------------------------- model adapter

@dataclass
class Snapshot:
    """A finished screen read: the model's cache after [instructions + screen + 'Question:']."""
    frame_id: int
    captured_at: float          # wall clock (time.time()) when the frame was captured
    read_started: float
    read_done: float
    budget: int
    prefix_len: int
    master: list = field(repr=False)
    spare: Optional[list] = field(default=None, repr=False)


class GemmaAdapter:
    """Gemma 4 E4B (4-bit MLX) with a pre-read screen prefix. Call every method on one thread."""

    name = "gemma-4-e4b-it-4bit"

    def __init__(self, path: Path = GEMMA_PATH, instruction: str = SPOKEN_INSTRUCTION):
        self.path = Path(path)
        self.instruction = instruction
        self.model = self.processor = None

    # -- setup
    def load(self):
        from mlx_vlm import load
        self.model, self.processor = load(str(self.path))
        self.lm = self.model.language_model
        self.tok = self.processor.tokenizer
        vocab = self.tok.get_vocab()
        self.stop_ids = {self.tok.eos_token_id} | {
            self.tok.convert_tokens_to_ids(t) for t in ("<turn|>", "<end_of_turn>") if t in vocab}
        return self

    def set_budget(self, n: int):
        self.processor.image_processor.max_soft_tokens = n
        self.budget = n

    # -- prompt pieces
    def full_prompt(self, question: str) -> str:
        from mlx_vlm.prompt_utils import apply_chat_template
        return apply_chat_template(self.processor, self.model.config,
                                   f"{self.instruction}\n{SPLIT_MARKER} {question}", num_images=1)

    def prefix_prompt(self) -> str:
        p = self.full_prompt("x")
        i = p.index(SPLIT_MARKER) + len(SPLIT_MARKER)
        return p[:i]

    def suffix_ids(self, question: str) -> list[int]:
        full, pre = self.full_prompt(question), self.prefix_prompt()
        assert full.startswith(pre)
        return self.tok.encode(full[len(pre):], add_special_tokens=False)

    def _inputs(self, image, prompt: str):
        from mlx_vlm.utils import prepare_inputs
        return prepare_inputs(self.processor, images=[image], prompts=[prompt])

    def _prefill(self, inp, upto: Optional[int] = None):
        """Encode the image and prefill input_ids[:upto] into a fresh cache. Returns (cache, n)."""
        ids = inp["input_ids"]
        extra = {k: v for k, v in inp.items() if k not in ("input_ids", "pixel_values", "attention_mask", "mask")}
        emb = self.model.get_input_embeddings(input_ids=ids, pixel_values=inp["pixel_values"], **extra)
        n = ids.shape[1] if upto is None else upto
        kw = {k: extra[k][:, :n] for k in ("mm_token_type_ids", "token_type_ids") if extra.get(k) is not None}
        cache = self.model.make_cache()
        logits = self.lm(inputs=None, inputs_embeds=emb.inputs_embeds[:, :n],
                         per_layer_inputs=emb.per_layer_inputs[:, :n], cache=cache, **kw).logits
        mx.eval([c.state for c in cache])
        return cache, n, logits

    # -- the two paths
    def read_screen(self, image, frame_id: int = 0, captured_at: Optional[float] = None) -> Snapshot:
        t0 = time.time()
        inp = self._inputs(image, self.prefix_prompt())
        cache, n, _ = self._prefill(inp)
        return Snapshot(frame_id=frame_id, captured_at=captured_at or t0, read_started=t0,
                        read_done=time.time(), budget=getattr(self, "budget", None),
                        prefix_len=n, master=cache)

    def generate(self, cache, ids: list[int], cancel: Optional[threading.Event] = None,
                 max_tokens: int = MAX_TOKENS, max_sentences: int = MAX_SENTENCES,
                 on_first_token: Optional[Callable[[], None]] = None) -> Iterator[str]:
        """Greedy decode after feeding `ids` into `cache` (mutated). Yields text deltas.
        Stops at a stop token, `max_sentences` sentence ends, `max_tokens`, or `cancel`.
        Sets self.last_stop to 'eos' | 'sentences' | 'cap' | 'cancel'."""
        self.last_stop = None
        y = mx.argmax(self.lm(inputs=mx.array([ids]), cache=cache).logits[:, -1, :], axis=-1)
        out, text = [], ""
        while True:
            t = int(y.item())
            if not out and on_first_token:
                on_first_token()
            if t in self.stop_ids:
                self.last_stop = "eos"
                return
            out.append(t)
            new = self.tok.decode(out)
            delta, text = new[len(text):], new
            if delta:
                yield delta
            if count_sentences(text) >= max_sentences and _SENTENCE_END.search(text.rstrip()[-1:] or " "):
                self.last_stop = "sentences"
                return
            if len(out) >= max_tokens:
                self.last_stop = "cap"
                return
            if cancel is not None and cancel.is_set():
                self.last_stop = "cancel"
                return
            y = mx.argmax(self.lm(inputs=y[None], cache=cache).logits[:, -1, :], axis=-1)

    def answer_cached(self, snap: Snapshot, question: str, **kw) -> str:
        return "".join(self.generate(clone_cache(snap.master), self.suffix_ids(question), **kw))

    def answer_uncached(self, image, question: str, **kw) -> str:
        """Reference path: whole prompt (screen + question) prefilled in one pass, same decode code."""
        inp = self._inputs(image, self.full_prompt(question))
        n = inp["input_ids"].shape[1]
        cache, _, _ = self._prefill(inp, upto=n - 1)          # all but the last prompt token
        last = int(inp["input_ids"][0, n - 1].item())
        return "".join(self.generate(cache, [last], **kw))

    def full_ids(self, image, question: str) -> list[int]:
        return self._inputs(image, self.full_prompt(question))["input_ids"][0].tolist()

    def prefix_ids(self, image) -> list[int]:
        return self._inputs(image, self.prefix_prompt())["input_ids"][0].tolist()


# ---------------------------------------------------------------- text to speech

class Tts:
    """Kokoro 82M on MLX. Must be used on the GPU worker thread."""

    def __init__(self, path: Path = KOKORO_PATH, voice: str = "af_heart", speed: float = 1.0):
        self.path, self.voice, self.speed = Path(path), voice, speed
        self.model = None
        self.sample_rate = 24000

    def load(self):
        from mlx_audio.tts.utils import load_model
        self.model = load_model(str(self.path))
        return self

    def synth(self, sentence: str, cancel: Optional[threading.Event] = None, trim: bool = True,
              tail_keep_ms: float = 120) -> Iterator[np.ndarray]:
        """Yields audio chunks. With trim, the first chunk loses its leading silence and every
        chunk's trailing silence is cut to tail_keep_ms (short pad between sentences)."""
        first = True
        for r in self.model.generate(sentence, voice=self.voice, speed=self.speed, lang_code="a"):
            if cancel is not None and cancel.is_set():
                return
            self.sample_rate = r.sample_rate
            a = np.asarray(r.audio, dtype=np.float32).reshape(-1)
            if trim:
                a = trim_silence(a, r.sample_rate, lead_keep_ms=30 if first else 1e9, tail_keep_ms=tail_keep_ms)
            first = False
            yield a


# ---------------------------------------------------------------- the GPU worker

@dataclass
class AnswerJob:
    question: str
    emit_audio: Callable[[np.ndarray, int], None]
    mark: Callable[[str], None]
    submitted: float = field(default_factory=time.perf_counter)
    read_in_flight_at_submit: bool = False
    done: threading.Event = field(default_factory=threading.Event)
    result: dict = field(default_factory=dict)


class GpuWorker(threading.Thread):
    """Owns MLX. Job priority: answer > spare cache copy > screen read.

    Naive baseline policy (MVP): a screen read never *starts* while a turn is active
    (recording, transcribing or answering); a read that is already running finishes.
    """

    def __init__(self, adapter: GemmaAdapter, tts: Tts, budget: int = 1120, warm_image=None,
                 first_chunk: str = "phrase", trim_silence: bool = True, mlx_cache_gb: float = 0.0):
        super().__init__(daemon=True, name="gpu-worker")
        self.adapter, self.tts, self.budget, self.warm_image = adapter, tts, budget, warm_image
        assert first_chunk in ("phrase", "sentence")
        self.first_chunk, self.trim, self.mlx_cache_gb = first_chunk, trim_silence, mlx_cache_gb
        self.cv = threading.Condition()
        self.pending_frame = None             # (frame_id, captured_at, image) latest wins
        self.answer_q: list[AnswerJob] = []
        self.turn_active = False
        self.reading = False
        self.snapshot: Optional[Snapshot] = None
        self.cancel = threading.Event()
        self.ready = threading.Event()
        self.startup: dict = {}
        self.stats = {"reads": 0, "frames_dropped": 0}
        self.read_log: list[dict] = []
        self.error: Optional[BaseException] = None
        self._stop = False

    # -- called from other threads
    def submit_frame(self, frame_id: int, captured_at: float, image):
        with self.cv:
            if self.pending_frame is not None:
                self.stats["frames_dropped"] += 1
            self.pending_frame = (frame_id, captured_at, image)
            self.cv.notify_all()

    def set_turn_active(self, active: bool):
        with self.cv:
            self.turn_active = active
            self.cv.notify_all()

    def submit_answer(self, job: AnswerJob) -> AnswerJob:
        with self.cv:
            job.read_in_flight_at_submit = self.reading
            self.cancel.clear()
            self.answer_q.append(job)
            self.cv.notify_all()
        return job

    def interrupt(self) -> float:
        """Stop the current answer: generation and TTS check this every token / sentence."""
        self.cancel.set()
        return time.perf_counter()

    def stop(self):
        with self.cv:
            self._stop = True
            self.cv.notify_all()

    def first_snapshot_wait(self, timeout: float) -> bool:
        t = time.time() + timeout
        while time.time() < t:
            if self.snapshot is not None:
                return True
            time.sleep(0.05)
        return False

    # -- worker thread
    def run(self):
        try:
            self._startup()
            self.ready.set()
            while True:
                with self.cv:
                    while not self._stop and not self.answer_q and not self._spare_needed() \
                            and not (self.pending_frame and not self.turn_active):
                        self.cv.wait(0.1)
                    if self._stop:
                        return
                    if self.answer_q:
                        job = self.answer_q.pop(0)
                        kind = "answer"
                    elif self._spare_needed():
                        kind = "spare"
                    else:
                        frame = self.pending_frame
                        self.pending_frame = None
                        self.reading = True
                        kind = "read"
                if kind == "answer":
                    self._answer(job)
                elif kind == "spare":
                    self._make_spare()
                else:
                    try:
                        self._read(*frame)
                    finally:
                        with self.cv:
                            self.reading = False
        except BaseException as e:  # surface to the app instead of dying silently
            self.error = e
            self.ready.set()
            raise

    def _startup(self):
        # MLX keeps freed buffers for reuse (~2.2 GB here); capping it cut the footprint
        # from ~8.7 to ~6.5 GB with no slowdown (2026-11-06 check). None = MLX default.
        if self.mlx_cache_gb is not None:
            mx.set_cache_limit(int(self.mlx_cache_gb * 1e9))
        t0 = time.perf_counter()
        self.adapter.load()
        self.adapter.set_budget(self.budget)
        self.startup["gemma_load_s"] = round(time.perf_counter() - t0, 2)
        t0 = time.perf_counter()
        self.tts.load()
        for _ in self.tts.synth("Ready."):
            pass
        self.startup["kokoro_load_warm_s"] = round(time.perf_counter() - t0, 2)
        if self.warm_image is not None:   # compile kernels for this budget: one read + a short answer
            t0 = time.perf_counter()
            snap = self.adapter.read_screen(self.warm_image)
            "".join(self.adapter.generate(clone_cache(snap.master), self.adapter.suffix_ids("hi"), max_tokens=4))
            self.startup["gemma_warm_s"] = round(time.perf_counter() - t0, 2)

    def _spare_needed(self) -> bool:
        s = self.snapshot
        return s is not None and s.spare is None

    def _make_spare(self):
        s = self.snapshot
        t0 = time.perf_counter()
        spare = clone_cache(s.master)
        if self.snapshot is s:
            s.spare = spare
        self.stats["last_spare_s"] = round(time.perf_counter() - t0, 4)

    def _read(self, frame_id, captured_at, image):
        t0 = time.perf_counter()
        snap = self.adapter.read_screen(image, frame_id=frame_id, captured_at=captured_at)
        self.snapshot = snap                   # swap: answers use the newest completed read
        self.stats["reads"] += 1
        self.read_log.append({"frame_id": frame_id, "captured_at": captured_at,
                              "read_s": round(time.perf_counter() - t0, 3), "prefix_len": snap.prefix_len})

    def _answer(self, job: AnswerJob):
        r = job.result
        picked = time.perf_counter()
        r["wait_behind_read_s"] = round(picked - job.submitted, 4)
        r["read_in_flight_at_submit"] = job.read_in_flight_at_submit
        snap = self.snapshot
        if snap is None:
            r["error"] = "no screen read yet"
            job.done.set()
            return
        r.update(frame_id=snap.frame_id, screen_captured_at=snap.captured_at,
                 screen_read_done=snap.read_done, budget=snap.budget)
        t0 = time.perf_counter()
        if snap.spare is not None:
            cache, snap.spare = snap.spare, None
        else:
            cache = clone_cache(snap.master)
        r["restore_s"] = round(time.perf_counter() - t0, 4)
        job.mark("restore_done")
        ids = self.adapter.suffix_ids(job.question)
        r["question_tokens"] = len(ids)
        text, buf, sentences = "", "", 0
        first_audio_marked = False

        chunks = 0

        def speak(sentence, mid_sentence=False):
            nonlocal first_audio_marked, sentences, chunks
            chunks += 1
            if not mid_sentence:
                sentences += 1
            if chunks == 1:
                job.mark("first_sentence_ready")      # first chunk handed to TTS (phrase or sentence)
                r["first_chunk_text"] = sentence
            for audio in self.tts.synth(sentence, cancel=self.cancel, trim=self.trim,
                                        tail_keep_ms=40 if mid_sentence else 120):
                if not first_audio_marked:
                    job.mark("tts_first_chunk")
                    first_audio_marked = True
                job.emit_audio(audio, self.tts.sample_rate)
                if self.cancel.is_set():
                    return

        for delta in self.adapter.generate(cache, ids, cancel=self.cancel,
                                           on_first_token=lambda: job.mark("first_token")):
            text += delta
            buf += delta
            done, buf = split_complete_sentences(buf)
            if self.first_chunk == "phrase" and chunks == 0 and not done:
                ph = split_first_phrase(buf)
                if ph is not None:
                    speak(ph[0], mid_sentence=True)
                    buf = ph[1]
            for s in done:
                if self.cancel.is_set():
                    break
                speak(s)
            if self.cancel.is_set():
                break
        if buf.strip() and not self.cancel.is_set():
            speak(buf.strip())
        r.update(answer=text.strip(), stop=self.adapter.last_stop if not self.cancel.is_set() else "cancel",
                 cap_hit=self.adapter.last_stop == "cap", sentences_spoken=sentences, chunks_spoken=chunks,
                 first_chunk_mode=self.first_chunk,
                 cancelled=self.cancel.is_set())
        job.mark("turn_complete")
        job.done.set()
