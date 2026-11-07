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
  - GpuWorker: the scheduler. Answer jobs first; screen reads run in pieces and pause for
    answers (policy p2), or follow the MVP baseline (p0) / wait-for-read (p1) for comparison.
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

# M3: keep torch out of the app process. transformers (Gemma's processor) and spaCy's thinc (Kokoro's
# phonemizer) both treat it as optional; a None entry makes `import torch` fail and find_spec return
# None. torch stays installed for the Core ML conversion (m0_ane/vision_coreml.py). LV_ALLOW_TORCH=1 undoes it.
if os.environ.get("LV_ALLOW_TORCH") != "1":
    import sys
    sys.modules.setdefault("torch", None)

import mlx.core as mx  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent
GEMMA_PATH = ROOT / "models/gemma-4-e4b-it-4bit"
KOKORO_PATH = ROOT / "models/kokoro-82m-bf16"

SPLIT_MARKER = "Question:"
TURN_USER, TURN_END, TURN_MODEL = "<|turn>user\n", "<turn|>\n", "<|turn>model\n"
SPOKEN_INSTRUCTION = (
    "You are a helpful voice assistant. The image above is the user's screen right now. "
    "Use it when the question is about the screen; answer any other question normally. "
    "Answer in one to three short spoken sentences. Write numbers as digits. "
    "No lists, no markdown, no code blocks."
)
MVP_INSTRUCTION = (            # until 2026-11-06: read as screen-only ("What is 3 plus 3?" was refused)
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

def _has_tensor(path: Path, key: str) -> bool:
    import json as _json
    import struct
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return key in _json.loads(f.read(n))


class MMapPerLayerEmbedding:
    """M2 (ARCHITECTURE.md §3.4): Gemma's per-layer embedding (PLE) table read from the model file.

    The table is 262,144 rows x 6,048 B (4-bit, group 64): 1.59 GB, but each token needs one row.
    Instead of keeping it in MLX memory, the three tensors (weight, scales, biases) are mapped
    read-only from model.safetensors; a call copies only the rows for the unique ids in the
    input and dequantizes them exactly like nn.QuantizedEmbedding:
        dequantize(weight[ids], scales[ids], biases[ids], group_size, bits, mode)
    so the result is bit-identical. Touched pages are clean file cache that macOS can drop."""

    def __init__(self, path: Path, prefix: str, group_size: int, bits: int, mode: str = "affine"):
        import json as _json
        import struct
        self.group_size, self.bits, self.mode = group_size, bits, mode
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            header = _json.loads(f.read(n))
        base = 8 + n
        dt = {"U32": np.uint32, "BF16": np.uint16, "F16": np.float16, "F32": np.float32}

        def mm(name):
            h = header[f"{prefix}.{name}"]
            a, b = h["data_offsets"]
            return np.memmap(path, dtype=dt[h["dtype"]], mode="r", offset=base + a, shape=tuple(h["shape"])), h["dtype"]

        (self.w, _), (self.s, self.s_dt), (self.b, self.b_dt) = mm("weight"), mm("scales"), mm("biases")
        self.rows_read = 0

    def _mx(self, arr, dtype_name):
        a = mx.array(np.ascontiguousarray(arr))
        return a.view(mx.bfloat16) if dtype_name == "BF16" else a

    def __call__(self, ids: mx.array) -> mx.array:
        ids_np = np.asarray(ids)
        uniq, inv = np.unique(ids_np.reshape(-1), return_inverse=True)
        self.rows_read += len(uniq)
        deq = mx.dequantize(self._mx(self.w[uniq], "U32"), scales=self._mx(self.s[uniq], self.s_dt),
                            biases=self._mx(self.b[uniq], self.b_dt), group_size=self.group_size,
                            bits=self.bits, mode=self.mode)
        return deq[mx.array(inv.astype(np.int32))].reshape(*ids_np.shape, deq.shape[-1])


class _audio_config_off:
    """Load Gemma with audio_config = None, so mlx-vlm never builds the audio tower and its
    sanitize step drops the audio weights (mlx_vlm/models/gemma4/gemma4.py:61-73, :269).
    MLX loads weights lazily, so dropped weights are never read from disk."""

    def __enter__(self):
        import mlx_vlm.utils as U
        self.U, self.orig = U, U.load_config

        def patched(model_path, **kw):
            cfg = self.orig(model_path, **kw)
            cfg["audio_config"] = None
            return cfg

        U.load_config = patched
        return self

    def __exit__(self, *exc):
        self.U.load_config = self.orig
        return False


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
    tail: str = "screen"            # what the context ends with: "screen" (…Question:) or "turn" (a finished exchange)
    ex_included: int = 0            # exchanges (monotonic count) included in this context
    encoded: Optional[tuple] = field(default=None, repr=False)   # (inputs, image tokens) for re-anchoring


class GemmaAdapter:
    """Gemma 4 E4B (4-bit MLX) with a pre-read screen prefix. Call every method on one thread."""

    name = "gemma-4-e4b-it-4bit"

    def __init__(self, path: Path = GEMMA_PATH, instruction: str = SPOKEN_INSTRUCTION,
                 skip_audio: bool = True, skip_shared: bool = True, stream_ple: bool = True):
        self.path = Path(path)
        self.instruction = instruction
        self.model = self.processor = None
        # M1 (ARCHITECTURE.md §3.3): don't build Gemma's audio encoder (0.61 GB; WhisperKit does speech).
        self.skip_audio = skip_audio
        # G1 (§4.2): during prefill, run the 18 KV-shared layers only for the last token. They store
        # no cache, so the snapshot is bit-identical (checked 2026-11-06). No speed gain in reads (S0): MLX is
        # lazy and a read evaluates only the cache, so those layers never ran anyway. Kept as a guard.
        self.skip_shared = skip_shared
        # M2 (§3.4): read the 1.59 GB per-layer embedding table's rows from the model file on demand.
        self.stream_ple = stream_ple

    # -- setup
    def load(self):
        from mlx_vlm import load
        lazy = self.stream_ple            # lazy: the PLE table must never be read into memory
        if self.skip_audio:
            with _audio_config_off():
                self.model, self.processor = load(str(self.path), lazy=lazy)
        else:
            self.model, self.processor = load(str(self.path), lazy=lazy)
        if self.stream_ple:
            tm = self.model.language_model.model
            old = tm["embed_tokens_per_layer"]
            files = [f for f in self.path.glob("*.safetensors") if _has_tensor(f, "language_model.model.embed_tokens_per_layer.weight")]
            tm.pop("embed_tokens_per_layer")             # drop the lazy array: never evaluated, never read
            tm.embed_tokens_per_layer = MMapPerLayerEmbedding(files[0], "language_model.model.embed_tokens_per_layer",
                                                              old.group_size, old.bits, getattr(old, "mode", "affine"))
            del old
        if lazy:
            mx.eval(self.model.parameters())
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

    def suffix_ids(self, question: str, tail: str = "screen") -> list[int]:
        """Question tokens to append to a snapshot. After the screen turn ("…Question:") the question
        continues that turn; after a finished exchange it opens a new user turn."""
        if tail == "turn":
            return self.text_ids(f"{TURN_USER}{question}{TURN_END}{TURN_MODEL}")
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
        if self.skip_shared:
            kw["logits_to_keep"] = 1          # callers never use prefill logits
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

    # -- context pieces (conversation history, ARCHITECTURE.md §9). Every piece runs text layers
    # 0-23 only (layers 24-41 share KV and store nothing) and yields once per piece.
    # Layout: <bos> [older exchanges] [screen turn: image + instructions + "Question:"] q1 -> a1 [<|turn>user q2 …]
    def text_ids(self, s: str) -> list[int]:
        return self.tok.encode(s, add_special_tokens=False)

    def screen_turn_prompt(self, note: Optional[str] = None) -> str:
        """The screen turn without <bos>. `note` (e.g. the open-windows line) goes right before
        "Question:", after the instructions."""
        p = self.prefix_prompt()
        assert p.startswith("<bos>")
        p = p[len("<bos>"):]
        if note:
            i = p.rindex(SPLIT_MARKER)
            p = p[:i] + note.strip() + "\n" + p[i:]
        return p

    EARLIER = "(Earlier, about a previous screen) "

    def exchange_ids(self, q: str, a: str, first_after_screen: bool = False, older: bool = False) -> list[int]:
        """older=True: an exchange about a previous screen, moved before the current screen turn. It is
        marked as earlier: in eval/history_eval.py (2026-11-06) this cut copying of a wrong earlier answer
        from 2/20 to 1/20 and kept cross-screen recall at 5/5 (a caution in the instructions made both worse)."""
        if first_after_screen:                     # closes the open screen turn ("…Question:")
            return self.text_ids(f" {q}{TURN_END}{TURN_MODEL}{a}{TURN_END}")
        return self.text_ids(f"{TURN_USER}{self.EARLIER if older else ''}{q}{TURN_END}{TURN_MODEL}{a}{TURN_END}")

    def _layer_steps(self, cache, h, pli, mm, layers_per_piece: int):
        tm = self.lm.model
        n_text = tm.first_kv_shared_layer_idx
        if tm.hidden_size_per_layer_input and pli is not None:
            pli = tm.project_per_layer_inputs(h, pli)
        masks = tm._make_masks(h, cache + [None] * (len(tm.layers) - len(cache)), mm)
        for start in range(0, n_text, layers_per_piece):
            end = min(n_text, start + layers_per_piece)
            for idx in range(start, end):          # layers < 24 own their KV: no sharing
                p_in = pli[:, :, idx, :] if pli is not None else None
                h, _, _ = tm.layers[idx](h, masks[idx], cache[idx], per_layer_input=p_in,
                                         shared_kv=None, offset=None)
            mx.eval(h, [c.state for c in cache[start:end]])
            yield None

    def prefill_text_steps(self, cache, ids: list[int], layers_per_piece: int = 2):
        """Append text tokens to `cache` (mutated). Generator; returns the cache."""
        tm = self.lm.model
        x = mx.array([ids])
        h = tm.embed_tokens(x) * tm.embed_scale
        pli = tm.get_per_layer_inputs(x) if tm.hidden_size_per_layer_input else None
        yield from self._layer_steps(cache, h, pli, None, layers_per_piece)
        return cache

    def new_prefix_steps(self, exchanges, layers_per_piece: int = 2):
        """<bos> + older exchanges, from scratch. Returns the cache."""
        ids = [self.tok.bos_token_id] + [t for q, a in exchanges for t in self.exchange_ids(q, a, older=True)]
        cache = yield from self.prefill_text_steps(self.model.make_cache(), ids, layers_per_piece)
        return cache

    def extend_prefix_steps(self, prefix_cache, exchanges, layers_per_piece: int = 2):
        """A copy of the prefix with more exchanges appended. Returns the new cache."""
        ids = [t for q, a in exchanges for t in self.exchange_ids(q, a, older=True)]
        cache = yield from self.prefill_text_steps(clone_cache(prefix_cache), ids, layers_per_piece)
        return cache

    def encode_steps(self, image, note: Optional[str] = None, inp=None):
        """Image prep + vision encoder in pieces (patch embedding, 16 layers, pooling).
        Returns (inputs of the screen turn without <bos>, image tokens). `inp`: already-prepared inputs."""
        inp = inp if inp is not None else self._inputs(image, self.screen_turn_prompt(note))
        yield None
        vt = self.model.vision_tower
        pixel_values = inp["pixel_values"]
        B, C, H, W = pixel_values.shape
        num_real = (H // vt.patch_size) * (W // vt.patch_size)
        output_length = num_real // (vt.pooling_kernel_size ** 2)
        positions, padding_mask, _ = vt._patch_positions_single(H, W, max_patches=num_real)
        patch_positions = mx.array(np.tile(positions[None], (B, 1, 1)))
        padding_positions = mx.array(np.tile(padding_mask[None], (B, 1)))
        h = vt.patch_embedder(pixel_values, patch_positions, padding_positions)
        valid = ~padding_positions
        attn_mask = mx.expand_dims(valid, 1) * mx.expand_dims(valid, 2)
        attn_mask = mx.where(attn_mask, mx.array(0.0, dtype=h.dtype), mx.array(-1e4, dtype=h.dtype))
        attn_mask = mx.expand_dims(attn_mask, 1)
        mx.eval(h, attn_mask)
        yield None
        for layer in vt.encoder.layers:
            h = layer(h, patch_positions, attn_mask)
            mx.eval(h)
            yield None
        pooled, pool_mask = vt.pooler(h, patch_positions, padding_positions, output_length=output_length)
        valid_mask = pool_mask if pool_mask.shape[1] == output_length else ~pool_mask
        real = [pooled[i, :int(valid_mask[i].astype(mx.int32).sum().item())] for i in range(B)]
        feats = mx.concatenate(real, axis=0)[None]
        if vt.config.standardize:
            feats = (feats - vt.std_bias) * vt.std_scale
        feats = self.model.embed_vision(feats)
        mx.eval(feats)
        yield None
        return inp, feats

    def encode_steps_ane(self, image, ane, note: Optional[str] = None):
        """encode_steps with the vision tower on the Neural Engine (ane_vision.AneVisionEncoder, §6.1).
        Pieces: image prep, one per Core ML chunk (blocking calls, GPU idle), embed_vision on the GPU.
        Falls back to encode_steps when no converted bucket matches the image's processed size.
        Sets self.last_encode to 'ane' or 'gpu'."""
        inp = self._inputs(image, self.screen_turn_prompt(note))
        pv = inp["pixel_values"]
        if ane is None or not ane.has(int(pv.shape[2]), int(pv.shape[3])):
            self.last_encode = "gpu"
            return (yield from self.encode_steps(image, note, inp=inp))
        self.last_encode = "ane"
        yield None
        g = ane.encode_steps(np.asarray(pv, dtype=np.float32))
        while True:
            try:
                next(g)
            except StopIteration as e:
                tokens = e.value
                break
            yield None
        vt = self.model.vision_tower
        feats = mx.array(tokens).astype(vt.patch_embedder.input_proj.weight.dtype)
        if vt.config.standardize:
            feats = (feats - vt.std_bias) * vt.std_scale
        feats = self.model.embed_vision(feats)
        mx.eval(feats)
        yield None
        return inp, feats

    def screen_steps(self, prefix_cache, encoded, layers_per_piece: int = 2):
        """Append the screen turn to a copy of prefix_cache, or to a fresh cache starting with <bos>
        when there is no older history (then the tokens are exactly today's single-turn prompt).
        Returns (cache, number of tokens in the context)."""
        inp, feats = encoded
        ids, extra = inp["input_ids"], {k: v for k, v in inp.items()
                                         if k not in ("input_ids", "pixel_values", "attention_mask", "mask")}
        if prefix_cache is None:
            ids = mx.concatenate([mx.array([[self.tok.bos_token_id]], dtype=ids.dtype), ids], axis=1)
            extra = {k: mx.concatenate([mx.zeros((1, 1), dtype=v.dtype), v], axis=1) for k, v in extra.items()}
            cache = self.model.make_cache()
        else:
            cache = clone_cache(prefix_cache)
        emb = self.model.get_input_embeddings(input_ids=ids, pixel_values=inp["pixel_values"],
                                              cached_image_features=feats, **extra)
        mx.eval(emb.inputs_embeds, emb.per_layer_inputs)
        yield None
        yield from self._layer_steps(cache, emb.inputs_embeds, emb.per_layer_inputs,
                                     extra.get("mm_token_type_ids"), layers_per_piece)
        return cache, int(cache[0].offset)

    def read_pieces_estimate(self, layers_per_piece: int = 2) -> int:
        tm = self.lm.model
        return 1 + 1 + len(self.model.vision_tower.encoder.layers) + 1 + 1 + \
            -(-tm.first_kv_shared_layer_idx // layers_per_piece)

    def read_screen_steps(self, image, frame_id: int = 0, captured_at: Optional[float] = None,
                          layers_per_piece: int = 2, split: str = "layers", chunk: int = 256):
        """The same read as read_screen, split into pieces the scheduler can pause between.

        Generator: does one piece of GPU work, evaluates it, then yields (pieces_done, pieces_total).
        Returns the Snapshot (StopIteration.value). Pieces: image prep, patch embedding, each of
        the 16 vision layers, pooling + projection, token embeddings, then the text prefill.

        split="layers" (default, bit-exact): all tokens go through text layers 0-23, `layers_per_piece`
        at a time, and the read stops there. Layers 24-41 share earlier layers' KV and store nothing,
        so a snapshot never needs them (G1, taken one step further: no shared layers, no logits).
        split="tokens": the usual chunked prefill, `chunk` tokens at a time. Changes rounding
        (different matrix shapes): 17/20 answers word-identical at 1120 on 2026-11-06, same content.

        The vision and text loops mirror mlx_vlm gemma4 VisionModel.__call__ (fixed-size image
        branch) and Gemma4TextModel.__call__; eval/exactness.py and tests check the result."""
        import math
        t0 = time.time()
        inp = self._inputs(image, self.prefix_prompt())
        ids = inp["input_ids"]
        n = ids.shape[1]
        extra = {k: v for k, v in inp.items() if k not in ("input_ids", "pixel_values", "attention_mask", "mask")}
        vt = self.model.vision_tower
        layers = vt.encoder.layers
        tm = self.lm.model
        n_text = tm.first_kv_shared_layer_idx
        text_pieces = math.ceil(n_text / layers_per_piece) if split == "layers" else math.ceil(n / chunk)
        total = 2 + len(layers) + 2 + text_pieces
        done = 1
        yield done, total

        # patch embedding (VisionModel.__call__, no pixel_position_ids branch)
        pixel_values = inp["pixel_values"]
        B, C, H, W = pixel_values.shape
        num_real = (H // vt.patch_size) * (W // vt.patch_size)
        output_length = num_real // (vt.pooling_kernel_size ** 2)
        positions, padding_mask, _ = vt._patch_positions_single(H, W, max_patches=num_real)
        patch_positions = mx.array(np.tile(positions[None], (B, 1, 1)))
        padding_positions = mx.array(np.tile(padding_mask[None], (B, 1)))
        h = vt.patch_embedder(pixel_values, patch_positions, padding_positions)
        valid = ~padding_positions
        attn_mask = mx.expand_dims(valid, 1) * mx.expand_dims(valid, 2)
        attn_mask = mx.where(attn_mask, mx.array(0.0, dtype=h.dtype), mx.array(-1e4, dtype=h.dtype))
        attn_mask = mx.expand_dims(attn_mask, 1)
        mx.eval(h, attn_mask)
        done += 1
        yield done, total

        for layer in layers:                                   # one vision layer per piece
            h = layer(h, patch_positions, attn_mask)
            mx.eval(h)
            done += 1
            yield done, total

        pooled, pool_mask = vt.pooler(h, patch_positions, padding_positions, output_length=output_length)
        valid_mask = pool_mask if pool_mask.shape[1] == output_length else ~pool_mask
        real = [pooled[i, :int(valid_mask[i].astype(mx.int32).sum().item())] for i in range(B)]
        feats = mx.concatenate(real, axis=0)[None]
        if vt.config.standardize:
            feats = (feats - vt.std_bias) * vt.std_scale
        feats = self.model.embed_vision(feats)
        mx.eval(feats)
        done += 1
        yield done, total

        emb = self.model.get_input_embeddings(input_ids=ids, pixel_values=pixel_values,
                                              cached_image_features=feats, **extra)
        mx.eval(emb.inputs_embeds, emb.per_layer_inputs)
        done += 1
        yield done, total

        cache = self.model.make_cache()
        if split == "layers":
            h = emb.inputs_embeds
            pli = emb.per_layer_inputs
            if tm.hidden_size_per_layer_input and pli is not None:
                pli = tm.project_per_layer_inputs(h, pli)
            masks = tm._make_masks(h, cache + [None] * (len(tm.layers) - len(cache)), extra.get("mm_token_type_ids"))
            for start in range(0, n_text, layers_per_piece):
                end = min(n_text, start + layers_per_piece)
                for idx in range(start, end):                  # layers < 24 own their KV: no sharing
                    p_in = pli[:, :, idx, :] if pli is not None else None
                    h, _, _ = tm.layers[idx](h, masks[idx], cache[idx], per_layer_input=p_in,
                                             shared_kv=None, offset=None)
                mx.eval(h, [c.state for c in cache[start:end]])
                done += 1
                yield done, total
            return Snapshot(frame_id=frame_id, captured_at=captured_at or t0, read_started=t0,
                            read_done=time.time(), budget=getattr(self, "budget", None),
                            prefix_len=n, master=cache)
        for i in range(0, n, chunk):                           # prefill chunks into one cache
            j = min(n, i + chunk)
            kw = {k: extra[k][:, i:j] for k in ("mm_token_type_ids", "token_type_ids") if extra.get(k) is not None}
            if self.skip_shared:
                kw["logits_to_keep"] = 1
            self.lm(inputs=None, inputs_embeds=emb.inputs_embeds[:, i:j],
                    per_layer_inputs=emb.per_layer_inputs[:, i:j], cache=cache, **kw)
            mx.eval([c.state for c in cache])
            done += 1
            yield done, total
        return Snapshot(frame_id=frame_id, captured_at=captured_at or t0, read_started=t0,
                        read_done=time.time(), budget=getattr(self, "budget", None),
                        prefix_len=n, master=cache)

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
    newest_at_submit: int = -1          # newest frame id the worker had seen when the question arrived
    ex_needed: int = 0                  # exchanges that must be in the context before answering
    decision: Optional[str] = None      # freshness decision: 'answer' | 'read_first'
    wait_for_frame: int = -1            # with 'read_first': answer once this frame has been read
    done: threading.Event = field(default_factory=threading.Event)
    result: dict = field(default_factory=dict)


@dataclass
class ReadJob:
    """A screen read in progress: a generator of pieces (GemmaAdapter.read_screen_steps)."""
    frame_id: int
    captured_at: float
    gen: Iterator = field(repr=False, default=None)
    done: int = 0
    total: int = 0
    gpu_s: float = 0.0
    started: float = 0.0              # perf_counter at start
    pauses: int = 0                   # answers that ran while this read was paused
    pinned: bool = False              # an answer waits for this read: never drop it for a newer frame
    kind: str = "read"                # "read" (screen) | "append" (a finished exchange) | "reanchor"
    image: object = field(repr=False, default=None)
    encoded: Optional[tuple] = field(repr=False, default=None)   # kept across restarts
    ex_mark: int = 0                  # exchanges completed when the job took its history
    note: Optional[str] = None        # open-windows line for the screen turn
    restarts: int = 0


def _whole_read(adapter, image, frame_id, captured_at):
    """One-piece fallback for adapters without read_screen_steps."""
    snap = adapter.read_screen(image, frame_id=frame_id, captured_at=captured_at)
    yield 1, 1
    return snap


POLICIES = ("p0", "p1", "p2")
FRESHNESS = ("A", "B", "C")


class GpuWorker(threading.Thread):
    """Owns MLX; this loop is the scheduler (ARCHITECTURE.md §5).

    Work, highest priority first: answer > spare cache copy > screen-read pieces.

    Policies:
      p0  MVP baseline: no read starts during a turn; once started, a read runs to the end.
      p1  reads may run during a turn; an answer waits for a read in progress to finish.
      p2  reads may run during a turn, in pieces; an answer goes first at the next piece
          boundary, under the freshness rule (when a newer frame than the last read exists):
            A  finish the read first if its remaining pieces take <= fresh_ms, else answer now (stale)
            B  always finish the read first
            C  always answer now from the last completed read
    Latest frame wins: in p1/p2 a newer frame drops a partial read, unless an answer waits for it.
    Load caps: a new read starts at most once per min_read_interval_s, and reads use at most
    duty_cap of GPU time over the last 60 s. A read an answer is waiting for ignores both caps.
    """

    def __init__(self, adapter: GemmaAdapter, tts: Tts, budget: int = 1120, warm_image=None,
                 first_chunk: str = "phrase", trim_silence: bool = True, mlx_cache_gb: float = 0.0,
                 policy: str = "p2", freshness: str = "B", fresh_ms: float = 300.0,
                 layers_per_piece: int = 2, min_read_interval_s: float = 0.0, duty_cap: float = 1.0,
                 history: bool = True, history_max_tokens: int = 2000, since_max_tokens: int = 1500):
        super().__init__(daemon=True, name="gpu-worker")
        self.adapter, self.tts, self.budget, self.warm_image = adapter, tts, budget, warm_image
        assert first_chunk in ("phrase", "sentence") and policy in POLICIES and freshness in FRESHNESS
        self.first_chunk, self.trim, self.mlx_cache_gb = first_chunk, trim_silence, mlx_cache_gb
        self.policy, self.freshness, self.fresh_ms = policy, freshness, fresh_ms
        self.layers_per_piece = layers_per_piece
        self.min_read_interval_s, self.duty_cap = min_read_interval_s, duty_cap
        self.cv = threading.Condition()
        self.pending_frame = None             # (frame_id, captured_at, image): latest wins
        self.latest_frame_id = -1
        self.read: Optional[ReadJob] = None
        self.answer_q: list[AnswerJob] = []
        self.turn_active = False
        self.snapshot: Optional[Snapshot] = None
        self.cancel = threading.Event()
        self.ready = threading.Event()
        self.startup: dict = {}
        self.stats = {"reads": 0, "frames_dropped": 0, "reads_dropped": 0, "answers_after_pause": 0}
        self.read_log: list[dict] = []
        self.error: Optional[BaseException] = None
        self._stop = False
        self._busy: list[tuple[float, float]] = []      # (end perf_counter, seconds) of read pieces
        self._calls: list = []                          # run_on_worker requests
        self._piece_ema: Optional[float] = None
        self._last_total = 32
        self._last_read_start = -1e9
        # conversation history (ARCHITECTURE.md §9): needs an adapter with the context pieces
        self.history = history and hasattr(adapter, "encode_steps")
        self.history_max_tokens, self.since_max_tokens = history_max_tokens, since_max_tokens
        self.ctx_old: list[tuple[str, str, int]] = []      # (q, a, tokens) before the current screen turn
        self.ctx_since: list[tuple[str, str, int]] = []    # exchanges after it, in the committed snapshot
        self.turn_q: list[tuple[str, str]] = []            # finished, not yet in the context
        self.prefix_cache = None                           # cache after <bos> + ctx_old (None if empty)
        self.ex_done = 0                                   # exchanges finished (monotonic)

    @property
    def reading(self) -> bool:
        return self.read is not None

    # -- called from other threads
    def submit_frame(self, frame_id: int, captured_at: float, image, note: Optional[str] = None):
        """note: short text about the screen (open windows, front to back), added to the screen turn."""
        with self.cv:
            if self.pending_frame is not None:
                self.stats["frames_dropped"] += 1
            self.pending_frame = (frame_id, captured_at, image)
            self.pending_note = note
            self.latest_frame_id = max(self.latest_frame_id, frame_id)
            self.cv.notify_all()

    def set_turn_active(self, active: bool):
        with self.cv:
            self.turn_active = active
            self.cv.notify_all()

    def submit_answer(self, job: AnswerJob) -> AnswerJob:
        with self.cv:
            job.read_in_flight_at_submit = self.read is not None
            job.newest_at_submit = self.latest_frame_id
            job.ex_needed = self.ex_done
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

    def run_on_worker(self, fn, timeout: float = 300.0):
        """Run fn() on the GPU worker thread (MLX ties queues to their thread); for tests and tools."""
        done = threading.Event()
        with self.cv:
            self._calls.append((fn, done))
            self.cv.notify_all()
        assert done.wait(timeout)

    def gpu_read_duty(self, window_s: float = 60.0) -> float:
        now = time.perf_counter()
        return sum(d for t, d in self._busy if t >= now - window_s) / window_s

    # -- worker thread
    def run(self):
        try:
            self._startup()
            self.ready.set()
            while True:
                with self.cv:
                    action = None
                    while action is None:
                        if self._stop:
                            return
                        action = self._next_action()
                        if action is None:
                            self.cv.wait(0.05)
                if action[0] == "call":
                    fn, done = action[1]
                    try:
                        fn()
                    finally:
                        done.set()
                elif action[0] == "answer":
                    self._answer(action[1])
                elif action[0] == "spare":
                    self._make_spare()
                else:
                    self._read_piece(run_to_end=action[1])
        except BaseException as e:  # surface to the app instead of dying silently
            self.error = e
            self.ready.set()
            raise

    def _next_action(self):
        """Called with self.cv held. Returns ("answer", job) | ("spare",) | ("piece", run_to_end) | None."""
        if self._calls:
            return ("call", self._calls.pop(0))
        if self.answer_q:
            job = self.answer_q[0]
            if job.decision is None:
                job.decision = self._decide()
                if job.decision == "read_first":       # the frame this answer waits for, and no newer
                    job.wait_for_frame = self._newest_unread()
            have = self.snapshot.frame_id if self.snapshot is not None else -1
            if job.decision == "read_first" and have < job.wait_for_frame:
                if self.read is None and self.pending_frame is not None:
                    self._start_read(urgent=True)
                if self.read is not None:
                    self.read.pinned = True
                    return ("piece", False)
            if self.history and self.snapshot is not None and self.snapshot.ex_included < job.ex_needed:
                # the conversation so far must be in the context: always wait (appends are short)
                if self.read is None and self.turn_q:
                    self._start_append()
                if self.read is not None:
                    self.read.pinned = True
                    return ("piece", False)
            self.answer_q.pop(0)
            return ("answer", job)
        if self._spare_needed():
            return ("spare",)
        if self.read is None and self.history and self.turn_q and self.snapshot is not None:
            self._start_append()                        # finished exchanges go into the context first
            return ("piece", False)
        if self.read is not None:
            if self.pending_frame is not None and not self.read.pinned and self.read.kind in ("read", "reanchor"):   # latest frame wins
                self.read.gen.close()
                self.read = None
                self.stats["reads_dropped"] += 1
            else:
                return ("piece", False)
        if self.pending_frame is not None and self._may_start_read():
            self._start_read()
            return ("piece", self.policy == "p0")       # p0: a read can't be interrupted
        if (self.history and self.snapshot is not None and self.snapshot.encoded is not None
                and sum(t for _, _, t in self.ctx_since) > self.since_max_tokens):
            self._start_read(reanchor=True)             # move a long tail of turns into the older history
            return ("piece", False)
        return None

    def _newest_unread(self) -> int:
        ids = [f for f in ((self.pending_frame or (None,))[0],
                           self.read.frame_id if self.read is not None else None) if f is not None]
        return max(ids) if ids else -1

    def _decide(self) -> str:
        """Freshness decision for the answer at the head of the queue: 'answer' or 'read_first'."""
        have = self.snapshot.frame_id if self.snapshot is not None else -1
        newer = [f for f in ((self.pending_frame or (None,))[0],
                             self.read.frame_id if self.read is not None else None) if f is not None]
        if not newer or max(newer) <= have:
            return "answer"
        if self.policy == "p0":
            return "answer"
        if self.policy == "p1":
            return "read_first" if self.read is not None else "answer"
        if self.snapshot is None or self.freshness == "B":
            return "read_first"
        if self.freshness == "C":
            return "answer"
        return "read_first" if self._remaining_s() * 1000 <= self.fresh_ms else "answer"

    def _remaining_s(self) -> float:
        per = self._piece_ema if self._piece_ema is not None else 0.07
        if self.read is not None and self.read.total:
            return (self.read.total - self.read.done) * per
        return self._last_total * per

    def _may_start_read(self) -> bool:
        if self.policy == "p0" and self.turn_active:
            return False
        if time.perf_counter() - self._last_read_start < self.min_read_interval_s:
            return False
        return self.gpu_read_duty() <= self.duty_cap

    def _start_read(self, urgent: bool = False, reanchor: bool = False):
        if reanchor:
            s = self.snapshot
            fid, cap, image = s.frame_id, s.captured_at, None
        else:
            fid, cap, image = self.pending_frame
            self.pending_frame = None
        if self.history:
            rj = ReadJob(fid, cap, started=time.perf_counter(), pinned=urgent,
                         kind="reanchor" if reanchor else "read", image=image,
                         encoded=self.snapshot.encoded if reanchor else None,
                         note=None if reanchor else getattr(self, "pending_note", None))
            rj.gen = self._context_read(rj)
            rj.total = self._read_total(rj)
        else:
            steps = getattr(self.adapter, "read_screen_steps", None)
            gen = (steps(image, frame_id=fid, captured_at=cap, layers_per_piece=self.layers_per_piece)
                   if steps else _whole_read(self.adapter, image, fid, cap))
            rj = ReadJob(fid, cap, gen=gen, started=time.perf_counter(), pinned=urgent)
        self.read = rj
        self._last_read_start = rj.started

    def _read_total(self, rj) -> int:
        est = getattr(self.adapter, "read_pieces_estimate", None)
        full = est(self.layers_per_piece) if est else self._last_total
        return full if rj.encoded is None else max(1, full - 19)       # encoded: skip prep + vision

    def _start_append(self):
        q, a = self.turn_q[0]
        base = self.snapshot
        rj = ReadJob(base.frame_id, base.captured_at, started=time.perf_counter(), kind="append")
        rj.gen = self._append_gen(base, q, a)
        rj.total = -(-24 // self.layers_per_piece)
        self.read = rj

    def _append_gen(self, base, q, a):
        ids = self.adapter.exchange_ids(q, a, first_after_screen=(base.tail == "screen"))
        cache = yield from self.adapter.prefill_text_steps(clone_cache(base.master), ids, self.layers_per_piece)
        snap = Snapshot(base.frame_id, base.captured_at, base.read_started, base.read_done, base.budget,
                        base.prefix_len + len(ids), master=cache, tail="turn",
                        ex_included=base.ex_included + 1, encoded=base.encoded)
        return ("append", snap, base, (q, a, len(ids)))

    def _context_read(self, rj):
        """A screen read with history: encode the image, extend the older history with the exchanges
        finished since the last screen, then append the screen turn. Generator; returns a commit tuple."""
        a = self.adapter
        if rj.encoded is None:
            rj.encoded = (yield from a.encode_steps(rj.image, note=rj.note)) if rj.note \
                else (yield from a.encode_steps(rj.image))
        with self.cv:
            rj.ex_mark = self.ex_done
            old = list(self.ctx_old)
            new = list(self.ctx_since) + [(q, ans, len(a.exchange_ids(q, ans))) for q, ans in self.turn_q]
            prefix = self.prefix_cache
        target = old + new
        dropped = False
        while target and sum(t for _, _, t in target) > self.history_max_tokens:
            target.pop(0)                          # oldest exchanges leave the context
            dropped = True
        pairs = [(q, ans) for q, ans, _ in target]
        if not target:
            prefix_new = None
        elif dropped or prefix is None:
            prefix_new = yield from a.new_prefix_steps(pairs, self.layers_per_piece)
        elif len(target) > len(old):
            prefix_new = yield from a.extend_prefix_steps(prefix, pairs[len(old):], self.layers_per_piece)
        else:
            prefix_new = prefix
        cache, n = yield from a.screen_steps(prefix_new, rj.encoded, self.layers_per_piece)
        snap = Snapshot(rj.frame_id, rj.captured_at, time.time(), time.time(), getattr(a, "budget", None),
                        n, master=cache, tail="screen", ex_included=rj.ex_mark, encoded=rj.encoded)
        return ("read", snap, prefix_new, target)

    def _read_piece(self, run_to_end: bool):
        rj = self.read
        while True:
            t0 = time.perf_counter()
            try:
                y = next(rj.gen)
                if isinstance(y, tuple):
                    rj.done, rj.total = y
                else:
                    rj.done += 1
                snap = None
            except StopIteration as e:
                snap = e.value
            t1 = time.perf_counter()
            if snap is None:
                dt = t1 - t0
                rj.gpu_s += dt
                self._busy.append((t1, dt))
                self._busy = [b for b in self._busy if b[0] >= t1 - 60]
                self._piece_ema = dt if self._piece_ema is None else 0.7 * self._piece_ema + 0.3 * dt
                if run_to_end:
                    continue
                return
            if isinstance(snap, tuple):               # history mode: commit, or restart if out of date
                self._commit(rj, snap)
                return
            with self.cv:
                self.snapshot = snap           # swap: answers use the newest completed read
                self.read = None
                self._last_total = rj.total or self._last_total
                self.stats["reads"] += 1
                self.read_log.append({"frame_id": rj.frame_id, "captured_at": rj.captured_at,
                                      "read_s": round(rj.gpu_s, 3), "wall_s": round(t1 - rj.started, 3),
                                      "pieces": rj.total, "pauses": rj.pauses, "prefix_len": snap.prefix_len})
                self.cv.notify_all()
            return

    def _commit(self, rj, result):
        kind = result[0]
        with self.cv:
            if kind == "append":
                _, snap, base, ex = result
                if self.snapshot is base:              # nothing else changed the context meanwhile
                    self.snapshot = snap
                    self.ctx_since.append(ex)
                    self.turn_q.pop(0)
                self.read = None
                self.stats["appends"] = self.stats.get("appends", 0) + 1
            else:
                _, snap, prefix_new, target = result
                if self.ex_done != rj.ex_mark:         # a turn finished during the read: redo from the prefix
                    rj.restarts += 1
                    self.stats["read_restarts"] = self.stats.get("read_restarts", 0) + 1
                    rj.gen = self._context_read(rj)
                    rj.done, rj.total = 0, self._read_total(rj)
                    return
                self.snapshot = snap
                self.prefix_cache = prefix_new
                self.ctx_old, self.ctx_since, self.turn_q = target, [], []
                self.read = None
                self.stats["reads"] += 1
                self.read_log.append({"frame_id": rj.frame_id, "captured_at": rj.captured_at, "kind": rj.kind,
                                      "read_s": round(rj.gpu_s, 3), "wall_s": round(time.perf_counter() - rj.started, 3),
                                      "pieces": rj.done, "pauses": rj.pauses, "restarts": rj.restarts,
                                      "prefix_len": snap.prefix_len, "history_exchanges": len(target)})
            self.cv.notify_all()

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
            steps = getattr(self.adapter, "read_screen_steps", None)
            gen = (steps(self.warm_image, layers_per_piece=self.layers_per_piece) if steps
                   else _whole_read(self.adapter, self.warm_image, 0, None))
            while True:
                try:
                    next(gen)
                except StopIteration as e:
                    snap = e.value
                    break
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

    def _answer(self, job: AnswerJob):
        r = job.result
        picked = time.perf_counter()
        r["wait_behind_read_s"] = round(picked - job.submitted, 4)
        r["read_in_flight_at_submit"] = job.read_in_flight_at_submit
        r["policy"], r["fresh_decision"] = self.policy, job.decision
        if self.read is not None:                       # a read is paused while this answer runs
            self.read.pauses += 1
            self.stats["answers_after_pause"] += 1
            r["read_paused"] = True
        snap = self.snapshot
        if snap is None:
            r["error"] = "no screen read yet"
            job.done.set()
            return
        r.update(frame_id=snap.frame_id, screen_captured_at=snap.captured_at,
                 screen_read_done=snap.read_done, budget=snap.budget,
                 stale_vs_submit=snap.frame_id < job.newest_at_submit)
        t0 = time.perf_counter()
        if snap.spare is not None:
            cache, snap.spare = snap.spare, None
        else:
            cache = clone_cache(snap.master)
        r["restore_s"] = round(time.perf_counter() - t0, 4)
        job.mark("restore_done")
        if self.history:
            ids = self.adapter.suffix_ids(job.question, tail=snap.tail)
            r["context_tokens"], r["history_exchanges"] = snap.prefix_len, snap.ex_included
        else:
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
        if self.history and text.strip():
            with self.cv:                               # the exchange goes into the context next
                self.turn_q.append((job.question, text.strip()))
                self.ex_done += 1
                self.cv.notify_all()
        job.mark("turn_complete")
        job.done.set()
