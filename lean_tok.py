"""lean_tok.py — Gemma's tokenizer without `tokenizers` / `transformers` in the app process.

The Hugging Face tokenizer for Gemma 4 (BPE, 262,144 pieces, 514,906 merges, byte fallback) held ~0.30 GB in the
app; deleting it freed only about half. The memory target (app + helpers ≤ 4.0 GB) asked for it to go. This is the
same algorithm on compact arrays (~20 MB): ids identical to the Hugging Face tokenizer, checked on every text the app
sees in the benchmark plus synthetic edge cases (tests/test_lean_tok.py).

The algorithm, as `tokenizer.json` defines it:
  - special (added) tokens are matched in the raw text first and map to their ids;
  - the rest: " " -> "▁" (the normalizer; the Split pre-tokenizer then has no spaces left to split on);
  - BPE over the whole text: lowest-ranked adjacent pair first (leftmost on ties), a character missing from the
    vocabulary falls back to its UTF-8 bytes as <0xNN> tokens.
  Every piece but one has the form "▁*word" (optional leading ▁ run, then no ▁), so no merge crosses the point where
  a ▁ run begins after other text; BPE can run word by word, with a cache. The exception, ">▁</", is kept whole.
Decoding: "▁" -> " ", runs of <0xNN> tokens -> their bytes as UTF-8 (one U+FFFD per byte when invalid, as the
ByteFallback decoder does), everything joined.

Build once (a separate process, so the app never loads the Hugging Face tokenizer):
  python -m lean_tok build [model dir]      -> <model dir>/lean_tokenizer.npz, <model dir>/lean_chat.json
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

FILE = "lean_tokenizer.npz"
CHAT = "lean_chat.json"               # the chat template's text around a user message (one image), see build_chat
NV = 1 << 18                          # ids < 262,144: a pair (a, b) packs into a * NV + b
_BYTE = re.compile(r"<0x([0-9A-F]{2})>")


def build(model_dir: Path) -> Path:
    """Compact arrays from tokenizer.json: pieces (one UTF-8 blob + offsets), merges sorted by packed pair."""
    import json
    t = json.loads((Path(model_dir) / "tokenizer.json").read_text())
    m = t["model"]
    assert m["type"] == "BPE" and m.get("byte_fallback") and not m.get("continuing_subword_prefix")
    assert t["normalizer"] == {"type": "Replace", "pattern": {"String": " "}, "content": "▁"}, t["normalizer"]
    vocab: dict = m["vocab"]
    n = max(vocab.values()) + 1
    assert n <= NV, n
    pieces = [""] * n
    for p, i in vocab.items():
        pieces[i] = p
    blobs = [p.encode("utf-8") for p in pieces]
    offsets = np.zeros(n + 1, np.int64)
    offsets[1:] = np.cumsum([len(b) for b in blobs])
    keys, vals = [], []
    for rank, mg in enumerate(m["merges"]):
        a, b = mg if isinstance(mg, list) else mg.split(" ", 1)
        keys.append(vocab[a] * NV + vocab[b])
        vals.append(rank * NV + vocab[a + b])          # rank and result packed: lower rank merges first
    keys, vals = np.array(keys, np.int64), np.array(vals, np.int64)
    order = np.argsort(keys)
    special = [(a["content"], a["id"]) for a in t["added_tokens"]]
    out = Path(model_dir) / FILE
    np.savez(out, blob=np.frombuffer(b"".join(blobs), np.uint8), offsets=offsets, keys=keys[order], vals=vals[order],
             special=np.array([s for s, _ in special]), special_ids=np.array([i for _, i in special], np.int64))
    return out


def build_chat(model_dir: Path) -> Path:
    """The chat template (one image, one user message, generation prompt) is head + message.strip() + tail: rendered
    once here with the real processor, checked on several messages, saved. The app then never needs the template."""
    import json
    from mlx_vlm import load
    from mlx_vlm.prompt_utils import apply_chat_template
    model, proc = load(str(model_dir), lazy=True)
    mark = "\ue000"
    r = apply_chat_template(proc, model.config, mark, num_images=1)
    head, tail = r.split(mark)
    for msg in ("what is the rent", "  spaced  ", "", "a\nb", "Q: <b>x</b> 🍩", "line\n\n"):
        want = apply_chat_template(proc, model.config, msg, num_images=1)
        assert want == head + msg.strip() + tail, (msg, want)
    out = Path(model_dir) / CHAT
    out.write_text(json.dumps({"head": head, "tail": tail}))
    return out


class LeanTokenizer:
    """encode / decode with the Hugging Face tokenizer's interface as the app uses it."""

    def __init__(self, model_dir: Path, cache_words: int = 50000):
        z = np.load(Path(model_dir) / FILE)
        blob, off = z["blob"].tobytes(), z["offsets"]
        self.pieces = [blob[off[i]:off[i + 1]].decode("utf-8") for i in range(len(off) - 1)]
        self._keys, self._vals = z["keys"], z["vals"]
        self.special = {s: int(i) for s, i in zip(z["special"].tolist(), z["special_ids"].tolist())}
        self._char = {p: i for i, p in enumerate(self.pieces) if len(p) == 1}
        byte_id = {int(m.group(1), 16): i for i, p in enumerate(self.pieces) if p.startswith("<0x") and (m := _BYTE.fullmatch(p))}
        self._byte_ids = [byte_id[b] for b in range(256)]
        self._byte_of = {tid: b for b, tid in enumerate(self._byte_ids)}
        self._special_re = re.compile("|".join(re.escape(s) for s in sorted(self.special, key=len, reverse=True)))
        self._words: dict[str, tuple] = {}
        self._cache_words = cache_words
        self.bos_token_id, self.eos_token_id = self.special["<bos>"], self.special["<eos>"]
        self.pad_token_id = self.special.get("<pad>")

    # ---------------------------------------------------------------- encode
    def _merge(self, ids: list[int]) -> tuple:
        """BPE on one word: repeatedly merge the lowest-ranked adjacent pair (leftmost on ties)."""
        keys, vals = self._keys, self._vals
        while len(ids) > 1:
            pairs = np.array(ids[:-1], np.int64) * NV + np.array(ids[1:], np.int64)
            pos = np.searchsorted(keys, pairs)
            pos[pos >= len(keys)] = 0
            hit = keys[pos] == pairs
            if not hit.any():
                break
            ranks = np.where(hit, vals[pos] // NV, np.iinfo(np.int64).max)
            j = int(np.argmin(ranks))                  # argmin takes the first: leftmost on ties
            merged = int(vals[pos[j]] % NV)
            ids = ids[:j] + [merged] + ids[j + 2:]
        return tuple(ids)

    def _word(self, w: str) -> tuple:
        out = self._words.get(w)
        if out is None:
            ids = []
            for ch in w:
                i = self._char.get(ch)
                if i is None:
                    ids.extend(self._byte_ids[b] for b in ch.encode("utf-8"))
                else:
                    ids.append(i)
            out = self._merge(ids)
            if len(self._words) >= self._cache_words:
                self._words.clear()                    # bounded: memory doesn't grow with use
            self._words[w] = out
        return out

    @staticmethod
    def _split(s: str) -> list[str]:
        """Words: a ▁ run starts a new word, except inside ">▁</" (the one piece that spans a ▁)."""
        cuts = [m.start() for m in re.finditer(r"(?<=[^▁])▁", s) if not (s[m.start() - 1] == ">" and s.startswith("▁</", m.start()))]
        bounds = [0] + cuts + [len(s)]
        return [s[a:b] for a, b in zip(bounds, bounds[1:]) if b > a]

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        out: list[int] = []
        pos = 0
        for m in self._special_re.finditer(text):
            out += self._encode_plain(text[pos:m.start()])
            out.append(self.special[m.group()])
            pos = m.end()
        out += self._encode_plain(text[pos:])
        return out

    def _encode_plain(self, text: str) -> list[int]:
        if not text:
            return []
        out: list[int] = []
        for w in self._split(text.replace(" ", "▁")):
            out.extend(self._word(w))
        return out

    def __call__(self, text, **kw):
        return {"input_ids": [self.encode(t) for t in text] if isinstance(text, list) else self.encode(text)}

    # ---------------------------------------------------------------- decode
    def decode(self, ids: Iterable[int], skip_special_tokens: bool = False) -> str:
        parts: list[str] = []
        run: list[int] = []

        def flush():
            if run:
                b = bytes(run)
                try:
                    parts.append(b.decode("utf-8"))
                except UnicodeDecodeError:
                    parts.append("�" * len(b))
                run.clear()

        specials = set(self.special.values())
        for i in ids:
            i = int(i)
            if i in self._byte_of:
                run.append(self._byte_of[i])
                continue
            flush()
            if skip_special_tokens and i in specials:
                continue
            parts.append(self.pieces[i].replace("▁", " "))
        flush()
        return "".join(parts)

    # ---------------------------------------------------------------- lookups
    def get_vocab(self) -> dict:
        return {**{p: i for i, p in enumerate(self.pieces) if p.startswith("<")}, **self.special}

    def convert_tokens_to_ids(self, token: str) -> Optional[int]:
        if token in self.special:
            return self.special[token]
        ids = self.encode(token)
        return ids[0] if len(ids) == 1 else None


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "build":
        root = Path(__file__).resolve().parent
        d = Path(sys.argv[2]) if len(sys.argv) > 2 else root / "models/gemma-4-e4b-it-4bit"
        print(build(d))
        print(build_chat(d))
    else:
        print(__doc__)
