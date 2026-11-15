"""The app without the Hugging Face tokenizer and processor (lean_tok.py, M3 2026-11-13) gives exactly what they gave:
token ids and decoded text, the rendered prompts, and a screen's model inputs. Needs the model files (and builds
lean_tokenizer.npz / lean_chat.json if missing); loads weights lazily, nothing runs on the GPU."""
import glob
import json
import random
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models/gemma-4-e4b-it-4bit"
pytestmark = pytest.mark.skipif(not (MODEL / "config.json").exists(), reason="model files not downloaded")


@pytest.fixture(scope="module")
def pair():
    """(lean adapter, processor adapter), weights lazy."""
    from engine import RECALL_INSTRUCTION, GemmaAdapter
    lean = GemmaAdapter(instruction=RECALL_INSTRUCTION, lean=True)
    assert lean._lean_files()
    old = GemmaAdapter(instruction=RECALL_INSTRUCTION, lean=False)
    from engine import _audio_config_off
    from mlx_vlm import load
    from mlx_vlm.models.gemma4.processing_gemma4 import Gemma4ImageProcessor
    from lean_tok import LeanTokenizer
    with _audio_config_off():
        model, proc = load(str(MODEL), lazy=True)
    old.model, old.processor, old.tok, old.image_processor = model, proc, proc.tokenizer, proc.image_processor
    lean.model, lean.tok, lean.image_processor = model, LeanTokenizer(MODEL), Gemma4ImageProcessor.from_pretrained(str(MODEL))
    lean.chat = json.loads((MODEL / "lean_chat.json").read_text())
    for a in (lean, old):
        a.set_budget(1120)
    return lean, old


def corpus():
    texts = []
    for f in glob.glob(str(ROOT / "eval/recall/sessions/*/session.json")):
        s = json.loads(Path(f).read_text())
        texts += [p.get("text", "") for p in s["pages"]] + [p.get("title", "") for p in s["pages"]]
        texts += [q.get("question", "") for q in s.get("probes", [])]
    texts += (ROOT / "ARCHITECTURE.md").read_text().split("\n\n")[:400]
    texts += ["a  b   c", "  lead", "trail  ", "\n\n\n", "x > </y>", "<div> </div>", "emoji 🍩🤖", "日本語", "naïve",
              "​", "<0x41>", "a\r\nb", "\x00\x07", "▁x▁", "$3,488/mo", "<|turn>user\nhi<turn|>"]
    rnd = random.Random(0)
    alphabet = list("ab ▁<>/\n\t.,é🍩日😀07") + ["<|turn>", "<turn|>", "<bos>", ">▁</", "> </"]
    texts += ["".join(rnd.choice(alphabet) for _ in range(rnd.randint(1, 50))) for _ in range(1000)]
    return [t for t in texts if t]


def test_ids_and_text_are_identical(pair):
    lean, old = pair
    for t in corpus():
        ids = old.tok.encode(t, add_special_tokens=False)
        assert lean.tok.encode(t) == ids, t[:60]
        for cut in (len(ids), max(1, len(ids) // 2)):
            assert lean.tok.decode(ids[:cut]) == old.tok.decode(ids[:cut]), t[:60]
    rnd = random.Random(1)
    for _ in range(500):
        ids = [rnd.randrange(262144) for _ in range(rnd.randint(1, 15))]
        assert lean.tok.decode(ids) == old.tok.decode(ids)
    assert (lean.tok.bos_token_id, lean.tok.eos_token_id) == (old.tok.bos_token_id, old.tok.eos_token_id)


def test_prompts_are_identical(pair):
    lean, old = pair
    for q in ("what was the rent?", "  spaced  ", "", "a\nb", "<b>x</b> 🍩"):
        assert lean.full_prompt(q) == old.full_prompt(q)
        assert lean.suffix_ids(q) == old.suffix_ids(q)
    assert lean.screen_turn_prompt() == old.screen_turn_prompt()
    assert lean.screen_turn_prompt("Open windows: Chrome") == old.screen_turn_prompt("Open windows: Chrome")


def test_screen_inputs_are_identical(pair):
    lean, old = pair
    pngs = sorted(glob.glob(str(ROOT / "eval/recall/sessions/h1/png/*.png")))[:2]
    if not pngs:
        pytest.skip("benchmark screens not made")
    imgs = [Image.open(p).convert("RGB") for p in pngs]
    imgs += [imgs[0].resize((1512, 982)), imgs[0].resize((2560, 1600))]
    for img in imgs:
        for prompt in (lean.screen_turn_prompt(), "<|turn>user\n(Earlier screen) 9:41 AM\n<|image|><turn|>\n"):
            a, b = lean._inputs(img, prompt), old._inputs(img, prompt)
            assert sorted(a) == sorted(b)
            for k in a:
                assert np.asarray(a[k]).dtype == np.asarray(b[k]).dtype and np.array_equal(np.asarray(a[k]), np.asarray(b[k])), k
