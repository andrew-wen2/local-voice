"""A kept screen's token layout without running the processor per screen: the image's token block is
cached per size, the text tokenized on its own; it must equal the processor's output exactly. Needs the model files
(the processor only; weights load lazily, nothing runs on the GPU)."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

MODEL = Path(__file__).resolve().parents[1] / "models/gemma-4-e4b-it-4bit"
pytestmark = pytest.mark.skipif(not (MODEL / "config.json").exists(), reason="model files not downloaded")


def test_cached_layout_equals_the_processor():
    from mlx_vlm import load
    from mlx_vlm.utils import prepare_inputs

    import recall_backend as RB
    from engine import RECALL_INSTRUCTION, TURN_END, TURN_USER, GemmaAdapter
    model, proc = load(str(MODEL), lazy=True)
    proc.image_processor.max_soft_tokens = 1120
    a = SimpleNamespace(budget=1120, _inputs=lambda img, p: prepare_inputs(proc, images=[img], prompts=[p]),
                        text_ids=lambda s: proc.tokenizer.encode(s, add_special_tokens=False))
    be = object.__new__(RB.RecallBackend)
    be.a, be._tpl = a, {}
    g = GemmaAdapter.__new__(GemmaAdapter)
    g.instruction, g.processor, g.tok, g.model = RECALL_INSTRUCTION, proc, proc.tokenizer, model
    titles = ["Some page title", "Re: Vendor demo — Inbox (3) · Gmail", "", "日本語のページ 🍩", "x\n\n  y"]
    texts = [f"{TURN_USER}(Earlier screen) 9:41 AM {t}\n<|image|>{TURN_END}" for t in titles]
    texts += [g.screen_turn_prompt(), g.screen_turn_prompt(note="Open windows: Chrome — Gmail; Slack")]
    for size in [(3024, 1964), (1512, 982), (2560, 1600)]:
        for t in texts:
            ref = prepare_inputs(proc, images=[Image.new("RGB", size)], prompts=[t])
            got = be._inputs_like(size, t)
            for k in ("input_ids", "mm_token_type_ids", "attention_mask"):
                assert np.array_equal(np.asarray(ref[k]), np.asarray(got[k])), (size, k, t[:40])
    assert len(be._tpl) == 3                            # one processor run per size
