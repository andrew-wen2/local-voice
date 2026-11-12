"""An attribute set on `self` must not hide a method of the same class: `self.spec = None` in RecallBackend.__init__
hid its spec() method and crashed the first live run that paused before key up (2026-11-11)."""
import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FILES = ["recall_backend.py", "recall_worker.py", "scheduler.py", "screen_store.py", "recall.py", "engine.py",
         "kv_reuse.py", "asr.py", "audio_io.py", "app.py"]


@pytest.mark.parametrize("name", FILES)
def test_no_attribute_hides_a_method(name):
    tree = ast.parse((ROOT / name).read_text())
    clashes = []
    for cls in (n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)):
        methods = {n.name for n in cls.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
        for node in ast.walk(cls):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(node, ast.AnnAssign) else []
            for t in targets:
                for el in (t.elts if isinstance(t, ast.Tuple) else [t]):
                    if (isinstance(el, ast.Attribute) and isinstance(el.value, ast.Name) and el.value.id == "self"
                            and el.attr in methods):
                        clashes.append(f"{cls.name}.{el.attr}")
    assert not clashes, f"attributes hiding methods: {sorted(set(clashes))}"
