#!/bin/zsh
# One-step setup for local-voice on an Apple silicon Mac (built and measured on a base M5, 16 GB, macOS 26).
#
#   ./setup.sh            # Python env, Swift helpers, models (~6 GB download the first time)
#
# Safe to re-run: every step skips what is already there. Everything the app runs offline afterwards;
# downloads happen only here. Model revisions are pinned to the ones the results were measured with.
set -e
cd "${0:A:h}"
step() { print -P "%B== $*%b" }

[[ "$(uname -s)-$(uname -m)" == "Darwin-arm64" ]] || { echo "Needs macOS on Apple silicon."; exit 1; }

step "Python 3.13 environment (.venv, requirements.txt pinned)"
PY313=$(command -v python3.13 || true)
[[ -n "$PY313" ]] || { echo "Python 3.13 not found. Install it (e.g. 'brew install python@3.13') and re-run."; exit 1; }
[[ -x .venv/bin/python ]] || "$PY313" -m venv .venv
.venv/bin/python -m pip install -q --upgrade pip
.venv/bin/python -m pip install -q -r requirements.txt

step "Swift helpers (window list, push-to-talk key, speech recognition and speech output servers)"
command -v swiftc >/dev/null || { echo "Swift not found: run 'xcode-select --install' and re-run."; exit 1; }
[[ -x tools/winlist ]] || swiftc -O tools/winlist.swift -o tools/winlist
[[ -x tools/hotkey ]] || swiftc -O tools/hotkey.swift -o tools/hotkey
[[ -x tools/parakeet/.build/release/parakeet-serve && -x tools/parakeet/.build/release/kokoro-serve ]] || \
  (cd tools/parakeet && swift build -c release)

step "Models (pinned Hugging Face revisions)"
HF=.venv/bin/hf
fetch() {   # fetch <repo> <revision> <dir>
  if [[ -f "$3/config.json" ]]; then echo "  $3: present"; return; fi
  "$HF" download "$1" --revision "$2" --local-dir "$3"
}
fetch mlx-community/gemma-4-e4b-it-4bit 475b9088d29754a3379866cf5aeb6b41acd313c2 models/gemma-4-e4b-it-4bit
if [[ -d models/parakeet-tdt-0.6b-v2/Encoder.mlmodelc ]]; then
  echo "  models/parakeet-tdt-0.6b-v2: present"
else   # Core ML bundles via FluidAudio 0.18.2 (pinned in tools/parakeet/Package.swift)
  tools/parakeet/.build/release/parakeet-serve --models models/parakeet-tdt-0.6b-v2 --download
fi
if [[ ! -f models/gemma-4-e4b-it-4bit/indexed_scales.safetensors ]]; then   # lossless scale index, ~3 s
  .venv/bin/python -m kernels.indexed_qmv build
fi
if [[ ! -f models/gemma-4-e4b-it-4bit/lean_tokenizer.npz || ! -f models/gemma-4-e4b-it-4bit/lean_chat.json ]]; then
  .venv/bin/python -m lean_tok build      # the app's own tokenizer and prompt template (no Hugging Face tokenizer in memory)
fi
if [[ -d models/kokoro-coreml/kokoro-82m-coreml ]]; then
  echo "  models/kokoro-coreml: present"
else   # Kokoro on the Neural Engine (the default speech output), via FluidAudio 0.18.2
  tools/parakeet/.build/release/kokoro-serve --models models/kokoro-coreml --download
fi

step "Check (model-free tests)"
.venv/bin/python -m pytest -q tests --ignore=tests/test_engine.py --ignore=tests/test_vision_qkv.py

cat <<'MSG'

Done. Before the first run, allow the terminal app in System Settings > Privacy & Security:
  Screen & System Audio Recording (reading the screen), Microphone (your question),
  Input Monitoring (hold Right Option anywhere to talk; optional, Enter works too).
Then:  .venv/bin/python menubar.py   (menu bar icon: Turn On / Turn Off; Off wipes what it saw)
   or:  .venv/bin/python app.py       (same assistant, run in the terminal)
MSG
