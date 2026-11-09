#!/bin/zsh
# One-step setup for local-voice on an Apple silicon Mac (built and measured on a base M5, 16 GB, macOS 26).
#
#   ./setup.sh            # Python env, Swift helpers, default models (~6 GB download the first time)
#   ./setup.sh --extras   # also the optional pieces: the speculative-decoding drafter, WhisperKit, the
#                         # Neural Engine image encoder (converted locally, takes a while)
#
# Safe to re-run: every step skips what is already there. Everything the app runs offline afterwards;
# downloads happen only here. Model revisions are pinned to the ones the results were measured with.
set -e
cd "${0:A:h}"
EXTRAS=0; [[ "$1" == "--extras" ]] && EXTRAS=1
step() { print -P "%B== $*%b" }

[[ "$(uname -s)-$(uname -m)" == "Darwin-arm64" ]] || { echo "Needs macOS on Apple silicon."; exit 1; }

step "Python 3.13 environment (.venv, requirements.txt pinned)"
PY313=$(command -v python3.13 || true)
[[ -n "$PY313" ]] || { echo "Python 3.13 not found. Install it (e.g. 'brew install python@3.13') and re-run."; exit 1; }
[[ -x .venv/bin/python ]] || "$PY313" -m venv .venv
.venv/bin/python -m pip install -q --upgrade pip
.venv/bin/python -m pip install -q -r requirements.txt

step "Swift helpers (window list, push-to-talk key, speech recognition server)"
command -v swiftc >/dev/null || { echo "Swift not found: run 'xcode-select --install' and re-run."; exit 1; }
[[ -x tools/winlist ]] || swiftc -O tools/winlist.swift -o tools/winlist
[[ -x tools/hotkey ]] || swiftc -O tools/hotkey.swift -o tools/hotkey
[[ -x tools/parakeet/.build/release/parakeet-serve ]] || (cd tools/parakeet && swift build -c release)

step "Models (pinned Hugging Face revisions)"
HF=.venv/bin/hf
fetch() {   # fetch <repo> <revision> <dir>
  if [[ -f "$3/config.json" ]]; then echo "  $3: present"; return; fi
  "$HF" download "$1" --revision "$2" --local-dir "$3"
}
fetch mlx-community/gemma-4-e4b-it-4bit 475b9088d29754a3379866cf5aeb6b41acd313c2 models/gemma-4-e4b-it-4bit
fetch mlx-community/Kokoro-82M-bf16 a71e4d38b236d968966a2002c4c895dbd12b1c3c models/kokoro-82m-bf16
if [[ -d models/parakeet-tdt-0.6b-v2/Encoder.mlmodelc ]]; then
  echo "  models/parakeet-tdt-0.6b-v2: present"
else   # Core ML bundles via FluidAudio 0.18.2 (pinned in tools/parakeet/Package.swift)
  tools/parakeet/.build/release/parakeet-serve --models models/parakeet-tdt-0.6b-v2 --download
fi

if (( EXTRAS )); then
  step "Extras"
  fetch mlx-community/gemma-4-E4B-it-assistant-bf16 844e008e06ef5562bdb89428d851d0634d119dcd models/gemma-4-e4b-it-assistant-bf16
  if [[ ! -d models/whisperkit-coreml/openai_whisper-large-v3-v20240930_turbo_632MB ]]; then
    "$HF" download argmaxinc/whisperkit-coreml --revision 0f63a7800b00dd0226abd051b906c246e1907482 \
      --include "openai_whisper-large-v3-v20240930_turbo_632MB/*" --local-dir models/whisperkit-coreml
    # WhisperKit looks for its tokenizer under <dir>/models/openai/whisper-large-v3
    "$HF" download openai/whisper-large-v3 --revision 06f233fe06e710322aca913c1bc4249a0d71fce1 --include "tokenizer.json" "tokenizer_config.json" "config.json" \
      --local-dir models/whisper-tokenizer/models/openai/whisper-large-v3
  fi
  [[ -d models/vision-coreml ]] || .venv/bin/python ane_vision.py convert
fi

step "Check (model-free tests)"
.venv/bin/python -m pytest -q tests --ignore=tests/test_ane_vision.py --ignore=tests/test_engine.py --ignore=tests/test_vision_qkv.py

cat <<'MSG'

Done. Before the first run, allow the terminal app in System Settings > Privacy & Security:
  Screen & System Audio Recording (reading the screen), Microphone (your question),
  Input Monitoring (hold Right Option anywhere to talk; optional, Enter works too).
Then:  .venv/bin/python app.py
Benchmark screens (optional, needs Google Chrome): .venv/bin/python eval/recall/make_sessions.py --sessions h1,h2,h3,h4
MSG
