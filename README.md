# local-voice

**A voice assistant that remembers your screen.** Hold a key and ask, *"how much was that kettle I was looking at?"*, about a tab you closed 40 minutes ago. It answers out loud in about half a second, fully offline, on a base 16 GB MacBook.

<!-- Demo GIF goes here (menu bar captions, closed-tab recall). -->

Three models run on one laptop, in about 4 GB:
- a vision-language model (Gemma 4 E4B) on the GPU;
- speech recognition (Parakeet) on the Neural Engine;
- speech synthesis (Kokoro) on the Neural Engine.

The assistant keeps the last hour of your screen as the model's own internal state, not as screenshots.

| Measured end to end (key release → first audio) | |
|---|---|
| Recall questions about closed pages | **492 ms** median |
| Questions about the current screen, with an hour of memory loaded | **477 ms** median |
| Accuracy, precise / vague / current-screen questions | 15/16 · 9/12 · 19/20 |
| Peak memory, all processes | **3.96 GB** |

*Synthetic hour-long session h1, replayed through the real app (spoken questions, screen changes mid-question), 4× speed, plugged in, M5 MacBook Pro.*

## How it gets there

**The work happens while you're still talking.** Partial transcripts arrive every ~0.3 s. Each one drives retrieval and builds the context, and the question is prefilled as the words come in. By the time you let go, the context was ready in 56 of 56 turns. A single scheduler runs every GPU job in pieces of a few milliseconds that can be paused: screen reads, retrieval, cache builds, prefill and decoding.

**Precomputed KV caches, spliced in.** Each remembered screen's KV cache block (~1,400 tokens) is computed in idle time and stored on SSD. At question time it is loaded and its keys are re-rotated to their new position. Only the 4 full-attention layers are stored, because the sliding-window layers can never reach that far back. Recall latency halved, 968 → 486 ms (the second run was also plugged in), at a cost of 1 answer in 112 against exact computation.

**Retrieval through the model's own attention.** Each kept screen's image-token keys at two attention layers are stored as int8. The spoken question's queries score every screen against them, calibrated against a content-free query and fused with text search over the screen's OCR. That finds screens that are mostly images, where text search has nothing to match.

**Custom Metal kernels and a memory budget.**
- A quantized-matmul kernel indexes Gemma's quantization scales through a small per-tensor table: 143 MB saved, outputs bit-identical. A 2-row variant is used for 2-token steps.
- A fused vision kernel speeds up the image encoder by 16%.
- Speech synthesis moved to the Neural Engine, cutting peak memory from 5.56 to 4.95 GB.
- Everything that grows with history lives on disk, memory-mapped.
- Bigger Metal command buffers cut the first token from 45 to 35 ms.

## Benchmark

The repo includes a benchmark built for this (`eval/recall/`). Headless Chrome renders hour-long browsing sessions with invented shops, flights, hotels, docs and issue trackers, so the model can't answer from what it already knows. That's 144 questions over 4 sessions, asked after every tab is closed. The pages are hard on purpose: look-alike values, 12 px fine print, facts below the fold, pages on screen for under 15 seconds, and questions about things that were never shown. Claude Opus scores 144/144 when given every screen of the hour, which shows the questions are answerable.

Every design decision was measured against a bar written down before the run. All of them are in [ARCHITECTURE.md](ARCHITECTURE.md), including the ones that failed, such as splitting the cache into many separate pages (fast, but recall fell from 94% to 29%) and speculative decoding (nothing to copy from an image).

**Open problems.** Asked about something it never saw, the model still makes up a look-alike answer in 4 of 8 cases on h1. The numbers above come from development runs. The final run (real time, all four sessions, on battery, beside a browser, editor and call) hasn't been done.

## Run it

macOS on Apple silicon, Python 3.13.

```sh
./setup.sh                    # environment, Swift helpers, models (~6 GB)
.venv/bin/python menubar.py   # menu bar app with live captions; hold Right Option to talk
```

Reproduce: `eval/recall/make_sessions.py --sessions h1,h2,h3,h4`, then `eval/replay_recall.py --session h1 --speed 4`.
