# local-voice

A push-to-talk voice assistant for the Mac that remembers what was on your screen for the past hour, including tabs you've closed, and answers out loud in about half a second. Everything runs offline on a base 16 GB MacBook.

> *"How much was that electric kettle I was looking at?"*: you closed the tab 40 minutes ago, and it answers from what it saw.

<!-- Demo GIF goes here (menu bar captions, closed-tab recall). -->

Hold Right Option in any app, ask, let go. The assistant reads the current screen, and it can also look back over everything it kept from the last hour. Its context holds a timeline of the pages you visited and the one earlier screen that best matches your question. It talks about the screen only when the question is about it, and it keeps answers to one to three short sentences.

## Results

All numbers were measured on one machine: a base M5 MacBook Pro (16 GB). The benchmark is synthetic (see [Benchmark](#benchmark)). "First audio" runs from releasing the talk key to the first audible sample handed to the output device. Speech recognition, retrieval, prefill, decoding and speech synthesis are all included. The device's own output delay is not.

**End-to-end replay through the real app** (`eval/replay_recall.py`): session h1, with the hour played at 4× speed and plugged in. The questions are spoken audio, and every 4th one has a screen change one second before the key comes up.

| | Result |
|---|---|
| Recall questions (36), first audio | **492 ms median** (p95 941 ms) |
| Current-screen reading questions (20), first audio, with an hour of memory loaded | **477 ms median** |
| Precise questions (name the page) | 15 / 16 |
| Vague questions (only describe it) | 9 / 12 |
| Reading questions | 19 / 20 |
| Never-shown questions answered with a made-up value | 4 / 8 |
| Peak memory, app + helper processes | 3.96 GB |

**How first audio came down**, on the same h1 replay at 4×:

| Step | Recall median | Recall correct |
|---|---|---|
| Before the latency work (on battery, under heavy swap) | 1,333 ms | 27 / 36 |
| Lossless scheduling and retrieval fixes (same outputs; on battery) | 968 ms | 27 / 36 |
| Stored KV blocks for kept screens (plugged in, so power also changed) | 486 ms | 28 / 36 |
| Question prefill kept when the last words change (plugged in) | 492 ms (p95 1,024 → 941 ms) | 28 / 36 |

**Stored KV blocks against exact computation**, offline on all four hard sessions (`eval/kernels/stored_blocks.py`, paired on identical contexts):

| | Exact | Stored blocks |
|---|---|---|
| Recall questions correct | 67 / 112 | 66 / 112 |
| Made-up answers on never-shown questions | 7 / 32 | 8 / 32 |
| Reading questions | 19 / 20 | 19 / 20 |

**What these numbers are not.** They come from development runs: one session, 4× speed, plugged in, nothing else running. The final measuring run hasn't been done: real time, all four sessions, on battery, beside a browser, editor and call. Peak memory ranges from 3.96 to 4.4 GB across runs, against a 4.0 GB target. Made-up answers are the main open problem: asked about something it never saw, the model often answers with a look-alike value from a similar page.

## How it works

```
 screen ──► ScreenCaptureKit ──► OCR (Apple Vision, in quiet time) ──► page store: visits, timeline, text
                                   │
                                   └─ keep rule ──► kept screens: Gemma vision features (int8, on disk)
                                                    + retrieval keys from Gemma's own attention (int8, mmap)
                                                    + a stored KV block per screen, built in quiet time

 hold key ──► mic ──► Parakeet (Neural Engine): partial transcripts every ~0.3 s
                         │
                         ├─ retrieval per partial: frame-text BM25 fused with Gemma key scores ─► one kept screen
                         ├─ its stored KV block spliced into the context (keys re-rotated to the new position)
                         └─ question prefilled as the words arrive
 release ──► final transcript (~50 ms) ──► decode (Gemma 4 E4B, 4-bit, MLX on the GPU)
                                            ──► first phrase ──► Kokoro (Neural Engine) ──► speaker
```

**Models:** Gemma 4 E4B (4-bit, MLX) reads screens and answers. Parakeet TDT 0.6B v2 does speech recognition and Kokoro 82M does speech, both on the Neural Engine through small Swift helpers built on FluidAudio.

**Context for a recall question:** the hour's timeline (one line per page visit: time, dwell, title), the conversation so far, one kept screen (its image tokens plus its OCR text), the current screen, then the question.

**Finding the right screen.** Each kept screen is run through Gemma's text layers once, in quiet time. The image tokens' keys at two full-attention layers (11 and 17) are kept as int8. At question time, the question's queries score every kept screen (max over image tokens, mean over heads), calibrated against a content-free query. The result is fused with a BM25 rank over the screen's OCR text. Text matching misses screens that are mostly visual, and the keys catch many of them.

**Answers with almost no prefill.** A kept screen's block (screen turn, memory header, OCR text, ~1,380 tokens) is computed once, after the context prefix as it was then, and stored on disk. Only the 4 full-attention layers are kept. Gemma's 20 sliding-window layers never see that far back, because the current screen sits between the block and the question. At question time the block is loaded and its keys are re-rotated to their new position. The offline paired check above shows this costs about one answer in 112 against computing it exactly.

**One GPU thread, one scheduler** (`scheduler.py`, `recall_worker.py`). MLX ties GPU queues to the thread that created them, so a single loop owns the GPU. Every job runs in pieces of a few milliseconds, so it can be paused: screen reads, retrieval passes, page and block builds, question prefill, decoding. While you speak, the scheduler builds the context the latest partial transcript points to. After you release the key, the answer goes first.

**Systems work, all exact (same outputs) unless noted:**
- **Custom Metal kernels** (`kernels/`):
  - Indexed quantization scales: each 4-bit tensor uses only 928–5,179 distinct (scale, bias) pairs, so a 16-bit index into a per-tensor table replaces them. That saves 143 MB, bit-identical.
  - A 2-row variant of the same kernel, for 2-token steps. Each row is bit-identical to the one-token kernel.
  - A fused q/k/v norm + 2D RoPE pass in the vision encoder: 812 → 684 ms per encoder pass. It computes in fp32, so it is closer to fp32 than MLX's bf16 path.
- **Memory:**
  - Kokoro moved from MLX on the GPU to the Neural Engine: peak 5.56 → 4.95 GB.
  - The per-layer embedding table (1.59 GB) stays on disk, memory-mapped.
  - The image encoder and output layer take turns in memory (−0.33 GB).
  - A lean tokenizer replaces the Hugging Face one.
  - Everything that grows with history lives on disk: features, keys and blocks.
  - Two leaks fixed: OCR (~25 MB a frame) and cache slices that kept their parent buffers alive.
- **Latency:**
  - Bigger Metal command buffers: first token 45 → 35 ms.
  - Pipelined decoding, with the next token queued before this token's CPU work: 40 → 28 ms a token in a standalone test.
  - Retrieval and prefill run as interruptible pieces while you speak.
  - A question prefill whose last words changed is rewound to the last snapshot it shares with the new words, not redone.

**Privacy.** No screenshots are stored. Kept screens are stored as vision features and keys in an owner-only temp folder. They are deleted after an hour, when the app quits, and when you switch it off from the menu bar. Models load offline, and a test runs the app under a no-network sandbox profile. On screen, the menu bar captions are hidden from the assistant's own screen capture.

## Benchmark

`eval/recall/`. Synthetic one-hour browsing sessions are rendered by headless Chrome from a seed. They use invented shops, airlines, hotels, docs and issue trackers, so the model can't answer from what it already knows. Facts are planted on pages, and at the end of the hour every tab is closed and the questions are asked.

- **Hard set (h1–h4), the main set:** 144 questions: 64 precise, 48 vague, 32 about things never shown (16 of them near-misses, like a shoe of a brand that wasn't shown while other trail shoes were open). Pages are dense, with look-alike values on them (other brands' prices, the return flight's seat). One fact per page is only in 12 px fine print, and one is only below the fold. Many pages are on screen for under 15 seconds. That's 101–129 screens an hour, or 34–40k tokens of text.
- **Validity check:** given every screen of the hour, Claude Opus scores 144/144 from the text and 144/144 from the screenshots.
- **The finding that shaped the design.** Gemma, given perfect text notes of the whole hour, gets 109/112 answerable questions right. But it makes up answers to 10 of the 32 never-shown questions, against 0 with no memory. More memory in the context means more made-up answers, so the app puts as little memory in as it can: the timeline plus one screen.
- **Timeline questions** (54: "what was I doing around 10:40?", "what came before the hotel page?") are built and checked, but not yet scored on the final design.

Two private real-screen recordings were used as development data. They stay off GitHub, and only the tools that measure them are here.

## What didn't work

Each was measured. The full write-ups, with bars set before each run, are in [ARCHITECTURE.md](ARCHITECTURE.md).
- **Many pages' KV pieces computed separately and joined.** First token went 2.57 s → 0.11 s, but recall dropped from 94% to 29% correct, because pages computed apart lose attention across pages. A single block after a fixed prefix, as the app uses now, holds up.
- **Prompt-lookup speculative decoding** reached 1.1–1.2 tokens per verify pass. The current screen is an image, so there is almost nothing to copy from.
- **8-bit vision encoder:** −139 MB, but it misread a small table number, the kind of reading the image budget exists for.
- **Image encoder on the Neural Engine:** low power, and quality was fine. But its process held ~0.74 GB, so kept screens are encoded on the GPU in quiet time instead.
- **Stopping made-up answers without training:**
  - a relevance gate against decoy pages;
  - a yes/no judge turn;
  - a fitted abstention gate;
  - quote-then-check;
  - attention bias against distractor memory.

  None met its bar. The model's own signals separate right from wrong answers at only about 0.7 AUC.
- **Selecting retrieval heads** (QRHead-style) gave no gain over using all heads.
- **Entropy-coding the 4-bit weights:** Gemma's codes carry 3.69 bits of 4, so the payoff is small. The scale metadata compressed well instead.

## Limits

- Synthetic, browser-only screens. They are cleaner than real ones, with no ads, pop-ups, editors or chat apps.
- Questions are asked at the end of the hour, one at a time, with no follow-ups.
- The app's end-to-end numbers come from one session at 4× speed, plugged in. The real-time, battery, real-workload run hasn't been done.
- Made-up answers on never-shown questions are well above target.
- Memory peaks around 4 GB, and idle memory drifts between 3.0 and 3.7 GB during an hour.
- macOS on Apple silicon only, English only. Setup is tested on the development machine, not on a clean one.

## Running it

Needs macOS on Apple silicon and Python 3.13.

```sh
./setup.sh                        # venv, Swift helpers, models (~6 GB download); safe to re-run
.venv/bin/python menubar.py       # menu bar app with captions (Turn On / Turn Off)
.venv/bin/python app.py           # the same assistant in the terminal
```

Allow the terminal app in System Settings › Privacy & Security under Screen & System Audio Recording, Microphone, and Input Monitoring (for the Right Option key; Enter works without it).

**Reproducing the results:**

```sh
.venv/bin/python eval/recall/make_sessions.py --sessions h1,h2,h3,h4   # render the hard sessions (needs Chrome)
.venv/bin/python eval/replay_recall.py --session h1 --speed 4          # end-to-end replay (1 = real time)
.venv/bin/python eval/kernels/stored_blocks.py                         # stored blocks vs exact, h1–h4
.venv/bin/python -m pytest                                             # fast tests; -m slow loads the models
```

## Layout

| Path | What |
|---|---|
| `app.py`, `menubar.py` | Entry points: terminal app and menu bar app |
| `recall_app.py`, `recall_worker.py`, `scheduler.py` | Wiring, the GPU thread and the turn scheduler |
| `recall_backend.py`, `kv_reuse.py` | Context assembly, stored KV pieces and blocks, retrieval |
| `recall.py`, `screen_store.py`, `memory.py` | OCR, page store and timeline, kept screens |
| `engine.py`, `vision_fast.py`, `lean_tok.py` | Gemma adapter, piecewise prefill and decode, speech helpers |
| `screen.py`, `asr.py`, `audio_io.py` | Screen capture, speech recognition client, audio |
| `kernels/` | Custom Metal kernels |
| `tools/` | Swift helpers: Parakeet and Kokoro servers, hotkey, window list |
| `eval/` | Benchmark, replay, and the scripts behind every number in ARCHITECTURE.md |
| `ARCHITECTURE.md` | Design notes and every experiment, with its bar set before the run |
