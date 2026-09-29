# Lecture Copilot

A Windows desktop app that listens to whatever your computer is playing, transcribes it
locally, translates it into Ukrainian as it goes, and saves the result as a readable
lecture note plus a bundle you can hand to an LLM.

It was built for online lectures: Webex/Teams/Zoom calls, recorded video, YouTube, VLC.
You point it at a loopback audio device, hit start, and forget about it.

---

## What it does

1. Captures **system audio** through WASAPI loopback — no microphone, no virtual cable.
2. Runs voice activity detection on the stream, so audio is only transcribed in phrases,
   not continuously. A phrase is sent to the recognizer once the speaker pauses.
3. Transcribes with **faster-whisper** locally (free, CPU), or with **OpenRouter's**
   `audio/transcriptions` endpoint (Whisper Large V3 Turbo) if you'd rather not heat your CPU.
4. Translates each finished phrase (Ukrainian by default) through **OpenRouter** or
   **Gemini**, retrying with backoff on 429/503.
5. Watches the screen for slides and blackboard notes, and OCRs a frame only after it has
   changed *and* settled — so you get finished slides, not every intermediate stroke.
6. Writes everything to SQLite, to a Markdown file, and to a compact AI bundle.

The GUI shows the original text and the translation side by side, with a level meter, a
timer, pause, and a button to mark a moment as important.

## How it flows

```
WASAPI loopback
  -> VAD (adaptive RMS floor + silence gap)
  -> faster-whisper  |  OpenRouter STT
  -> SQLite segment
  -> translator (OpenRouter / Gemini)   [separate thread]
  -> UI + lecture.md

screen grab (mss)
  -> perceptual fingerprint, change + settle detection
  -> OpenRouter vision  |  local Tesseract
  -> slides + slide_points -> SQLite

SQLite + slides
  -> deterministic cleaning (repeats, ASR junk, duplicate slides)
  -> lecture.pack.md / lecture.ai.json / manifest.json
```

## Requirements

- Windows (loopback capture is WASAPI-specific)
- Python 3.11+
- An OpenRouter or Google AI Studio key **if** you want translation or cloud STT.
  Transcription works fully offline without either.
- Tesseract is optional; it's only the fallback when there's no vision API key.

## Install

```powershell
git clone https://github.com/MisterMog25/Lecture-parser.git
cd Lecture-parser
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
```

Open `.env` and fill in whichever keys you have:

```
OPENROUTER_API_KEY=...     # one key for translation, cloud STT and slide OCR
GEMINI_API_KEY=...         # alternative translator
```

The first local run downloads the Whisper model (`small` is roughly 500 MB) into the
Hugging Face cache. That happens once.

## Running it

```powershell
python -m lecture_copilot
```

Before the lecture:

1. Start the call or the video that you want captured.
2. Pick the loopback device — usually the same device as your headphones.
3. Fill in subject, topic, source language, and any key terms (`malloc`, `realloc`, ...).
   The terms are fed to Whisper as a prompt, which noticeably improves technical vocabulary.
4. Press **START LECTURE**.

Press **STOP** when you're done. Output lands in:

```
data/lecture.db                                 # every lecture, queryable
data/lectures/YYYY-MM-DD-<subject>/lecture.md   # the readable note
```

## Translation and STT backends

| Mode | Speech-to-text | Translation | Cost |
| --- | --- | --- | --- |
| Local | faster-whisper on CPU | OpenRouter or Gemini | STT is free |
| Cloud | OpenRouter `whisper-large-v3-turbo` | OpenRouter or Gemini | roughly $0.006 / minute |

Pick the engine in the app window, or set `ASR_PROVIDER` in `.env`.

`TRANSLATION_PROVIDER=auto` tries OpenRouter first and falls back to Gemini. Both paths
retry with backoff, because Gemini returns 503 often enough to matter.

### Slide and blackboard capture

This part is off unless `SLIDES_ENABLED=1`. It polls the screen and only calls the vision
model when the picture has actually changed and then held still.

- The change check is local and free: a blurred 160x90 fingerprint, compared by the
  fraction of pixels that moved. Video noise and a moving cursor don't trigger it.
- Frames are compared against the last **saved** frame, so slow blackboard writing
  accumulates and eventually gets captured.
- Before saving, the OCR text is compared against the previous slide by word overlap.
  If it's too similar, the frame is dropped — this kills the duplicates you'd otherwise
  get from animations, re-renders and cursor movement.

The knobs (`SLIDES_CHANGE_THRESHOLD`, `SLIDES_SETTLE_SECONDS`, `SLIDES_MIN_INTERVAL_SECONDS`,
`SLIDES_DEDUP_RATIO`) are all in `.env.example` with comments. Lower `SLIDES_DEDUP_RATIO`
means more aggressive merging, which means fewer images.

Frames are saved as JPEGs and the extracted text, summary, formulas and code go into the
database and into the Markdown note.

### Lecturer slides from Moodle

Each lecture folder has a `moodle/` subfolder. Drop the lecturer's PDFs or PNGs in there
and they show up in `manifest.json` as `assets.moodle` — treated as the preferred visual
source, since they're cleaner than screenshots of someone's shared screen.

## Output files

After a lecture you get four files per lecture folder. Only the first is meant for you:

| File | For | Contents |
| --- | --- | --- |
| `lecture.md` | humans | readable transcript with slides inserted as images |
| `lecture.pack.md` | LLMs | one compact file: cleaned transcript + slides + Moodle assets |
| `lecture.ai.json` | LLMs / code | canonical structured bundle: segments, slides, `key_terms`, stats |
| `manifest.json` | agents | index of every asset and how to use it |

The cleaning step is deterministic and costs nothing. It collapses repeated ASR output
(Whisper likes to repeat itself), drops known junk (`[Music]`, subtitle boilerplate), and
merges repeated captures of the same slide, keeping a `repeats` count.

Re-export any time:

```powershell
python -m lecture_copilot.export --latest   # or a lecture id, or --all
```

## Generating a presentation

There's a template and a render script under `tools/`. The intended workflow is that an
agent reads the AI bundle and writes `deck.html`, then you render it to PDF and look at
the preview PNGs to check the result:

```bash
tools/render_deck.sh data/lectures/<slug>/deck
```

It produces `deck.pdf` plus `preview/p*.png`. The script finds a Windows Chrome or Edge
from WSL, and falls back to a Linux chromium if there is one.

This step is optional and the repo doesn't insist on any particular agent — the bundle is
plain Markdown and JSON, so anything that can read a file can do it.

## Speeding up local Whisper

- Use a smaller model (`tiny`, `base`) in the Whisper dropdown. Faster, less accurate.
- `WHISPER_CPU_THREADS=0` means auto. Try 4 or 6 if the CPU is shared with the call.
- `MAX_PENDING_PHRASES` is a latency guard: when the speaker is fast and the recognizer
  falls behind, the oldest queued phrase is dropped so the transcript stays in sync with
  real time instead of drifting.
- With an NVIDIA GPU you can switch to CUDA in `lecture_copilot/config.py`
  (`whisper_device=cuda`, `whisper_compute=float16`).

## Privacy

Recording system audio and your screen may be against your lecturer's or university's
rules. Use it where you're allowed to. In the default local mode nothing leaves your
machine. If you switch to the cloud engine the audio goes to OpenRouter, and slide frames
go to the vision model — so treat those as external services.

## Project layout

```
lecture_copilot/
  audio/        WASAPI loopback capture
  asr/          faster-whisper and OpenRouter engines
  translation/  OpenRouter / Gemini translator with retry
  vision/       screen capture, change detection, slide extraction
  pipeline/     session orchestration and the UI event queue
  storage/      SQLite layer, Markdown writer, AI bundle exporter
  ui/           the customtkinter window
  config.py     all settings, read from .env
  cost.py       live OpenRouter spend tracking
tools/          deck template and the HTML->PDF render script
```

`data/` and `.env` are gitignored on purpose. The database holds your lectures and the
`.env` holds your keys; neither belongs in a commit.

## Known limitations

- Windows only, because of loopback capture.
- Single mixed stream — there's no speaker separation.
- Translation quality is whatever the model gives you; short phrases with no context are
  harder than long ones, which is why the phrase buffer waits for a pause.
- If the CPU is busy, local Whisper will lag behind real time. That's what the cloud
  engine is for.
