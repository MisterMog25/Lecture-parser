from __future__ import annotations

import threading
import time
from queue import Empty, Queue

import numpy as np

from lecture_copilot.asr.whisper_engine import WhisperASR
from lecture_copilot.audio.capture import LoopbackCapture, LoopbackDevice, resample_mono
from lecture_copilot.config import AppConfig, LectureContext
from lecture_copilot.cost import COST
from lecture_copilot.pipeline.events import EventKind, PipelineEvent
from lecture_copilot.storage import (
    create_lecture,
    end_lecture,
    insert_segment,
    lecture_dir,
    update_segment_translation,
)
from lecture_copilot.translation.translator import Translator


class LectureSession:
    def __init__(
        self,
        config: AppConfig,
        ctx: LectureContext,
        asr: WhisperASR,
        translator: Translator,
        events: Queue[PipelineEvent],
        device: LoopbackDevice | None = None,
        slide_grabber=None,
    ):
        self.config = config
        self.ctx = ctx
        self.asr = asr
        self.translator = translator
        self.events = events
        self.device = device
        self.lecture_id: int | None = None
        self._stop = threading.Event()
        self._paused = threading.Event()
        self._thread: threading.Thread | None = None
        self._asr_thread: threading.Thread | None = None
        self._jobs: Queue[tuple[np.ndarray, float, float] | None] = Queue()
        self._started_monotonic = 0.0
        self._cost_start = 0.0
        self._watcher = None
        self._extractor = None
        self._slide_grabber = slide_grabber

    def start(self) -> None:
        title = self.ctx.topic or self.ctx.subject
        self.lecture_id = create_lecture(
            title=title,
            subject=self.ctx.subject,
            topic=self.ctx.topic,
            source_language=self.ctx.source_language,
            target_language=self.ctx.target_language,
        )
        self._stop.clear()
        self._started_monotonic = time.monotonic()
        self._cost_start = COST.total()
        self._start_slides()
        self._thread = threading.Thread(target=self._run, name="lecture-pipeline", daemon=True)
        self._asr_thread = threading.Thread(target=self._asr_loop, name="lecture-asr", daemon=True)
        self._thread.start()
        self._asr_thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=8)
            self._thread = None
        self._jobs.put(None)
        if self._asr_thread is not None:
            self._asr_thread.join(timeout=30)
            self._asr_thread = None
        self._stop_slides()
        if self.lecture_id is not None:
            end_lecture(self.lecture_id, cost_usd=max(0.0, COST.total() - self._cost_start))

    def _start_slides(self) -> None:
        if not self.config.slide_enabled or self.config.slide_provider == "none":
            return
        if self.lecture_id is None:
            return
        try:
            # Imported lazily to avoid a pipeline <-> vision import cycle.
            from lecture_copilot.vision.extract import build_extractor
            from lecture_copilot.vision.watcher import SlideWatcher

            self._extractor = build_extractor(self.config)
            if not self._extractor.available():
                self._emit(
                    PipelineEvent(
                        kind=EventKind.STATUS,
                        message="Слайди вимкнено: немає OpenRouter-ключа і не знайдено Tesseract.",
                    )
                )
                return
            self._watcher = SlideWatcher(
                config=self.config,
                ctx=self.ctx,
                lecture_id=self.lecture_id,
                folder=lecture_dir(self.lecture_id),
                events=self.events,
                extractor=self._extractor,
                grabber=self._slide_grabber,
            )
            self._watcher.start()
        except Exception as exc:  # noqa: BLE001
            self._emit(PipelineEvent(kind=EventKind.ERROR, message=f"Слайди: {exc}"))

    def _stop_slides(self) -> None:
        if self._watcher is not None:
            self._watcher.stop()
            self._watcher = None
        close = getattr(self._extractor, "close", None)
        if callable(close):
            close()
        self._extractor = None

    def set_paused(self, paused: bool) -> None:
        if paused:
            self._paused.set()
            self._emit(PipelineEvent(kind=EventKind.STATUS, message="Пауза"))
        else:
            self._paused.clear()
            self._emit(PipelineEvent(kind=EventKind.STATUS, message="Запис продовжено"))

    def is_paused(self) -> bool:
        return self._paused.is_set()

    def _emit(self, event: PipelineEvent) -> None:
        self.events.put(event)

    def _run(self) -> None:
        self._emit(
            PipelineEvent(
                kind=EventKind.STATUS,
                message=f"Запускаю {self.asr.label}... (перший старт локальної моделі 1–5 хв, не тисни STOP)",
            )
        )
        try:
            self.asr.load()
            self._emit(
                PipelineEvent(kind=EventKind.STATUS, message=f"{self.asr.label} готовий. Відкриваю звук...")
            )
        except Exception as exc:
            self._emit(PipelineEvent(kind=EventKind.ERROR, message=f"Whisper: {exc}"))
            return

        capture = LoopbackCapture(self.device)
        try:
            capture.start()
            self._emit(
                PipelineEvent(
                    kind=EventKind.STATUS,
                    message=f"Слухаю: {capture.device.name}. Грай звук — смужка рівня має рухатися.",
                )
            )
        except Exception as exc:
            self._emit(PipelineEvent(kind=EventKind.ERROR, message=str(exc)))
            return

        buf: list[np.ndarray] = []
        voiced = False
        silence_samples = 0
        voiced_samples = 0
        phrase_start = 0.0
        dst = self.config.sample_rate
        src = capture.device.sample_rate
        silence_needed = int(dst * self.config.phrase_silence_ms / 1000)
        min_voiced = int(dst * self.config.min_phrase_ms / 1000)
        max_voiced = int(dst * self.config.max_phrase_ms / 1000)
        noise = 0.001
        last_level = 0.0

        try:
            while not self._stop.is_set():
                if self._paused.is_set():
                    capture.read()
                    time.sleep(0.02)
                    continue
                chunk = resample_mono(capture.read(), src, dst)
                rms = float(np.sqrt(np.mean(np.square(chunk))) + 1e-12)
                now = time.monotonic() - self._started_monotonic
                # Only adapt the noise floor while we are NOT inside a phrase.
                # Updating it during speech makes it converge on the speech level,
                # which pushes the threshold above normal talking and locks out
                # every later phrase (sound is visible, but nothing is transcribed).
                if not voiced:
                    noise = 0.98 * noise + 0.02 * rms
                # Cap the adaptive part so it can never run away from the floor.
                threshold = max(
                    self.config.vad_rms_threshold,
                    min(noise * 3.5, self.config.vad_rms_threshold * 6),
                )

                if now - last_level >= 0.2:
                    last_level = now
                    self._emit(
                        PipelineEvent(
                            kind=EventKind.LEVEL,
                            level=rms,
                            message="VAD: чує мову" if voiced else "VAD: тиша",
                        )
                    )

                if rms >= threshold:
                    if not voiced:
                        phrase_start = max(0.0, now - chunk.shape[0] / dst)
                        voiced = True
                    buf.append(chunk)
                    voiced_samples += chunk.shape[0]
                    silence_samples = 0
                    if voiced_samples >= max_voiced:
                        self._queue_phrase(buf, phrase_start, now)
                        buf = []
                        voiced = False
                        voiced_samples = 0
                else:
                    if voiced:
                        buf.append(chunk)
                        silence_samples += chunk.shape[0]
                        if silence_samples >= silence_needed and voiced_samples >= min_voiced:
                            self._queue_phrase(buf, phrase_start, now)
                            buf = []
                            voiced = False
                            voiced_samples = 0
                            silence_samples = 0
                    else:
                        buf = []
        finally:
            if buf and voiced_samples >= min_voiced:
                now = time.monotonic() - self._started_monotonic
                self._queue_phrase(buf, phrase_start, now)
            capture.stop()
            self._jobs.put(None)

    def _queue_phrase(self, parts: list[np.ndarray], start: float, end: float) -> None:
        audio = np.concatenate(parts)
        # Keep the transcript live under fast speech: if the ASR worker is
        # behind, drop the oldest *pending* phrase rather than falling further
        # and further behind in real time.
        dropped = False
        while self._jobs.qsize() >= self.config.max_pending_phrases:
            try:
                self._jobs.get_nowait()
            except Empty:
                break
            dropped = True
        if dropped:
            self._emit(
                PipelineEvent(
                    kind=EventKind.STATUS,
                    message=f"{self.asr.label} не встигає — пропускаю найстаріший фрагмент, щоб не відставати",
                )
            )
        self._jobs.put((audio, start, end))

    def _asr_loop(self) -> None:
        while True:
            item = self._jobs.get()
            if item is None:
                if self._stop.is_set():
                    break
                continue
            audio, start, end = item
            try:
                original = self.asr.transcribe(audio, self.ctx)
            except Exception as exc:
                self._emit(PipelineEvent(kind=EventKind.ERROR, message=f"ASR: {exc}"))
                continue
            if not original:
                self._emit(
                    PipelineEvent(
                        kind=EventKind.STATUS,
                        message="Чути звук, але ASR не дав текст. Перевір мову / джерело звуку.",
                    )
                )
                continue
            assert self.lecture_id is not None
            try:
                segment_id = insert_segment(self.lecture_id, start, end, original, None)
            except Exception as exc:
                # A DB hiccup must NOT kill the ASR worker, or transcription stops
                # for the rest of the lecture with no visible error.
                self._emit(PipelineEvent(kind=EventKind.ERROR, message=f"DB: {exc}"))
                continue
            self._emit(
                PipelineEvent(
                    kind=EventKind.SEGMENT,
                    original=original,
                    start_time=start,
                    end_time=end,
                )
            )
            if self.translator.available():
                threading.Thread(
                    target=self._translate,
                    args=(segment_id, original, start, end),
                    name="translate",
                    daemon=True,
                ).start()

    def _translate(self, seg_id: int, src: str, t0: float, t1: float) -> None:
        try:
            translated = self.translator.translate(src, self.ctx)
        except Exception as exc:
            self._emit(PipelineEvent(kind=EventKind.ERROR, message=f"Переклад: {exc}"))
            return
        if translated:
            update_segment_translation(seg_id, translated)
            self._emit(
                PipelineEvent(
                    kind=EventKind.SEGMENT,
                    original=src,
                    translated=translated,
                    start_time=t0,
                    end_time=t1,
                )
            )
