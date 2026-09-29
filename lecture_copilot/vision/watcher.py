from __future__ import annotations

import re
import threading
import time
from pathlib import Path
from queue import Queue
from typing import Callable

from PIL import Image

from lecture_copilot.config import AppConfig, LectureContext
from lecture_copilot.pipeline.events import EventKind, PipelineEvent
from lecture_copilot.storage import insert_slide, insert_slide_point
from lecture_copilot.vision.change import difference, signature
from lecture_copilot.vision.extract import SlideContent, SlideExtractor
from lecture_copilot.vision.screen import ScreenCapture

Grabber = Callable[[], Image.Image]


_WORD = re.compile(r"[0-9a-zа-яіїєґ_+-]+", re.IGNORECASE)


def _normalize(text: str) -> str:
    return " ".join((text or "").lower().split())


def _tokens(text: str) -> set[str]:
    return set(_WORD.findall(text.lower()))


def _similar(a: str, b: str, ratio: float) -> bool:
    """Word-set Jaccard similarity: tolerant of OCR noise, strict on new content."""
    sa, sb = _tokens(a), _tokens(b)
    if not sa or not sb:
        return False
    return len(sa & sb) / len(sa | sb) >= ratio


class SlideWatcher:
    """Polls the screen cheaply and OCRs only frames that settle after a change.

    Flow: grab -> perceptual fingerprint -> if changed, remember frame ->
    once the frame has been stable for `settle` seconds, save + extract.
    This captures a finished slide or finished blackboard notes, not every
    intermediate stroke, and never re-extracts while nothing changes.
    """

    def __init__(
        self,
        config: AppConfig,
        ctx: LectureContext,
        lecture_id: int,
        folder: Path,
        events: Queue[PipelineEvent],
        extractor: SlideExtractor,
        grabber: Grabber | None = None,
    ):
        self.config = config
        self.ctx = ctx
        self.lecture_id = lecture_id
        self.folder = folder
        self.events = events
        self.extractor = extractor
        self._grabber = grabber
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_key = ""

    def _emit(self, event: PipelineEvent) -> None:
        self.events.put(event)

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="slide-watcher", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=10)
            self._thread = None

    def _run(self) -> None:
        capture: ScreenCapture | None = None
        grabber = self._grabber
        if grabber is None:
            try:
                capture = ScreenCapture(self.config.slide_monitor)
                grabber = capture.grab
            except Exception as exc:  # noqa: BLE001
                self._emit(PipelineEvent(kind=EventKind.ERROR, message=f"Екран: {exc}"))
                return

        scan = self.config.slide_scan_seconds
        settle = self.config.slide_settle_seconds
        threshold = self.config.slide_change_threshold
        min_interval = self.config.slide_min_interval_seconds
        started = time.monotonic()

        self._emit(
            PipelineEvent(
                kind=EventKind.STATUS,
                message=f"Слайди: {self.extractor.label}. Сканую екран кожні {scan:g} с.",
            )
        )

        prev_sig = None
        baseline_sig = None  # signature of the last *emitted* slide
        pending: Image.Image | None = None
        last_change = started
        last_emit = 0.0
        try:
            while not self._stop.is_set():
                try:
                    frame = grabber()
                except Exception as exc:  # noqa: BLE001
                    self._emit(PipelineEvent(kind=EventKind.ERROR, message=f"Захоплення екрана: {exc}"))
                    self._stop.wait(scan)
                    continue

                sig = signature(frame)
                now = time.monotonic()

                # Active motion (cursor, writing) resets the settle timer.
                if prev_sig is None or difference(prev_sig, sig) >= threshold:
                    last_change = now
                prev_sig = sig

                # Any frame different from the last captured slide is a candidate.
                # Comparing to the baseline (not the previous frame) lets slow
                # writing accumulate until it is worth an OCR call.
                if baseline_sig is None or difference(baseline_sig, sig) >= threshold:
                    pending = frame

                if (
                    pending is not None
                    and (now - last_change) >= settle
                    and (now - last_emit) >= min_interval
                ):
                    emitted = pending
                    pending = None
                    last_emit = now
                    baseline_sig = signature(emitted)
                    self._process(emitted, now - started)

                self._stop.wait(scan)
        finally:
            if capture is not None:
                capture.close()

    def _process(self, image: Image.Image, elapsed: float) -> None:
        content = SlideContent()
        if self.extractor.available():
            try:
                content = self.extractor.extract(image, self.ctx)
            except Exception as exc:  # noqa: BLE001
                self._emit(PipelineEvent(kind=EventKind.ERROR, message=f"Слайд OCR: {exc}"))

        # Content-level dedup: a visually different frame with the same text
        # (mouse-jittered slide, animation, re-render) must not become a new
        # picture. Compare recognised text+summary against the previous slide.
        key = _normalize(f"{content.text} {content.summary}")
        if not key:
            return  # nothing recognised -> not worth a picture
        if self._last_key and _similar(key, self._last_key, self.config.slide_dedup_ratio):
            self._emit(
                PipelineEvent(
                    kind=EventKind.STATUS,
                    message="Слайд без нової інформації — пропускаю",
                )
            )
            return
        self._last_key = key

        path = self._save_image(image, elapsed)
        try:
            slide_id = insert_slide(
                self.lecture_id,
                elapsed,
                str(path) if path else None,
                content.text,
                content.summary,
                content.title,
                content.formulas,
                content.code,
            )
            for point in content.key_points:
                insert_slide_point(slide_id, point)
        except Exception as exc:  # noqa: BLE001
            self._emit(PipelineEvent(kind=EventKind.ERROR, message=f"Слайд DB: {exc}"))
            return

        self._emit(
            PipelineEvent(
                kind=EventKind.SLIDE,
                original=content.text,
                translated=content.summary,
                message=content.title or "Слайд",
                start_time=elapsed,
                end_time=elapsed,
            )
        )

    def _save_image(self, image: Image.Image, elapsed: float) -> Path | None:
        try:
            slides_dir = self.folder / "slides"
            slides_dir.mkdir(parents=True, exist_ok=True)
            path = slides_dir / f"{int(elapsed * 1000):07d}.jpg"
            save = image
            if save.width > self.config.slide_max_width:
                height = max(1, int(save.height * self.config.slide_max_width / save.width))
                save = save.resize((self.config.slide_max_width, height), Image.BILINEAR)
            save.convert("RGB").save(path, format="JPEG", quality=82)
            return path
        except Exception as exc:  # noqa: BLE001
            self._emit(PipelineEvent(kind=EventKind.ERROR, message=f"Збереження слайда: {exc}"))
            return None
