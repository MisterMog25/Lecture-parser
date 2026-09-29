from __future__ import annotations

import threading

import numpy as np
from faster_whisper import WhisperModel

from lecture_copilot.config import AppConfig, LectureContext


class WhisperASR:
    label = "Whisper (local)"

    def __init__(self, config: AppConfig):
        self.config = config
        self._model: WhisperModel | None = None
        self._lock = threading.Lock()
        self._load_lock = threading.Lock()

    def load(self) -> None:
        if self._model is not None:
            return
        # The pipeline thread and the ASR thread can both trigger the first load.
        # Without this lock they each build the model (double download / crash).
        with self._load_lock:
            if self._model is not None:
                return
            self._model = WhisperModel(
                self.config.whisper_model,
                device=self.config.whisper_device,
                compute_type=self.config.whisper_compute,
                cpu_threads=self.config.whisper_cpu_threads,
            )

    def transcribe(self, audio_16k: np.ndarray, ctx: LectureContext) -> str:
        self.load()
        assert self._model is not None
        if audio_16k.size < 1600:
            return ""
        audio = np.ascontiguousarray(audio_16k, dtype=np.float32)
        with self._lock:
            lang = ctx.source_language.strip().lower() if ctx.source_language else "auto"
            segments, _info = self._model.transcribe(
                audio,
                language=None if lang in {"", "auto"} else lang,
                initial_prompt=ctx.whisper_prompt() or None,
                vad_filter=False,
                beam_size=1,
                condition_on_previous_text=False,
            )
            text = " ".join(seg.text.strip() for seg in segments).strip()
        return text
