from __future__ import annotations

from typing import Protocol

import numpy as np

from lecture_copilot.config import AppConfig, LectureContext


class ASREngine(Protocol):
    label: str

    def load(self) -> None: ...

    def transcribe(self, audio_16k: np.ndarray, ctx: LectureContext) -> str: ...


def build_asr(config: AppConfig) -> ASREngine:
    """Pick the ASR backend from config.asr_provider."""
    if config.asr_provider == "openrouter":
        from lecture_copilot.asr.openrouter_engine import OpenRouterASR

        return OpenRouterASR(config)
    from lecture_copilot.asr.whisper_engine import WhisperASR

    return WhisperASR(config)
