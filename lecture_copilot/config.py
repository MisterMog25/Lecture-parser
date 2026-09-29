from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")
DATA_DIR = ROOT / "data"
LECTURES_DIR = DATA_DIR / "lectures"
DB_PATH = DATA_DIR / "lecture.db"


@dataclass
class LectureContext:
    subject: str = "Lecture"
    topic: str = ""
    source_language: str = "auto"
    target_language: str = "uk"
    keywords: list[str] = field(default_factory=list)

    def whisper_prompt(self) -> str:
        bits = [self.subject]
        if self.topic:
            bits.append(self.topic)
        if self.keywords:
            bits.append("Vocabulary: " + ", ".join(self.keywords))
        return ". ".join(bits)[:800]

    def language_hint(self) -> str | None:
        lang = (self.source_language or "auto").strip().lower()
        return None if lang in {"", "auto"} else lang


@dataclass
class AppConfig:
    gemini_api_key: str = field(default_factory=lambda: os.getenv("GEMINI_API_KEY", "").strip())
    translate_model: str = field(
        default_factory=lambda: os.getenv("GEMINI_TRANSLATE_MODEL", "gemini-3.5-flash-lite")
    )
    slide_model: str = field(
        default_factory=lambda: os.getenv("GEMINI_SLIDE_MODEL", "gemini-3.8-flash")
    )
    transcribe_model: str = field(
        default_factory=lambda: os.getenv("GEMINI_TRANSCRIBE_MODEL", "gemini-3.5-transcribe-live")
    )

    # OpenRouter (https://openrouter.ai/keys): one key for translation AND cloud STT.
    openrouter_api_key: str = field(
        default_factory=lambda: os.getenv("OPENROUTER_API_KEY", "").strip()
    )
    openrouter_base_url: str = field(
        default_factory=lambda: os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/")
    )
    openrouter_translate_model: str = field(
        default_factory=lambda: os.getenv("OPENROUTER_TRANSLATE_MODEL", "google/gemini-2.5-flash-lite")
    )
    openrouter_transcribe_model: str = field(
        default_factory=lambda: os.getenv(
            "OPENROUTER_TRANSCRIBE_MODEL", "openai/whisper-large-v3-turbo"
        )
    )
    # "auto" | "openrouter" | "gemini" — which translator to try first.
    translation_provider: str = field(
        default_factory=lambda: os.getenv("TRANSLATION_PROVIDER", "auto").strip().lower()
    )
    # "whisper" (local CPU) | "openrouter" (cloud STT).
    asr_provider: str = field(
        default_factory=lambda: os.getenv("ASR_PROVIDER", "whisper").strip().lower()
    )

    whisper_model: str = "small"
    whisper_device: str = "cpu"
    whisper_compute: str = "int8"
    whisper_cpu_threads: int = field(
        default_factory=lambda: int(os.getenv("WHISPER_CPU_THREADS", "0") or 0)
    )
    sample_rate: int = 16000
    phrase_silence_ms: int = 600
    min_phrase_ms: int = 1000
    max_phrase_ms: int = 8000
    vad_rms_threshold: float = 0.004
    # If the ASR falls behind (fast speaker), drop the oldest pending phrase so
    # the transcript stays live instead of lagging further and further behind.
    max_pending_phrases: int = field(
        default_factory=lambda: int(os.getenv("MAX_PENDING_PHRASES", "3") or 3)
    )

    # -- Slide / blackboard capture (Phase 3) --
    openrouter_slide_model: str = field(
        default_factory=lambda: os.getenv("OPENROUTER_SLIDE_MODEL", "google/gemini-2.5-flash-lite")
    )
    slide_enabled: bool = field(
        default_factory=lambda: os.getenv("SLIDES_ENABLED", "1").strip().lower()
        not in {"0", "false", "no", "off"}
    )
    # auto | openrouter | tesseract | none
    slide_provider: str = field(
        default_factory=lambda: os.getenv("SLIDES_PROVIDER", "auto").strip().lower()
    )
    slide_monitor: int = field(default_factory=lambda: int(os.getenv("SLIDES_MONITOR", "1") or 1))
    slide_scan_seconds: float = field(
        default_factory=lambda: float(os.getenv("SLIDES_SCAN_SECONDS", "2.0") or 2.0)
    )
    slide_settle_seconds: float = field(
        default_factory=lambda: float(os.getenv("SLIDES_SETTLE_SECONDS", "2.5") or 2.5)
    )
    slide_change_threshold: float = field(
        default_factory=lambda: float(os.getenv("SLIDES_CHANGE_THRESHOLD", "0.0015") or 0.0015)
    )
    slide_min_interval_seconds: float = field(
        default_factory=lambda: float(os.getenv("SLIDES_MIN_INTERVAL_SECONDS", "12.0") or 12.0)
    )
    # Skip a captured frame if its recognised text is this similar to the
    # previous slide (0..1). Higher = stricter = fewer pictures.
    slide_dedup_ratio: float = field(
        default_factory=lambda: float(os.getenv("SLIDES_DEDUP_RATIO", "0.85") or 0.85)
    )
    slide_max_width: int = field(
        default_factory=lambda: int(os.getenv("SLIDES_MAX_WIDTH", "1280") or 1280)
    )

    def _provider_ready(self, provider: str) -> bool:
        if provider == "gemini":
            return bool(self.gemini_api_key)
        if provider == "openrouter":
            return bool(self.openrouter_api_key)
        return False

    def translation_providers(self) -> list[str]:
        """Ordered providers to try; the first is primary, the rest are fallbacks."""
        pref = self.translation_provider
        if pref == "gemini":
            order = ["gemini", "openrouter"]
        elif pref == "openrouter":
            order = ["openrouter", "gemini"]
        else:  # auto: prefer OpenRouter when present, Gemini otherwise
            order = ["openrouter", "gemini"]
        return [p for p in order if self._provider_ready(p)]

    @property
    def translation_enabled(self) -> bool:
        return bool(self.translation_providers())


def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    LECTURES_DIR.mkdir(parents=True, exist_ok=True)
