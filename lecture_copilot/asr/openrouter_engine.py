from __future__ import annotations

import base64
import io
import wave

import httpx
import numpy as np

from lecture_copilot.config import AppConfig, LectureContext
from lecture_copilot.cost import COST
from lecture_copilot.net import request_with_retry


def _to_wav_bytes(audio_16k: np.ndarray, sample_rate: int) -> bytes:
    """Encode mono float32 [-1, 1] audio as 16-bit PCM WAV bytes."""
    clipped = np.clip(audio_16k, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype("<i2")
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm.tobytes())
    return buffer.getvalue()


class OpenRouterASR:
    """Cloud speech-to-text through OpenRouter's /audio/transcriptions endpoint.

    Offloads transcription from the CPU (routed to fast Whisper-class providers
    such as Groq), which also keeps up with fast speakers.
    """

    label = "OpenRouter (cloud)"

    def __init__(self, config: AppConfig):
        self.config = config
        self._http: httpx.Client | None = None

    def _client(self) -> httpx.Client:
        if self._http is None:
            self._http = httpx.Client(
                base_url=self.config.openrouter_base_url,
                headers={
                    "Authorization": f"Bearer {self.config.openrouter_api_key}",
                    "HTTP-Referer": "https://github.com/lecture-copilot",
                    "X-Title": "Lecture Copilot",
                },
            )
        return self._http

    def close(self) -> None:
        if self._http is not None:
            self._http.close()
            self._http = None

    def load(self) -> None:
        # Nothing to download; fail early if the key is missing.
        if not self.config.openrouter_api_key:
            raise RuntimeError(
                "ASR_PROVIDER=openrouter, але OPENROUTER_API_KEY порожній. Додай ключ у .env"
            )

    def transcribe(self, audio_16k: np.ndarray, ctx: LectureContext) -> str:
        if audio_16k.size < 1600:
            return ""
        audio_b64 = base64.b64encode(
            _to_wav_bytes(audio_16k.astype(np.float32), self.config.sample_rate)
        ).decode("ascii")
        payload: dict[str, object] = {
            "model": self.config.openrouter_transcribe_model,
            "input_audio": {"data": audio_b64, "format": "wav"},
        }
        language = ctx.language_hint()
        if language:
            payload["language"] = language
        # NOTE: vocabulary hints are provider-specific on OpenRouter
        # (e.g. provider.options.groq.prompt); a top-level "prompt" can 400.

        response = request_with_retry(
            self._client(),
            "POST",
            "/audio/transcriptions",
            json=payload,
            timeout=90.0,
            attempts=5,
        )
        data = response.json()
        usage = data.get("usage") or {}
        COST.add(usage.get("cost") or usage.get("total_cost"), "stt")
        return str(data.get("text", "")).strip()
