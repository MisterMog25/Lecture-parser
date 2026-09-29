from __future__ import annotations

import httpx

from lecture_copilot.config import AppConfig, LectureContext
from lecture_copilot.cost import COST
from lecture_copilot.net import request_with_retry, retry_call


TRANSLATE_SYSTEM = """You are a precise lecture translator.
Translate the source lecture speech into Ukrainian.
Keep programming, math, and scientific terminology accurate.
Do not simplify the meaning.
Preserve function names, identifiers, and code snippets unchanged.
Return only the translation, no quotes or commentary."""


class Translator:
    """Translates lecture text; each provider is retried with backoff on 503/429."""

    def __init__(self, config: AppConfig):
        self.config = config
        self._gemini = None
        self._http: httpx.Client | None = None

    def available(self) -> bool:
        return self.config.translation_enabled

    def provider_label(self) -> str:
        providers = self.config.translation_providers()
        return ", ".join(providers) if providers else "none"

    def close(self) -> None:
        if self._http is not None:
            self._http.close()
            self._http = None

    def _http_client(self) -> httpx.Client:
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

    def _gemini_client(self):  # type: ignore[no-untyped-def]
        if self._gemini is None:
            from google import genai

            self._gemini = genai.Client(api_key=self.config.gemini_api_key)
        return self._gemini

    def translate(self, text: str, ctx: LectureContext) -> str:
        if not text.strip():
            return ""
        providers = self.config.translation_providers()
        if not providers:
            return ""
        last: Exception | None = None
        for provider in providers:
            try:
                if provider == "openrouter":
                    return self._translate_openrouter(text, ctx)
                if provider == "gemini":
                    return self._translate_gemini(text, ctx)
            except Exception as exc:  # noqa: BLE001 - try the next provider
                last = exc
        if last is not None:
            raise last
        return ""

    def _user_prompt(self, text: str, ctx: LectureContext) -> str:
        vocab = ", ".join(ctx.keywords) if ctx.keywords else "(none)"
        return (
            f"Subject: {ctx.subject}\n"
            f"Topic: {ctx.topic or '(unspecified)'}\n"
            f"Source language code: {ctx.source_language}\n"
            f"Teacher vocabulary: {vocab}\n\n"
            f"Text:\n{text.strip()}"
        )

    def _translate_openrouter(self, text: str, ctx: LectureContext) -> str:
        payload = {
            "model": self.config.openrouter_translate_model,
            "temperature": 0.2,
            # Translation needs no chain-of-thought: disabling it cuts latency
            # and any hidden "thinking" token cost on reasoning models.
            "reasoning": {"enabled": False},
            "messages": [
                {"role": "system", "content": TRANSLATE_SYSTEM},
                {"role": "user", "content": self._user_prompt(text, ctx)},
            ],
        }
        response = request_with_retry(
            self._http_client(),
            "POST",
            "/chat/completions",
            json=payload,
            timeout=60.0,
            attempts=5,
        )
        data = response.json()
        usage = data.get("usage") or {}
        COST.add(usage.get("cost") or usage.get("total_cost"), "translate")
        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError(f"OpenRouter: empty response {data}")
        content = (choices[0].get("message") or {}).get("content") or ""
        return content.strip()

    def _translate_gemini(self, text: str, ctx: LectureContext) -> str:
        client = self._gemini_client()
        prompt = f"{TRANSLATE_SYSTEM}\n\n{self._user_prompt(text, ctx)}"

        def _call():  # type: ignore[no-untyped-def]
            return client.models.generate_content(
                model=self.config.translate_model,
                contents=prompt,
            )

        response = retry_call(_call, attempts=5, base_delay=0.8, max_delay=12.0)
        return (response.text or "").strip()
