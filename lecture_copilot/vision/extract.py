from __future__ import annotations

import base64
import io
import json
import re
from dataclasses import dataclass, field
from typing import Protocol

import httpx
from PIL import Image

from lecture_copilot.config import AppConfig, LectureContext
from lecture_copilot.cost import COST
from lecture_copilot.net import request_with_retry


@dataclass
class SlideContent:
    text: str = ""
    title: str = ""
    summary: str = ""
    key_points: list[str] = field(default_factory=list)
    formulas: str = ""
    code: str = ""


SLIDE_SYSTEM = """You read a single frame from a university lecture (slide, blackboard,
or shared screen). Extract what a student would need later.

Rules:
- Transcribe the visible text verbatim first (may be English, Slovak, Czech, Ukrainian).
- Then write a short summary in Ukrainian.
- List the concrete concepts/key points that were taught.
- Keep formulas and code exactly as written; do not invent anything.
- If the frame is not informative (video call faces, blank desktop), say so briefly.

Return ONLY JSON with these keys:
{"text": str, "title": str, "summary": str, "key_points": [str], "formulas": str, "code": str}"""


def image_to_data_uri(image: Image.Image, max_width: int = 1280, quality: int = 80) -> str:
    """Downscale + JPEG-encode to keep vision tokens and bytes low."""
    if image.width > max_width:
        height = max(1, int(image.height * max_width / image.width))
        image = image.resize((max_width, height), Image.BILINEAR)
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=quality)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _user_prompt(ctx: LectureContext) -> str:
    vocab = ", ".join(ctx.keywords) if ctx.keywords else "(none)"
    return (
        f"Subject: {ctx.subject}\n"
        f"Topic: {ctx.topic or '(unspecified)'}\n"
        f"Lecture language: {ctx.source_language}\n"
        f"Known vocabulary: {vocab}"
    )


def parse_slide_content(raw: str) -> SlideContent:
    if not raw or not raw.strip():
        return SlideContent()
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?|```$", "", text).strip()
    data: dict | None = None
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
            except json.JSONDecodeError:
                data = None
    if not isinstance(data, dict):
        return SlideContent(text=text, summary=text[:400])
    points = data.get("key_points") or []
    if isinstance(points, str):
        points = [points]
    return SlideContent(
        text=str(data.get("text") or "").strip(),
        title=str(data.get("title") or "").strip(),
        summary=str(data.get("summary") or data.get("text") or "").strip(),
        key_points=[str(p).strip() for p in points if str(p).strip()],
        formulas=str(data.get("formulas") or "").strip(),
        code=str(data.get("code") or "").strip(),
    )


class SlideExtractor(Protocol):
    label: str

    def available(self) -> bool: ...

    def extract(self, image: Image.Image, ctx: LectureContext) -> SlideContent: ...

    def close(self) -> None: ...


class OpenRouterVisionExtractor:
    """Gemini Flash-Lite vision via OpenRouter — structured text cheaply."""

    label = "OpenRouter vision"

    def __init__(self, config: AppConfig):
        self.config = config
        self._http: httpx.Client | None = None

    def available(self) -> bool:
        return bool(self.config.openrouter_api_key)

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

    def extract(self, image: Image.Image, ctx: LectureContext) -> SlideContent:
        payload = {
            "model": self.config.openrouter_slide_model,
            "temperature": 0.1,
            "reasoning": {"enabled": False},
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": SLIDE_SYSTEM},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": _user_prompt(ctx)},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_to_data_uri(image, self.config.slide_max_width)
                            },
                        },
                    ],
                },
            ],
        }
        response = request_with_retry(
            self._client(),
            "POST",
            "/chat/completions",
            json=payload,
            timeout=90.0,
            attempts=5,
        )
        data = response.json()
        usage = data.get("usage") or {}
        COST.add(usage.get("cost") or usage.get("total_cost"), "slides")
        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError(f"OpenRouter vision: empty response {data}")
        content = (choices[0].get("message") or {}).get("content") or ""
        return parse_slide_content(content)


class TesseractExtractor:
    """Free, offline OCR — only keyed text, no summary. Needs the Tesseract binary."""

    label = "Tesseract (local)"

    def available(self) -> bool:
        try:
            import pytesseract
        except ImportError:
            return False
        try:
            pytesseract.get_tesseract_version()
            return True
        except Exception:
            return False

    def close(self) -> None:
        return None

    def extract(self, image: Image.Image, ctx: LectureContext) -> SlideContent:
        import pytesseract

        text = (pytesseract.image_to_string(image) or "").strip()
        return SlideContent(text=text, summary=text[:400])


def build_extractor(config: AppConfig) -> SlideExtractor:
    provider = config.slide_provider
    if provider == "tesseract":
        return TesseractExtractor()
    if provider == "openrouter":
        return OpenRouterVisionExtractor(config)
    return OpenRouterVisionExtractor(config) if config.openrouter_api_key else TesseractExtractor()
