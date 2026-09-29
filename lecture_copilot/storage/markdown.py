from __future__ import annotations

from datetime import date
from pathlib import Path

from lecture_copilot.config import LECTURES_DIR, ensure_dirs
from lecture_copilot.storage.db import get_lecture, list_segments, list_slide_points, list_slides


def _fmt_ts(seconds: float) -> str:
    total = int(max(0, seconds))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def _slug(value: str) -> str:
    keep = []
    for ch in value.lower().replace(" ", "-"):
        if ch.isalnum() or ch in "-_":
            keep.append(ch)
    return "".join(keep).strip("-") or "lecture"


def lecture_dir(lecture_id: int) -> Path:
    """Folder for a lecture's markdown, slides and other assets."""
    row = get_lecture(lecture_id)
    if row is None:
        raise ValueError(f"Lecture {lecture_id} not found")
    subject = row["subject"] or "Lecture"
    day = row["date"] or date.today().isoformat()
    return LECTURES_DIR / f"{day}-{_slug(subject)}"


def export_markdown(lecture_id: int, dest: Path | None = None) -> Path:
    ensure_dirs()
    row = get_lecture(lecture_id)
    if row is None:
        raise ValueError(f"Lecture {lecture_id} not found")

    subject = row["subject"] or "Lecture"
    topic = row["topic"] or ""
    day = row["date"] or date.today().isoformat()
    folder = dest or lecture_dir(lecture_id)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "slides").mkdir(exist_ok=True)

    segments = list_segments(lecture_id)
    lines = [f"# {subject} — {day}", ""]
    if topic:
        lines += [f"**Тема:** {topic}", ""]
    cost = row["cost_usd"] if "cost_usd" in row.keys() else 0
    if cost:
        lines += [f"**Витрати API:** ${cost:.4f}", ""]

    for seg in segments:
        lines.append(f"## {_fmt_ts(seg['start_time'])}")
        lines.append("")
        if seg["translated_text"]:
            lines.append(seg["translated_text"].strip())
            lines.append("")
            lines.append(f"> {seg['original_text'].strip()}")
        else:
            lines.append(seg["original_text"].strip())
        lines.append("")

    slides = list_slides(lecture_id)
    if slides:
        lines += ["---", "", "# Слайди та дошка", ""]
        for slide in slides:
            stamp = _fmt_ts(slide["timestamp"])
            title = (slide["title"] or "").strip()
            lines.append(f"## {stamp}" + (f" — {title}" if title else ""))
            if slide["image_path"]:
                rel = _relative_image(folder, slide["image_path"])
                if rel:
                    lines += ["", f"![слайд]({rel})"]
            lines.append("")
            if slide["summary"]:
                lines += [slide["summary"].strip(), ""]
            for point in list_slide_points(slide["id"]):
                lines.append(f"- {point['content'].strip()}")
            if slide["formulas"]:
                lines += ["", "**Формули:**", "", slide["formulas"].strip()]
            if slide["code"]:
                lines += ["", "**Код:**", "", "```", slide["code"].strip(), "```"]
            ocr = (slide["ocr_text"] or "").strip()
            if ocr and ocr != (slide["summary"] or "").strip():
                lines += ["", "<details><summary>Розпізнаний текст</summary>", "", ocr, "", "</details>"]
            lines.append("")

    md_path = folder / "lecture.md"
    md_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    return md_path


def _relative_image(folder: Path, image_path: str) -> str | None:
    try:
        return str(Path(image_path).resolve().relative_to(folder.resolve())).replace("\\", "/")
    except Exception:
        return None
