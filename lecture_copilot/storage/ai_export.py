"""AI-first export: the machine-facing artefacts an LLM/agent consumes.

- ``lecture.ai.json``  canonical structured bundle (single source of truth)
- ``lecture.pack.md``  compact, cleaned markdown (cheap to ingest)
- ``manifest.json``    index of every asset + how to use them
- ``moodle/``          drop-zone for the teacher's own slides (PDF/images)

A deterministic cleaning pass runs first, so the deck agent gets signal, not noise.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import date
from pathlib import Path

from lecture_copilot.config import ensure_dirs
from lecture_copilot.storage.db import (
    get_lecture,
    list_lectures,
    list_segments,
    list_slide_points,
    list_slides,
)
from lecture_copilot.storage.markdown import lecture_dir

BUNDLE_SCHEMA = "lecture-copilot/bundle/v1"
PACK_SCHEMA = "lecture-copilot/pack/v1"

# ASR hallucinations / boilerplate that Whisper loves to invent.
_JUNK_PATTERNS = [
    re.compile(r"^\s*[\W_]*\s*$"),
    re.compile(
        r"(?i)\b(продолжение следует|субтитры|редактор субтитров|"
        r"thanks? for watching|subscribe|like and subscribe|"
        r"підпишись|підписуйся|підпишіться|дякую за перегляд)\b"
    ),
    re.compile(r"^\s*[\[(]?\s*(музика|music|аплодисменти|applause|сміх|laughter)\s*[\])]?\s*$", re.I),
    re.compile(r"^\s*[♪♫#*\-–—]{1,}\s*$"),
]

_WORD = re.compile(r"[0-9a-zа-яіїєґäöüßčšžťďňĺŕýáéíóú_+\-’'ʼ]+", re.IGNORECASE)

_STOP = {
    "the", "a", "an", "and", "or", "but", "if", "then", "so", "of", "to", "in", "on", "at",
    "is", "are", "was", "were", "be", "been", "it", "this", "that", "these", "those", "we",
    "you", "i", "he", "she", "they", "as", "for", "with", "by", "from", "not", "no", "yes",
    "do", "does", "did", "can", "will", "would", "should", "have", "has", "had", "there",
    "і", "й", "та", "а", "але", "чи", "що", "це", "цей", "ця", "ці", "той", "так", "не",
    "ні", "ми", "ви", "він", "вона", "вони", "я", "ти", "як", "то", "ще", "вже", "буде",
    "було", "є", "для", "на", "в", "у", "з", "із", "до", "від", "по", "за", "про", "при",
    "який", "яка", "яке", "які", "можна", "треба", "дуже", "ось", "тут", "там",
    "a", "na", "do", "sa", "že", "je", "to", "v", "s", "z", "o", "k", "po", "za", "pre",
}


def _norm(text: str) -> str:
    return " ".join((text or "").lower().split())


def _tokens(text: str) -> set[str]:
    return set(_WORD.findall((text or "").lower()))


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _is_junk(text: str) -> bool:
    t = (text or "").strip()
    if len(t) < 2:
        return True
    return any(p.search(t) for p in _JUNK_PATTERNS)


def clean_segments(rows, dedup_ratio: float = 0.8) -> list[dict]:
    """Collapse repeats / drop ASR junk. Returns plain dicts."""
    out: list[dict] = []
    for row in rows:
        original = " ".join((row["original_text"] or "").split())
        translated = " ".join((row["translated_text"] or "").split())
        if _is_junk(original) and _is_junk(translated):
            continue
        canonical = translated or original
        toks = _tokens(canonical)
        if out and _jaccard(toks, out[-1]["_tokens"]) >= dedup_ratio:
            # near-duplicate consecutive ASR output -> extend the previous span
            out[-1]["end"] = float(row["end_time"])
            if len(canonical) > len(out[-1]["text"]):
                out[-1]["text"] = canonical
                out[-1]["original"] = original or out[-1]["original"]
                out[-1]["translated"] = translated or None
                out[-1]["_tokens"] = toks
            continue
        out.append(
            {
                "start": round(float(row["start_time"]), 2),
                "end": round(float(row["end_time"]), 2),
                "text": canonical,
                "original": original or None,
                "translated": translated or None,
                "_tokens": toks,
            }
        )
    for seg in out:
        seg.pop("_tokens", None)
    return out


def _rel(folder: Path, path: str | None) -> str | None:
    if not path:
        return None
    try:
        return str(Path(path).resolve().relative_to(folder.resolve())).replace("\\", "/")
    except Exception:
        return None


def _kind(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}:
        return "image"
    if suffix == ".pdf":
        return "pdf"
    if suffix in {".md", ".txt"}:
        return "text"
    return "other"


def _key_terms(bundle_text: str, limit: int = 40) -> list[str]:
    counts: dict[str, int] = {}
    for tok in _WORD.findall(bundle_text.lower()):
        if len(tok) < 4 or tok in _STOP or tok.isdigit():
            continue
        counts[tok] = counts.get(tok, 0) + 1
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return [w for w, _ in ranked[:limit]]


def _slide_body(slide: dict) -> str:
    return _norm(" ".join([slide.get("ocr_text", ""), slide.get("summary", ""), " ".join(slide.get("points", []))]))


def _slide_weight(slide: dict) -> int:
    return len(slide.get("summary", "")) + sum(len(p) for p in slide.get("points", [])) + len(slide.get("ocr_text", ""))


def clean_slides(slides: list[dict], dedup_ratio: float = 0.85) -> list[dict]:
    """Collapse consecutive re-captures of the same slide (cursor jitter, re-render).

    Keeps the richest version and records how many times it was seen.
    """
    out: list[dict] = []
    bodies: list[set[str]] = []
    for slide in slides:
        slide = dict(slide)
        body = _slide_body(slide)
        toks = _tokens(body)
        if out:
            prev = out[-1]
            same_title = bool(_norm(slide["title"])) and _norm(slide["title"]) == _norm(prev["title"])
            similar = _jaccard(toks, bodies[-1]) >= dedup_ratio
            if same_title or similar:
                prev["repeats"] = prev.get("repeats", 1) + 1
                if _slide_weight(slide) > _slide_weight(prev):
                    repeats = prev["repeats"]
                    slide["repeats"] = repeats
                    out[-1] = slide
                    bodies[-1] = toks
                continue
        slide["repeats"] = 1
        out.append(slide)
        bodies.append(toks)
    return out


def build_bundle(lecture_id: int) -> tuple[dict, Path]:
    """Build the canonical AI bundle. Returns (bundle, lecture_folder)."""
    row = get_lecture(lecture_id)
    if row is None:
        raise ValueError(f"Lecture {lecture_id} not found")
    folder = lecture_dir(lecture_id)
    segments = clean_segments(list_segments(lecture_id))

    slides = []
    for slide in list_slides(lecture_id):
        slides.append(
            {
                "t": round(float(slide["timestamp"]), 2),
                "title": (slide["title"] or "").strip(),
                "summary": (slide["summary"] or "").strip(),
                "points": [p["content"].strip() for p in list_slide_points(slide["id"]) if p["content"].strip()],
                "formulas": (slide["formulas"] or "").strip(),
                "code": (slide["code"] or "").strip(),
                "ocr_text": (slide["ocr_text"] or "").strip(),
                "image": _rel(folder, slide["image_path"]),
                "important": bool(slide["important"]),
            }
        )
    slides = clean_slides(slides)

    moodle = []
    moodle_dir = folder / "moodle"
    if moodle_dir.exists():
        for path in sorted(p for p in moodle_dir.rglob("*") if p.is_file() and not p.name.startswith(".")):
            if path.name == "README.txt":
                continue
            moodle.append(
                {
                    "path": str(path.relative_to(folder)).replace("\\", "/"),
                    "kind": _kind(path),
                    "bytes": path.stat().st_size,
                }
            )

    duration = max([s["end"] for s in segments] + [s["t"] for s in slides] + [0.0])
    all_text = " ".join(s["text"] for s in segments)

    bundle = {
        "schema": BUNDLE_SCHEMA,
        "lecture": {
            "id": int(row["id"]),
            "title": row["title"],
            "subject": row["subject"],
            "topic": row["topic"],
            "date": row["date"] or date.today().isoformat(),
            "source_language": row["source_language"],
            "target_language": row["target_language"],
            "cost_usd": round(float(row["cost_usd"] or 0), 4),
        },
        "stats": {
            "duration_sec": round(duration, 1),
            "segments": len(segments),
            "slides": len(slides),
            "moodle_files": len(moodle),
            "words": len(all_text.split()),
        },
        "key_terms": _key_terms(all_text),
        "segments": segments,
        "slides": slides,
        "moodle": moodle,
    }
    return bundle, folder


def render_pack(bundle: dict) -> str:
    lec = bundle["lecture"]
    stats = bundle["stats"]
    lines = [
        "---",
        f"schema: {PACK_SCHEMA}",
        f"subject: {lec['subject']}",
        f"topic: {lec['topic']}",
        f"date: {lec['date']}",
        f"language: {lec['source_language']} -> {lec['target_language']}",
        f"duration_min: {round(stats['duration_sec'] / 60, 1)}",
        f"segments: {stats['segments']}",
        f"slides: {stats['slides']}",
        f"moodle_files: {stats['moodle_files']}",
        "---",
        "",
        f"# {lec['subject']} — {lec['topic'] or lec['title']} ({lec['date']})",
        "",
        "> Машинний конспект лекції. Транскрипт очищено від повторів і шуму ASR.",
        "> Використовуй `segments` як зміст лекції, `slides` як підтвердження того,",
        "> що показували на екрані, `moodle` як слайди викладача (пріоритетне джерело візуалів).",
        "",
    ]

    if bundle["key_terms"]:
        lines += ["## Ключові терміни", "", ", ".join(bundle["key_terms"]), ""]

    lines += ["## Транскрипт (очищений)", ""]
    for seg in bundle["segments"]:
        stamp = _fmt(seg["start"])
        text = seg["text"]
        lines.append(f"[{stamp}] {text}")
        if seg.get("original") and seg.get("translated") and seg["original"] != seg["translated"]:
            lines.append(f"    src: {seg['original']}")
    lines.append("")

    if bundle["slides"]:
        lines += ["## Екранні слайди (calibration)", ""]
        for slide in bundle["slides"]:
            head = f"### [{_fmt(slide['t'])}]" + (f" {slide['title']}" if slide["title"] else "")
            lines.append(head)
            if slide["image"]:
                lines.append(f"![slide]({slide['image']})")
            if slide["summary"]:
                lines.append(slide["summary"])
            for point in slide["points"]:
                lines.append(f"- {point}")
            if slide["formulas"]:
                lines += ["Формули:", "```", slide["formulas"], "```"]
            if slide["code"]:
                lines += ["Код:", "```", slide["code"], "```"]
            if slide["ocr_text"]:
                lines += ["<details><summary>OCR</summary>", "", slide["ocr_text"], "", "</details>"]
            lines.append("")

    if bundle["moodle"]:
        lines += ["## Матеріали Moodle (слайди викладача)", ""]
        for item in bundle["moodle"]:
            lines.append(f"- `{item['path']}` ({item['kind']})")
        lines.append("")

    return "\n".join(lines).strip() + "\n"


def render_manifest(bundle: dict, files: dict) -> str:
    manifest = {
        "schema": BUNDLE_SCHEMA,
        "lecture": bundle["lecture"],
        "stats": bundle["stats"],
        "files": files,
        "assets": {
            "slides": [s["image"] for s in bundle["slides"] if s["image"]],
            "moodle": [m["path"] for m in bundle["moodle"]],
        },
        "consumer_hint": {
            "task": "build a teaching deck from this lecture",
            "prefer": "moodle images over screen captures for visuals",
            "text": "segments[].text is the cleaned transcript; translated wins over original",
            "images_are_relative_to": "this file's directory",
            "math": "formulas/code are verbatim; render them as math/code, do not paraphrase",
        },
    }
    return json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"


def _write_moodle_readme(moodle_dir: Path) -> None:
    readme = moodle_dir / "README.txt"
    if readme.exists():
        return
    readme.write_text(
        "MOODLE DROP-ZONE\n"
        "===============\n\n"
        "Скинь сюди слайди викладача з Moodle — PDF або картинки (png/jpg).\n"
        "Вони підуть у manifest.json як 'moodle' assets, і агент lecture-deck\n"
        "використає їх як ПРІОРИТЕТНЕ джерело візуалів (краще за скріншоти екрана).\n\n"
        "Після додавання файлів переекспортуй бандл:\n"
        "  python -m lecture_copilot.storage.ai_export <lecture_id>\n",
        encoding="utf-8",
    )


def _fmt(seconds: float) -> str:
    total = int(max(0, seconds))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def export_ai(lecture_id: int, dest: Path | None = None) -> dict[str, Path]:
    """Write lecture.ai.json + lecture.pack.md + manifest.json. Returns the paths."""
    ensure_dirs()
    bundle, folder = build_bundle(lecture_id)
    folder = dest or folder
    folder.mkdir(parents=True, exist_ok=True)
    moodle_dir = folder / "moodle"
    moodle_dir.mkdir(exist_ok=True)
    _write_moodle_readme(moodle_dir)

    files = {
        "json": "lecture.ai.json",
        "pack": "lecture.pack.md",
        "manifest": "manifest.json",
        "human": "lecture.md",
    }
    json_path = folder / files["json"]
    pack_path = folder / files["pack"]
    manifest_path = folder / files["manifest"]

    json_path.write_text(json.dumps(bundle, ensure_ascii=False, indent=2), encoding="utf-8")
    pack_path.write_text(render_pack(bundle), encoding="utf-8")
    manifest_path.write_text(render_manifest(bundle, files), encoding="utf-8")

    return {"json": json_path, "pack": pack_path, "manifest": manifest_path, "folder": folder}


def export_all() -> list[dict[str, Path]]:
    return [export_ai(row["id"]) for row in list_lectures()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export AI-first lecture bundles.")
    parser.add_argument("lecture_id", nargs="?", type=int, help="lecture id to export")
    parser.add_argument("--all", action="store_true", help="export every lecture")
    parser.add_argument("--latest", action="store_true", help="export the newest lecture")
    args = parser.parse_args(argv)

    if args.all:
        results = export_all()
    elif args.latest:
        rows = list_lectures()
        if not rows:
            print("No lectures found.")
            return 1
        results = [export_ai(rows[0]["id"])]
    elif args.lecture_id is not None:
        results = [export_ai(args.lecture_id)]
    else:
        parser.error("provide a lecture id, --latest or --all")

    for res in results:
        print(f"[{res['folder'].name}] {res['json'].name}, {res['pack'].name}, {res['manifest'].name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
