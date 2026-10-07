#!/usr/bin/env python3
"""Print a COMPACT index of a Lecture Copilot bundle, for planning without loading
the whole 0.5 MB pack.

Usage:
    python3 tools/bundle_index.py <lecture_folder>
Options:
    --segments N   how many transcript excerpts to print (default 0)
    --slides N     max slides to list (default all)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def fmt(seconds: float) -> str:
    total = int(max(0, seconds))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 1
    folder = Path(argv[0])
    segments_n = 0
    slides_n = 10_000
    if "--segments" in argv:
        segments_n = int(argv[argv.index("--segments") + 1])
    if "--slides" in argv:
        slides_n = int(argv[argv.index("--slides") + 1])

    bundle = json.loads((folder / "lecture.ai.json").read_text(encoding="utf-8"))
    lec, stats = bundle["lecture"], bundle["stats"]
    print(f"# {lec['subject']} — {lec['topic'] or lec['title']}  ({lec['date']})")
    print(f"language: {lec['source_language']} -> {lec['target_language']}")
    print(f"duration: {fmt(stats['duration_sec'])} | segments: {stats['segments']} | "
          f"unique slides: {stats['slides']} | moodle: {stats['moodle_files']} | words: {stats['words']}")
    print(f"key_terms: {', '.join(bundle['key_terms'][:30])}")
    print()
    print(f"## slides ({len(bundle['slides'])})  [t] reps | title — summary | image")
    for s in bundle["slides"][:slides_n]:
        title = s["title"] or s["summary"] or s["ocr_text"] or "(untitled)"
        extra = f" — {s['summary']}" if s["summary"] and s["title"] else ""
        print(f"[{fmt(s['t'])}] x{s['repeats']} | {title[:80].replace(chr(10),' ')}{extra[:80]} | {s['image']}")
    print()
    if segments_n:
        print(f"## first {segments_n} segments")
        for seg in bundle["segments"][:segments_n]:
            print(f"[{fmt(seg['start'])}] {seg['text'][:140]}")
    else:
        step = max(1, len(bundle["segments"]) // 40)
        print("## transcript sampling (every ~2.5 min)")
        for seg in bundle["segments"][::step]:
            print(f"[{fmt(seg['start'])}] {seg['text'][:120]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
