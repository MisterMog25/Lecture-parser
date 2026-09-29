"""CLI: python -m lecture_copilot.export <lecture_id> | --latest | --all"""

from __future__ import annotations

from lecture_copilot.storage.ai_export import main

if __name__ == "__main__":
    raise SystemExit(main())
