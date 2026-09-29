from __future__ import annotations

import os

from lecture_copilot.tcl_fix import ensure_tcl_env

ensure_tcl_env()
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from lecture_copilot.ui.app import run

if __name__ == "__main__":
    run()
