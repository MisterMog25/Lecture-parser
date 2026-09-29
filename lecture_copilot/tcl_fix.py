from __future__ import annotations

import os
import sys
from pathlib import Path


def ensure_tcl_env() -> None:
    """Python 3.13 on Windows often looks in lib/tcl8.6; the files live in tcl/tcl8.6."""
    prefix = Path(getattr(sys, "base_prefix", None) or sys.prefix)
    tcl = prefix / "tcl" / "tcl8.6"
    tk = prefix / "tcl" / "tk8.6"
    if tcl.is_dir() and not os.environ.get("TCL_LIBRARY"):
        os.environ["TCL_LIBRARY"] = str(tcl)
    if tk.is_dir() and not os.environ.get("TK_LIBRARY"):
        os.environ["TK_LIBRARY"] = str(tk)
