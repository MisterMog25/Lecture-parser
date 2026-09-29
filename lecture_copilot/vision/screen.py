from __future__ import annotations

from dataclasses import dataclass

from PIL import Image


@dataclass
class Monitor:
    index: int
    name: str
    width: int
    height: int


def _mss_class():  # type: ignore[no-untyped-def]
    import mss

    return getattr(mss, "MSS", None) or mss.mss


def list_monitors() -> list[Monitor]:
    try:
        cls = _mss_class()
    except ImportError:
        return []
    with cls() as sct:
        monitors: list[Monitor] = []
        for index, mon in enumerate(sct.monitors[1:], start=1):
            monitors.append(
                Monitor(
                    index=index,
                    name=str(mon.get("name") or f"Monitor {index}"),
                    width=int(mon["width"]),
                    height=int(mon["height"]),
                )
            )
    return monitors


class ScreenCapture:
    """Grabs frames from a monitor with mss (fast, no external binary)."""

    def __init__(self, monitor_index: int = 1):
        cls = _mss_class()
        self._sct = cls()
        monitors = self._sct.monitors
        if monitor_index >= len(monitors):
            monitor_index = 1
        self.monitor_index = monitor_index
        self._monitor = monitors[monitor_index]

    def grab(self) -> Image.Image:
        raw = self._sct.grab(self._monitor)
        return Image.frombytes("RGB", raw.size, raw.bgra, "raw", "BGRX")

    def close(self) -> None:
        self._sct.close()
