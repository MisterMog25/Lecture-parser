from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto


class EventKind(Enum):
    PARTIAL = auto()
    SEGMENT = auto()
    SLIDE = auto()
    STATUS = auto()
    ERROR = auto()
    LEVEL = auto()


@dataclass
class PipelineEvent:
    kind: EventKind
    original: str = ""
    translated: str = ""
    start_time: float = 0.0
    end_time: float = 0.0
    message: str = ""
    level: float = 0.0
