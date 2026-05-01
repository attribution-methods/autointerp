"""Session events emitted by the autointerp agent runtime."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class OpType(str, Enum):
    USER_INPUT = "user_input"
    APPROVAL = "approval"
    INTERRUPT = "interrupt"
    SHUTDOWN = "shutdown"


@dataclass
class Event:
    event_type: str
    data: dict[str, Any] | None = None
