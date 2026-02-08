"""Progress reporting for the auto-process pipeline."""

from dataclasses import dataclass, field
from typing import Callable


@dataclass
class ProgressUpdate:
    phase: str       # "download", "assessment", "decision", "execution", "done"
    message: str     # Human-readable status
    percent: float   # 0.0-1.0, or -1 for indeterminate
    detail: dict = field(default_factory=dict)


ProgressCallback = Callable[[ProgressUpdate], None]


def print_progress(update: ProgressUpdate):
    """Default callback — preserves current CLI behavior."""
    print(update.message)
