"""Progress reporting for the auto-process pipeline."""

import time
from dataclasses import dataclass, field
from typing import Callable, Optional


@dataclass
class ProgressUpdate:
    phase: str       # "download", "assessment", "decision", "execution", "step", "done"
    message: str     # Human-readable status
    percent: float   # 0.0-1.0, or -1 for indeterminate
    detail: dict = field(default_factory=dict)


ProgressCallback = Callable[[ProgressUpdate], None]


def format_time(seconds):
    """Format seconds into a human-readable string."""
    if seconds < 0:
        return "estimating..."
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes = seconds / 60
    if minutes < 60:
        secs = int(seconds) % 60
        return f"{int(minutes)}m {secs}s"
    hours = int(minutes) // 60
    mins = int(minutes) % 60
    return f"{hours}h {mins}m"


def print_progress(update: ProgressUpdate):
    """Default callback — formats step updates for terminal."""
    if update.phase == "step":
        d = update.detail
        elapsed_str = format_time(d["elapsed"])
        if d["remaining"] >= 0:
            remaining_str = f"~{format_time(d['remaining'])} remaining"
        else:
            remaining_str = "estimating..."
        print(f"[{d['step']}/{d['total']}] {update.message}... "
              f"({elapsed_str} elapsed, {remaining_str})")
    elif update.message.strip():
        print(update.message)


class StepTracker:
    """Tracks pipeline steps with timing and progress estimation."""

    def __init__(self, on_progress=None):
        self.steps: list = []
        self.on_progress = on_progress or print_progress
        self.start_time: Optional[float] = None

    def add_step(self, name: str, weight: float = 1.0):
        """Add a step to the pipeline."""
        self.steps.append({
            "name": name,
            "weight": weight,
            "status": "pending",
            "start": None,
            "end": None,
        })

    def skip(self, name: str):
        """Mark a pending step as skipped."""
        for s in self.steps:
            if s["name"] == name and s["status"] == "pending":
                s["status"] = "skipped"

    def begin(self, name: str):
        """Start a step."""
        if self.start_time is None:
            self.start_time = time.time()
        for s in self.steps:
            if s["name"] == name and s["status"] == "pending":
                s["status"] = "active"
                s["start"] = time.time()
                self._emit()
                return

    def finish(self, name: str):
        """Complete the active step."""
        for s in self.steps:
            if s["name"] == name and s["status"] == "active":
                s["status"] = "done"
                s["end"] = time.time()
                return

    def done(self):
        """Emit a final 100% progress update."""
        elapsed = time.time() - self.start_time if self.start_time else 0
        active = self._active_steps()
        total = len(active)
        self.on_progress(ProgressUpdate(
            phase="step",
            message="Complete",
            percent=1.0,
            detail={
                "step": total,
                "total": total,
                "elapsed": elapsed,
                "remaining": 0,
            },
        ))

    def _active_steps(self):
        return [s for s in self.steps if s["status"] != "skipped"]

    def _emit(self):
        active = self._active_steps()
        current = None
        step_num = 0
        for i, s in enumerate(active):
            if s["status"] == "active":
                current = s
                step_num = i + 1
                break

        if not current:
            return

        total = len(active)
        elapsed = time.time() - self.start_time if self.start_time else 0

        done_weight = sum(s["weight"] for s in active if s["status"] == "done")
        remaining_weight = sum(
            s["weight"] for s in active if s["status"] in ("pending", "active")
        )
        total_weight = done_weight + remaining_weight
        fraction = done_weight / total_weight if total_weight > 0 else 0

        if done_weight > 0:
            est_remaining = elapsed / done_weight * remaining_weight
        else:
            est_remaining = -1

        self.on_progress(ProgressUpdate(
            phase="step",
            message=current["name"],
            percent=fraction,
            detail={
                "step": step_num,
                "total": total,
                "elapsed": elapsed,
                "remaining": est_remaining,
            },
        ))
