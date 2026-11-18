from __future__ import annotations
import time
from contextlib import contextmanager


class StageTimer:
    def __init__(self) -> None:
        self.stages: dict[str, float] = {}
        self._t0 = time.perf_counter()

    @contextmanager
    def stage(self, name: str):
        t = time.perf_counter()
        try:
            yield
        finally:
            self.stages[name] = self.stages.get(name, 0.0) + time.perf_counter() - t

    @property
    def total(self) -> float:
        return time.perf_counter() - self._t0

    def report(self, audio_seconds: float) -> str:
        lines = [f"  {k:<10}{v:8.2f} s" for k, v in self.stages.items()]
        lines.append(f"  {'TOTAL':<10}{self.total:8.2f} s")
        if audio_seconds > 0:
            asr = self.stages.get("asr", 0.0)
            lines.append(
                f"  audio {audio_seconds:.1f}s | RTF total {self.total / audio_seconds:.3f}"
                f" | RTF asr {asr / audio_seconds:.3f}"
                f" | {audio_seconds / max(self.total, 1e-9):.1f}x realtime")
        return "\n".join(lines)