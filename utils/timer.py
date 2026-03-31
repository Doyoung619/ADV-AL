import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional


def format_seconds(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:05.2f}"
    return f"{m:02d}:{s:05.2f}"


class Timer:
    def __init__(self):
        self._start: Optional[float] = None
        self.elapsed: float = 0.0

    def start(self):
        self._start = time.perf_counter()
        return self

    def stop(self) -> float:
        if self._start is None:
            return self.elapsed
        self.elapsed = time.perf_counter() - self._start
        self._start = None
        return self.elapsed

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0.0
        self.count = 0
        self.avg = 0.0

    def update(self, value: float, n: int = 1):
        self.sum += float(value) * n
        self.count += n
        self.avg = self.sum / max(1, self.count)


@dataclass
class TimingRecorder:
    rows: List[Dict] = field(default_factory=list)

    def add(self, event: str, duration: float, round_idx: Optional[int] = None, extra: Optional[Dict] = None):
        row = {"event": event, "duration_sec": float(duration), "round": round_idx}
        if extra:
            row.update(extra)
        self.rows.append(row)

