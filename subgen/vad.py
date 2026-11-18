from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .audio import SAMPLE_RATE


@dataclass(frozen=True)
class Segment:
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class Chunk:
    """One ASR call: contiguous audio spanning these speech regions."""
    segments: list[Segment]

    @property
    def start(self) -> float:
        return self.segments[0].start

    @property
    def end(self) -> float:
        return self.segments[-1].end


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    d = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(d == 1).tolist(), np.flatnonzero(d == -1).tolist()))


def speech_segments(audio: np.ndarray, sr: int = SAMPLE_RATE, frame_ms: int = 20,
                    margin_db: float = 9.0, min_speech: float = 0.20,
                    min_silence: float = 0.25, pad: float = 0.12,
                    max_segment: float = 12.0) -> list[Segment]:
    """Dependency-free adaptive energy VAD. Good on clean speech/dialogue."""
    hop = int(sr * frame_ms / 1000)
    if audio.size < hop:
        return []

    frames = audio[: (audio.size // hop) * hop].reshape(-1, hop)
    rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1))
    db = 20.0 * np.log10(rms + 1e-9)
    db = np.convolve(db, np.ones(3) / 3.0, mode="same")          # 60 ms smoothing

    noise, loud = float(np.percentile(db, 10)), float(np.percentile(db, 90))
    thr = min(max(noise + margin_db, -60.0), loud - 6.0)
    mask = db > thr

    def fr(sec: float) -> int:
        return max(1, int(round(sec * 1000 / frame_ms)))

    merged: list[list[int]] = []                                  # bridge short pauses
    for a, b in _runs(mask):
        if merged and a - merged[-1][1] < fr(min_silence):
            merged[-1][1] = b
        else:
            merged.append([a, b])
    merged = [[a, b] for a, b in merged if b - a >= fr(min_speech)]

    cut: list[tuple[int, int]] = []                               # split at quietest point
    lim = max(2, fr(max_segment))

    def split(a: int, b: int) -> None:
        if b - a <= lim:
            cut.append((a, b))
            return
        lo, hi = a + lim // 2, b - lim // 2
        c = (a + b) // 2 if hi <= lo else lo + int(np.argmin(db[lo:hi]))
        split(a, c)
        split(c, b)

    for a, b in merged:
        split(a, b)

    out: list[Segment] = []                                       # pad, clamp, merge
    p = int(pad * sr)
    for a, b in cut:
        s = max(0, a * hop - p) / sr
        e = min(audio.size, b * hop + p) / sr
        if out and s <= out[-1].end:
            out[-1] = Segment(out[-1].start, max(out[-1].end, e))
        else:
            out.append(Segment(s, e))
    return out


def group_segments(segments: list[Segment], max_span: float = 20.0,
                   max_gap: float = 1.0) -> list[Chunk]:
    chunks: list[Chunk] = []
    cur: list[Segment] = []
    for s in segments:
        if cur and (s.start - cur[-1].end > max_gap or s.end - cur[0].start > max_span):
            chunks.append(Chunk(cur))
            cur = []
        cur.append(s)
    if cur:
        chunks.append(Chunk(cur))
    return chunks