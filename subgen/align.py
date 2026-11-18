from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .vad import Segment


@dataclass
class Word:
    text: str
    start: float
    end: float


def spread_words(text: str, segments: list[Segment]) -> list[Word]:
    """Place words on a 'speech clock' (silence removed), then map back to wall time.

    Cost model: a word takes time proportional to len(word)+1. Pauses between
    regions consume no words, so cue boundaries snap to real pauses.
    """
    tokens = text.split()
    if not tokens or not segments:
        return []
    w = np.array([len(t) + 1 for t in tokens], dtype=np.float64)
    speech = sum(s.duration for s in segments)
    if speech <= 0.0:
        return []
    edges = np.concatenate(([0.0], np.cumsum(w))) * (speech / w.sum())

    st: list[float] = []   # speech-time knots
    wt: list[float] = []   # wall-time knots

    def knot(t: float, wall: float) -> None:
        if st and t <= st[-1]:
            t = st[-1] + 1e-6
        st.append(t)
        wt.append(wall)

    acc = 0.0
    for s in segments:
        knot(acc, s.start)
        acc += s.duration
        knot(acc, s.end)

    times = np.interp(edges, np.asarray(st), np.asarray(wt))
    return [Word(t, float(times[i]), float(times[i + 1])) for i, t in enumerate(tokens)]


def tidy_ctc(words: list[Word]) -> list[Word]:
    """wav2vec2 CTC gives UPPERCASE unpunctuated text -> make it readable."""
    out = [Word(w.text.lower(), w.start, w.end) for w in words]
    for w in out:
        if w.text == "i" or w.text.startswith("i'"):
            w.text = "I" + w.text[1:]
    return out