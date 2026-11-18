from __future__ import annotations
from dataclasses import dataclass
from .align import Word


@dataclass
class Cue:
    start: float
    end: float
    text: str


def wrap(text: str, width: int = 42, max_lines: int = 2) -> str:
    """Balanced 2-line wrap (easier to read than greedy fill)."""
    if max_lines <= 1 or len(text) <= width:
        return text
    words = text.split()
    best = None
    for i in range(1, len(words)):
        a, b = " ".join(words[:i]), " ".join(words[i:])
        if len(a) <= width and len(b) <= width:
            score = abs(len(a) - len(b))
            if best is None or score < best[0]:
                best = (score, a + "\n" + b)
    if best:
        return best[1]
    lines, cur = [], ""
    for w in words:
        if cur and len(cur) + 1 + len(w) > width:
            lines.append(cur)
            cur = w
        else:
            cur = (cur + " " + w).strip()
    if cur:
        lines.append(cur)
    return "\n".join(lines)          # never drop words


def words_to_cues(words: list[Word], max_chars: int = 42, max_lines: int = 2,
                  max_cue: float = 6.0, max_gap: float = 0.7,
                  min_sentence: float = 1.2) -> list[Cue]:
    budget = max_chars * max_lines - (max_lines - 1)
    cues: list[Cue] = []
    buf: list[Word] = []

    def flush() -> None:
        if not buf:
            return
        text = " ".join(w.text for w in buf)
        text = text[0].upper() + text[1:]
        cues.append(Cue(buf[0].start, buf[-1].end, wrap(text, max_chars, max_lines)))
        buf.clear()

    for w in words:
        if buf:
            length = sum(len(x.text) + 1 for x in buf) + len(w.text)
            if (length > budget
                    or (w.end - buf[0].start) > max_cue
                    or (w.start - buf[-1].end) > max_gap):
                flush()
        buf.append(w)
        span = w.end - buf[0].start
        if (w.text[-1:] in ".?!" and span >= min_sentence) or (
                w.text[-1:] in ",;:" and sum(len(x.text) + 1 for x in buf) > max_chars):
            flush()
    flush()
    return cues


def enforce_timing(cues: list[Cue], min_cue: float = 0.9, min_gap: float = 0.04,
                   audio_duration: float | None = None) -> list[Cue]:
    for i, c in enumerate(cues):
        nxt = cues[i + 1].start - min_gap if i + 1 < len(cues) else float("inf")
        c.start = max(0.0, c.start)
        if c.end < c.start + min_cue:
            c.end = c.start + min_cue
        if c.end > nxt:
            c.end = max(c.start + 0.30, nxt)
        if audio_duration is not None:
            c.end = min(c.end, audio_duration)
    return cues


def _ts(t: float, sep: str) -> str:
    ms = max(0, int(round(t * 1000)))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def to_srt(cues: list[Cue]) -> str:
    return "\n".join(
        f"{i}\n{_ts(c.start, ',')} --> {_ts(c.end, ',')}\n{c.text}\n"
        for i, c in enumerate(cues, 1))


def to_vtt(cues: list[Cue]) -> str:
    body = "\n".join(
        f"{i}\n{_ts(c.start, '.')} --> {_ts(c.end, '.')}\n{c.text}\n"
        for i, c in enumerate(cues, 1))
    return "WEBVTT\n\n" + body


def to_txt(cues: list[Cue]) -> str:
    return "\n".join(c.text.replace("\n", " ") for c in cues)