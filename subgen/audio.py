from __future__ import annotations
import json
import shutil
import subprocess
import numpy as np

SAMPLE_RATE = 16_000


def _require(tool: str) -> str:
    path = shutil.which(tool)
    if path is None:
        raise RuntimeError(f"{tool} is required: sudo apt install ffmpeg")
    return path


def probe_audio(path: str) -> list[dict]:
    """Audio streams in file order. 'ordinal' is the N in ffmpeg's a:N selector."""
    cmd = [_require("ffprobe"), "-v", "error", "-select_streams", "a",
           "-show_entries",
           "stream=index,codec_name,channels,sample_rate,duration"
           ":stream_tags=language,title,handler_name"
           ":stream_disposition=default",
           "-of", "json", path]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if p.returncode != 0:
        raise RuntimeError("ffprobe failed:\n" + p.stderr.decode("utf-8", "replace")[-1500:])
    out = []
    for i, s in enumerate(json.loads(p.stdout or b"{}").get("streams", [])):
        tags = s.get("tags") or {}
        out.append({
            "ordinal": i,                       # a:N  (what ffmpeg wants)
            "index": s.get("index"),            # global stream index (what ffprobe prints)
            "codec": s.get("codec_name", "?"),
            "channels": s.get("channels", 0),
            "sample_rate": s.get("sample_rate", "?"),
            "duration": s.get("duration"),
            "language": (tags.get("language") or "und").lower(),
            "title": tags.get("title") or "",
            "handler": tags.get("handler_name") or "",
            "default": bool((s.get("disposition") or {}).get("default", 0)),
        })
    return out


def resolve_track(tracks: list[dict], spec: str | int | None) -> int:
    """None -> default-flagged track, else first. Digits -> ordinal. Else language/title."""
    if not tracks:
        raise RuntimeError("no audio streams found")
    if spec is None or (isinstance(spec, str) and not spec.strip()):
        return next((t["ordinal"] for t in tracks if t["default"]), 0)

    if isinstance(spec, int) or str(spec).strip().lstrip("+").isdigit():
        n = int(spec)
        if not 0 <= n < len(tracks):
            raise RuntimeError(f"audio track {n} out of range (0..{len(tracks) - 1})")
        return n

    low = str(spec).strip().lower()
    hits = [t["ordinal"] for t in tracks if t["language"] == low]
    if not hits:
        hits = [t["ordinal"] for t in tracks
                if low in t["title"].lower() or low in t["handler"].lower()]
    if not hits:
        avail = ", ".join(f'{t["ordinal"]}={t["language"]}' for t in tracks)
        raise RuntimeError(f"no audio track matching {spec!r} (available: {avail})")
    if len(hits) > 1:
        raise RuntimeError(f"audio track {spec!r} is ambiguous: ordinals {hits}; use an index")
    return hits[0]


def format_tracks(tracks: list[dict]) -> str:
    if not tracks:
        return "  (no audio streams)"
    lines = []
    for t in tracks:
        dur = f'{float(t["duration"]):.1f}s' if t["duration"] else "?"
        flag = " *default" if t["default"] else ""
        name = t["title"] or t["handler"] or "-"
        lines.append(f'  {t["ordinal"]}: {t["language"]:4} {t["codec"]:10} '
                     f'{t["channels"]}ch {t["sample_rate"]:>6}Hz {dur:>8}  {name}{flag}')
    return "\n".join(lines)


def load_audio(path: str, track: str | int | None = None,
               *, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Decode ONE audio track to mono float32 in [-1, 1].

    track: None (default-flagged, else first) | ordinal 0,1,... | "eng"/"mal" | title substring.
    RAM: 16000 * 4 bytes/s = 64 KB/s -> 1 h audio ~= 230 MB. Fine on 16 GB.
    """
    _require("ffmpeg")
    ordinal = resolve_track(probe_audio(path), track)
    cmd = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
           "-i", path, "-vn", "-sn", "-dn", "-map", f"0:a:{ordinal}",
           "-ac", "1", "-ar", str(sr), "-f", "f32le", "pipe:1"]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if p.returncode != 0:
        raise RuntimeError("ffmpeg failed:\n" + p.stderr.decode("utf-8", "replace")[-1500:])
    audio = np.frombuffer(p.stdout, dtype="<f4").copy()
    if audio.size == 0:
        raise RuntimeError("no audio decoded (empty track?)")
    return audio