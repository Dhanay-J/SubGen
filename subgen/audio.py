from __future__ import annotations
import shutil
import subprocess
import numpy as np

SAMPLE_RATE = 16_000


def load_audio(path: str, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Decode anything ffmpeg understands into mono float32 in [-1, 1].

    RAM: 16000 * 4 bytes/s = 64 KB/s -> 1 h audio ~= 230 MB. Fine on 16 GB.
    """
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required: sudo apt install ffmpeg")
    cmd = [
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
        "-i", path, "-vn", "-sn", "-dn", "-map", "a:0",
        "-ac", "1", "-ar", str(sr), "-f", "f32le", "pipe:1",
    ]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if p.returncode != 0:
        raise RuntimeError("ffmpeg failed:\n" + p.stderr.decode("utf-8", "replace")[-1500:])
    audio = np.frombuffer(p.stdout, dtype="<f4").copy()
    if audio.size == 0:
        raise RuntimeError("no audio decoded (missing audio track?)")
    return audio