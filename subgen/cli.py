from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path

from .align import Word, spread_words, tidy_ctc
from .audio import SAMPLE_RATE, format_tracks, load_audio, probe_audio, resolve_track
from .engines import build_engine
from .subtitles import enforce_timing, to_srt, to_txt, to_vtt, words_to_cues
from .timing import StageTimer
from .vad import group_segments, speech_segments


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="subgen", description="CPU-only SRT/WebVTT generator")
    ap.add_argument("input")
    ap.add_argument("-o", "--output", help="output basename (default: input without extension)")
    ap.add_argument("--engine", choices=("moonshine", "voice", "wav2vec2"), default="moonshine")
    ap.add_argument("--model", help="moonshine/base | moonshine/tiny | HF CTC model id")
    ap.add_argument("--audio-track", default=None, metavar="N|LANG|TITLE",
                    help="audio track for multi-audio files: ordinal (0,1,...), "
                         "language code (eng, mal), or title substring. "
                         "Default: the track flagged default, else the first.")
    ap.add_argument("--list-tracks", action="store_true",
                    help="list audio tracks and exit")
    ap.add_argument("--models-dir", help="local ONNX dir (offline / quantized)")
    ap.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--int8", action="store_true", help="dynamic int8 (wav2vec2)")
    ap.add_argument("--formats", default="srt,vtt")
    ap.add_argument("--max-chars", type=int, default=42)
    ap.add_argument("--max-lines", type=int, default=2)
    ap.add_argument("--max-cue", type=float, default=6.0)
    ap.add_argument("--min-cue", type=float, default=0.9)
    ap.add_argument("--max-chunk", type=float, default=20.0)
    ap.add_argument("--vad-margin", type=float, default=9.0)
    ap.add_argument("--dry-run", action="store_true", help="VAD only, no ASR")
    args = ap.parse_args(argv)

    timer = StageTimer()
    
    if args.list_tracks:
        print(format_tracks(probe_audio(args.input)))
        return 0

    with timer.stage("decode"):
        tracks = probe_audio(args.input)
        ordinal = resolve_track(tracks, args.audio_track)
        t = tracks[ordinal]
        audio = load_audio(args.input, ordinal)
    dur = audio.size / SAMPLE_RATE

    if len(tracks) > 1 or args.audio_track is not None:
            print(f"audio track {ordinal}: {t['language']} {t['codec']} "
                f"{t['channels']}ch {t['sample_rate']}Hz"
                + ("  (default)" if t["default"] else ""), file=sys.stderr)
            
    with timer.stage("vad"):
        segments = speech_segments(audio, margin_db=args.vad_margin)
        chunks = group_segments(segments, max_span=args.max_chunk)
    speech = sum(s.duration for s in segments)
    print(f"audio {dur:.1f}s | speech {speech:.1f}s "
          f"({100 * speech / max(dur, 1e-9):.0f}%) | {len(segments)} regions "
          f"-> {len(chunks)} chunks", file=sys.stderr)

    if args.dry_run:
        for c in chunks:
            print(f"{c.start:8.2f} -> {c.end:8.2f}  ({c.end - c.start:5.2f}s, "
                  f"{len(c.segments)} regions)")
        return 0
    if not chunks:
        print("no speech detected - try --vad-margin 6", file=sys.stderr)
        return 2

    with timer.stage("model"):
        engine = build_engine(args.engine, model=args.model, models_dir=args.models_dir,
                              threads=args.threads, int8=args.int8)

    words: list[Word] = []
    with timer.stage("asr"):
        if getattr(engine, "whole_file", False):
            words = engine.transcribe(audio)      # native VAD + alignment
            print(f"\r  whole file  words={len(words)}", end="", file=sys.stderr)
        else:
            for i, ch in enumerate(chunks, 1):
                piece = audio[int(ch.start * SAMPLE_RATE):int(ch.end * SAMPLE_RATE)]
                out = engine.transcribe(piece)
                if engine.has_timestamps:
                    words.extend(Word(w.text, w.start + ch.start, w.end + ch.start)
                                 for w in tidy_ctc(out))
                else:
                    words.extend(spread_words(out, ch.segments))
                print(f"\r  chunk {i}/{len(chunks)}  t={ch.end:7.1f}s  words={len(words)}",
                      end="", file=sys.stderr)
    print(file=sys.stderr)

    with timer.stage("cues"):
        cues = words_to_cues(words, args.max_chars, args.max_lines, args.max_cue)
        cues = enforce_timing(cues, min_cue=args.min_cue, audio_duration=dur)

    base = Path(args.output) if args.output else Path(args.input).with_suffix("")
    renderers = {"srt": to_srt, "vtt": to_vtt, "txt": to_txt}
    with timer.stage("write"):
        for fmt in (f.strip().lower() for f in args.formats.split(",")):
            if fmt not in renderers:
                print(f"skip unknown format: {fmt}", file=sys.stderr)
                continue
            path = Path(f"{base}.{fmt}")           # not with_suffix: names may contain dots
            path.write_text(renderers[fmt](cues), encoding="utf-8")
            print(f"wrote {path}  ({len(cues)} cues)", file=sys.stderr)

    print(timer.report(dur), file=sys.stderr)
    return 0