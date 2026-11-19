from __future__ import annotations
import ctypes
import numpy as np
from ..align import Word
from ..audio import SAMPLE_RATE


class VoiceEngine:
    """Moonshine Voice: native VAD + attention word timings, whole-file call."""
    has_timestamps = True
    whole_file = True          # cli must NOT chunk, and must NOT offset times

    def __init__(self, arch: str = "base", language: str = "en",
                 word_timestamps: bool = True, vad_threshold: float = 0.5,
                 max_segment: float = 15.0, **_):
        from moonshine_voice import (Transcriber, get_model_for_language,
                                     string_to_model_arch)

        model_arch = string_to_model_arch(arch)   # tiny | base | *-streaming
        model_path, model_arch = get_model_for_language(
            language, model_arch, include_word_timestamps=word_timestamps)

        opts = {
            "return_audio_data": "false",         # else every line's PCM is copied to Python
            "vad_threshold": str(vad_threshold),
            "vad_max_segment_duration": str(max_segment),
        }
        if word_timestamps:
            opts["word_timestamps"] = "true"

        self.t = Transcriber(model_path=model_path, model_arch=model_arch,
                             options=opts)
        self.name = f"voice/{arch}"

    def transcribe(self, audio: np.ndarray) -> list[Word]:
        if audio.size < 1600:
            return []
        # Zero-copy: the binding's own method does (c_float*n)(*audio.tolist()),
        # which costs ~4 s and ~700 MB for 24 min. Call the C API directly.
        from moonshine_voice.moonshine_api import TranscriptC
        from moonshine_voice.errors import check_error

        flat = np.ascontiguousarray(audio, dtype=np.float32).ravel()
        ptr = flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        out = ctypes.POINTER(TranscriptC)()
        err = self.t._lib.moonshine_transcribe_without_streaming(
            self.t._handle, ptr, flat.size, SAMPLE_RATE, 0, ctypes.byref(out))
        check_error(err)
        tr = self.t._parse_transcript(out)

        words: list[Word] = []
        for line in tr.lines:
            if line.words:
                words += [Word(w.word.strip(), w.start, w.end)
                          for w in line.words if w.word.strip()]
            elif line.text.strip():               # no alignment model present
                words.append(Word(line.text.strip(), line.start_time,
                                  line.start_time + line.duration))
        return words

    def close(self):
        self.t.close()