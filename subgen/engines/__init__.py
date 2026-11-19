from __future__ import annotations


def build_engine(name: str, **kw):
    if name == "moonshine":
        from .moonshine_engine import MoonshineEngine
        return MoonshineEngine(model_name=kw.get("model") or "moonshine/base",
                        models_dir=kw.get("models_dir"),
                        threads=kw.get("threads", 4))
    if name == "wav2vec2":
        from .wav2vec2_engine import Wav2Vec2Engine
        return Wav2Vec2Engine(model_id=kw.get("model") or "facebook/wav2vec2-base-960h",
                              threads=kw.get("threads", 4), int8=kw.get("int8", False))
    if name == "voice":
        from .voice_engine import VoiceEngine
        arch = (kw.get("model") or "base").split("/")[-1]
        return VoiceEngine(arch=arch)
    raise ValueError(f"unknown engine: {name}")