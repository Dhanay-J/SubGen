from __future__ import annotations
import numpy as np


class MoonshineEngine:
    """Moonshine ONNX (CPU). Returns text only -> timing comes from VAD."""
    has_timestamps = False

    def __init__(self, model_name: str = "moonshine/base",
                 models_dir: str | None = None, threads: int = 4):
        import onnxruntime as ort
        from moonshine_onnx import MoonshineOnnxModel, load_tokenizer

        so = ort.SessionOptions()
        so.intra_op_num_threads = max(1, threads)
        so.inter_op_num_threads = 1
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.add_session_config_entry("session.intra_op.allow_spinning", "0")

        # dynamic shapes: don't cache per-shape memory plans, don't hoard an arena
        so.enable_mem_pattern = False
        so.enable_cpu_mem_arena = False
        
        # MoonshineOnnxModel builds its sessions with no SessionOptions.
        # Wrap the constructor for the duration of __init__ so the first
        # (and only) load already has our thread config.
        real_session = ort.InferenceSession

        def capped(path_or_bytes, sess_options=None, **kw):
            kw.setdefault("providers", ["CPUExecutionProvider"])
            return real_session(path_or_bytes, sess_options or so, **kw)

        ort.InferenceSession = capped
        try:
            self.model = MoonshineOnnxModel(model_name=model_name,
                                            models_dir=models_dir)
        finally:
            ort.InferenceSession = real_session

        self.tokenizer = load_tokenizer()
        self.name = model_name
        self.threads = max(1, threads)

    def transcribe(self, audio: np.ndarray) -> str:
        if audio.size < 1600:                      # <100 ms: skip, avoids junk tokens
            return ""
        x = np.asarray(audio, dtype=np.float32).reshape(1, -1)   # (batch, samples) @16k
        tokens = self.model.generate(x)
        if hasattr(self.tokenizer, "decode_batch"):
            text = self.tokenizer.decode_batch(tokens)[0]
        else:
            text = self.tokenizer.decode(tokens[0])
        return " ".join(text.split())