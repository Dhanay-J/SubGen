from __future__ import annotations
import numpy as np
from ..align import Word
from ..audio import SAMPLE_RATE


class Wav2Vec2Engine:
    """facebook/wav2vec2 CTC. Real frame-level word timestamps, no punctuation."""
    has_timestamps = True

    def __init__(self, model_id: str = "facebook/wav2vec2-base-960h",
                 threads: int = 4, int8: bool = False):
        import torch
        from transformers import AutoModelForCTC, AutoProcessor, AutoTokenizer

        torch.set_num_threads(max(1, threads))
        torch.set_grad_enabled(False)
        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(model_id)
        # slow tokenizer guarantees output_word_offsets support
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
        model = AutoModelForCTC.from_pretrained(model_id).eval()
        self.step = model.config.inputs_to_logits_ratio / SAMPLE_RATE   # 320/16000 = 20 ms
        if int8:
            model = torch.ao.quantization.quantize_dynamic(
                model, {torch.nn.Linear}, dtype=torch.qint8)
        self.model = model
        self.name = model_id

    def transcribe(self, audio: np.ndarray) -> list[Word]:
        torch = self.torch
        if audio.size < 1600:
            return []
        inputs = self.processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt")
        with torch.inference_mode():
            logits = self.model(**inputs).logits
        ids = logits.argmax(dim=-1)
        res = self.tokenizer.batch_decode(ids, output_word_offsets=True)
        return [Word(d["word"], d["start_offset"] * self.step, d["end_offset"] * self.step)
                for d in res.word_offsets[0]]