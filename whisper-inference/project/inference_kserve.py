
import logging
import numpy as np
import torch
import kserve

from typing import Dict
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

MAX_SEGMENT_DURATION = 30

class WhisperModel(kserve.Model):
    def __init__(
        self,
        name: str,
        model_name: str
    ):
        super().__init__(name)
        self.name = name
        self.ready = False
        self.model = None
        self.model_name = model_name
        self.model_sr = 16000
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.warning(f"Running on device: {self.device}")
        
    def load(self) -> None:
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(self.model_name)
        self.model = self.model.model.encoder.half().to(self.device)
        self.processor = AutoProcessor.from_pretrained(self.model_name, device_map=self.device)
        self.ready = True

    def predict(self, request_data: Dict, request_headers: Dict) -> Dict:
        audio = request_data["audio"]

        inp = torch.tensor(self.processor(audio, sampling_rate=self.model_sr)['input_features']).half().to(self.device)
        
        with torch.no_grad():
            logits = self.model(inp).last_hidden_state
        
        part = float(len(audio)) / (MAX_SEGMENT_DURATION * self.model_sr)
        thr = int(logits.shape[1] * part)
        logits = logits[:, :thr].mean(dim=1)

        return {
            "embedding": logits.tolist()
        }

if __name__ == "__main__":
    model = WhisperModel(name="encoder", model_name="antony66/whisper-large-v3-russian")
    model.load()
    kserve.ModelServer(http_port=8001, enable_grpc=False).start([model])