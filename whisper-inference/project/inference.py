import io
import logging
import numpy as np
import torch
import uvicorn
import librosa

from contextlib import asynccontextmanager
from typing import Dict
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

from .settings import _settings


logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(asctime)s %(msecs)d %(name)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)

model: AutoModelForSpeechSeq2Seq
processor: AutoProcessor

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, processor, device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForSpeechSeq2Seq.from_pretrained(_settings.model_name).half().to(device)
    processor = AutoProcessor.from_pretrained(_settings.model_name)
    yield

app = FastAPI(lifespan=lifespan)

@app.post("/encode")
async def encode(input: Dict) -> JSONResponse:
    audio = input["audio"]

    inp = torch.tensor(processor(audio, sampling_rate=_settings.model_sr)['input_features'])
    
    with torch.no_grad():
        logits = model.model.encoder(inp.half().to(device)).last_hidden_state

    return JSONResponse(content=logits.mean(dim=-1).tolist())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)