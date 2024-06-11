import io
import logging
import numpy as np
import torch
import torch.nn.functional as F
import uvicorn
import librosa

from contextlib import asynccontextmanager
from typing import Dict
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

from .settings import _settings
from .feature_extraction.extractor import featExtractor
from .feature_extraction.network_architectures import weak_mxh64_1024


logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(asctime)s %(msecs)d %(name)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)

model: weak_mxh64_1024
extractor: featExtractor

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, extractor, device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = weak_mxh64_1024(527, F.avg_pool2d).to(device)
    extractor = featExtractor(
        model,
        [
            'layer1',
            'layer2',
            'layer4',
            'layer5',
            'layer7',
            'layer8',
            'layer10',
            'layer11',
            'layer13',
            'layer14',
            'layer16',
            'layer18'
        ]
    )
    yield

app = FastAPI(lifespan=lifespan)

@app.post("/encode")
async def encode(input: Dict) -> JSONResponse:
    audio = input["audio"]

    audio = librosa.util.normalize(audio)

    spectgram = librosa.power_to_db(
        librosa.feature.melspectrogram(y=audio, sr=44100, n_fft=1024, hop_length=512, n_mels=128)).T
    spectgram = np.concatenate([spectgram, np.zeros((176 - spectgram.shape[0] % 176, 128))])  # zero padding

    spectgram = np.concatenate([spectgram, np.zeros((87 - spectgram.shape[0] % 87, 128))])  # zero padding
    spectgram = np.reshape(spectgram, (spectgram.shape[0] // 87, 1, 87, 128))  # shape needed from pytorch
    spectgram = np.concatenate([spectgram[:-1], spectgram[1:]], axis=2).astype(np.float32)
    spectgram = torch.from_numpy(spectgram).to(device)

    features = []
    for i in range(spectgram.shape[0]//128 + 1):
        batch = spectgram[i*128:(i+1)*128]
        if batch.shape[0] > 0:
            features.append(extractor(batch).data.cpu().numpy())
    features = np.mean(features, axis=1)
    return JSONResponse(
        content=features.tolist()
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)