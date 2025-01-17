import logging
import requests
import numpy as np
import librosa
import uvicorn
import os
import tempfile

from contextlib import asynccontextmanager
from typing import Dict
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, FileResponse
from pymilvus import (
    FieldSchema,
    DataType
)
import os
import json
import os.path as osp
from pathlib import Path
from tqdm import tqdm
from project.utils import inference_algorithm
from project.settings import _settings
from project.storage.milvus import CustomMilvusClient

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(asctime)s %(msecs)d %(name)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)

milvus_client: CustomMilvusClient

@asynccontextmanager
async def lifespan(app: FastAPI):
    global milvus_client
    milvus_client = CustomMilvusClient(
        milvus_endpoint=_settings.milvus_endpoint,
        db_name="default"
    )
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, descrition="id", is_primary=True, auto_id=True),
        FieldSchema(name="video_id", dtype=DataType.VARCHAR, max_length=200, description="Video id"),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, description="Audio embedding", dim=_settings.embedding_dim),
        FieldSchema(name="segment_start", dtype=DataType.INT64, description="Start time of segment (seconds)"),
        FieldSchema(name="segment_duration", dtype=DataType.FLOAT, description="Length of segment (seconds)"),
    ]
    index_params = {
        "field_name": "embedding",
        "metric_type": _settings.milvus_sim_metric,
        "index_type": _settings.milvus_index_type,
        "params":{
            "nlist": _settings.milvus_index_nlist
        }
    }
    milvus_client.get_or_create_collection(
        collection_key= _settings.milvus_collection_name,
        fields=fields,
        index_params=[index_params]
    )
    yield
    
app = FastAPI(lifespan=lifespan)

@app.post("/upload")
async def upload(video: UploadFile = File(...)):
    """
    Принимает видео файл и сохраняет его в файловой системе с временным именем.
    """
    with tempfile.NamedTemporaryFile(delete=False, dir=_settings.upload_folder, suffix=".mp4") as temp_file:
        file_path = temp_file.name
        contents = await video.read()
        temp_file.write(contents)
        logger.warning(temp_file.name)
        analog_info, upload_info = inference_algorithm(file_path, _settings.database_videos_path, _settings.model_sr)

        logger.warning(
            f"Upload Info:\n{upload_info}"
        )
        logger.warning(
            f"Analog Info:\n{analog_info}"
        )

        return JSONResponse(
            content={
                "message": "Видео успешно загружено!",
                "analog_info": analog_info,
                "upload_info": upload_info
            },
            status_code=200
        )

@app.post("/upload/milvus")
async def upload_milvus(video: UploadFile = File(...)) -> JSONResponse:
    with tempfile.NamedTemporaryFile(delete=False, dir=_settings.upload_folder, suffix=".mp4") as temp_file:
        file_path = temp_file.name
        contents = await video.read()
        temp_file.write(contents)

        audio, _ = librosa.load(file_path, sr=_settings.model_sr)

        segment_duration= input["segment_duration"]
        segment_shift= input["segment_shift"]
        collection_name = input["colllection_name"]

        assert segment_shift >= 0
        assert segment_shift <= segment_duration


        for i in tqdm(list(range(0, int(len(audio) - _settings.model_sr * (segment_duration - segment_shift - 1)), _settings.model_sr))):
            segment = audio[max(0, int(i-segment_shift*_settings.model_sr)):int(i+_settings.model_sr * (segment_duration - segment_shift))]

            embedding = requests.post(
                _settings.embedder_endpoint,
                json = {
                    "audio": segment.tolist()
                }
            ).json()["embedding"]
            
            milvus_client.insert_to_collection(
                collection_key=collection_name,
                insert_data=[
                    {
                        "video_id": input["filename"],
                        "embedding": embedding[0],
                        "segment_start": i//_settings.model_sr,
                        "segment_duration": float(segment.shape[0]) / _settings.model_sr
                    }
                ]
            )
        return JSONResponse(
            content={
                "status": "ok"
            }
        )
    
@app.get("/video/{video_file}")
async def get_video(video_file: str):
    logger.warning(f"Возвращаем: {video_file}")
    video_path = f"/files/{video_file}"
    return FileResponse(video_path)

if __name__ == "__main__":
    uvicorn.run("project.app:app", port=_settings.port, host=_settings.host, workers=_settings.n_workers)
