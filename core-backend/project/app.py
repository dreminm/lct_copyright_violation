import logging
import requests
import numpy as np
import librosa
import uvicorn
import os

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
from .utils import chain_search_algorithm, filter_candidates, VideoReader, cpu
from .settings import _settings
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
            "nlist":_settings.milvus_index_nlist
        }
    }
    milvus_client.get_or_create_collection(
        collection_key= _settings.milvus_collection_name,
        fields=fields,
        index_params=[index_params]
    )
    yield
    
app = FastAPI(lifespan=lifespan)

# @app.post("/upload")
# async def upload():
#     pass

@app.post("/upload")
async def upload_video(video: UploadFile = File(...)):
    """
    Принимает видео файл и сохраняет его в файловой системе.
    """
    try:
        file_path = os.path.join(UPLOAD_FOLDER, video.filename)

        # Сохраняем файл
        # with open(file_path, "wb") as buffer:
        #     contents = await video.read()
        #     buffer.write(contents)

        #video = await video.read()#["video"]
        SR = 16_000
        audio, _ = librosa.load(file_path, sr=SR)
        duration = len(audio) // SR
        audio_candidates = chain_search_algorithm(
            audio,
            16_000,
            duration,
            10,
            1,
            2,
            "whisper__10__0",
            None,
            100
        )
        result = dict()
        source_reader = VideoReader(file_path, ctx=cpu(0))

        candidates = audio_candidates

        for video_id, candidates_list in candidates.items():
            video_check_reader = VideoReader(
                osp.join(
                    os.environ['DATABASE_VIDEOS_PATH'],
                    video_id
                ),
                ctx=cpu(0)
            )
            selected_cand = filter_candidates(
                source_reader, video_check_reader,
                candidates_list
            )
            if len(selected_cand) > 0:
                result[video_id] = selected_cand

        analog_info = [
            {"filename": "http://127.0.0.1:8000/video/video2.mp4", "time_intervals": [
                {"start_sec": 27, "t_start": "0:0:27 - 0:0:50"},
                {"start_sec": 30, "t_start": "0:0:30 - 0:0:53"}]},
            {"filename": "http://127.0.0.1:8000/video/video3.mp4", "time_intervals": [
                {"start_sec": 125, "t_start": "0:2:5 - 0:2:20"}]}
        ]
        upload_info = [
            [{"start_sec": 40, "t_start": "0:0:40 - 0:1:3"},
             {"start_sec": 45, "t_start": "0:0:45 - 0:1:8"}],
            [{"start_sec": 140, "t_start": "0:2:20 - 0:2:35"}]
        ]

        return JSONResponse(content={"message": "Видео успешно загружено!",
                                     "algorithm_result": result,
                                     "analog_info": analog_info,
                                     "upload_info": upload_info
                                     }, status_code=200)
    except Exception as e:
        return JSONResponse(content={"message": f"Ошибка загрузки видео: {str(e)}"}, status_code=500)

@app.post("/upload/milvus")
async def upload(input: Dict) -> JSONResponse:
    audio, _ = librosa.load(f"../data/videos/{input['filename']}", sr=_settings.model_sr)

    segment_duration= input["segment_duration"]
    segment_shift= input["segment_shift"]
    collection_name = input["colllection_name"]

    assert segment_shift >= 0
    assert segment_shift <= segment_duration


    for i in tqdm(list(range(0, int(len(audio) - _settings.model_sr * (segment_duration - segment_shift - 1)), _settings.model_sr))):
        segment = audio[max(0, int(i-segment_shift*_settings.model_sr)):int(i+_settings.model_sr * (segment_duration - segment_shift))]

        embedding = requests.post(
            "http://localhost:8001/v1/models/encoder:predict",
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
    # Путь к файлу видео
    video_path = f"./files/{video_file}"
    return FileResponse(video_path)


if __name__ == "__main__":
    uvicorn.run("project.app:app", port=_settings.port, host=_settings.host, workers=_settings.n_workers)
