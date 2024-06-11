import logging
import requests
import numpy as np
import librosa
import uvicorn
import base64

from io import BytesIO
from joblib import Parallel, delayed, parallel_config
from contextlib import asynccontextmanager
from typing import Dict
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from PIL import Image
from pymilvus import (
    FieldSchema,
    DataType
)
from tqdm import tqdm
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

@app.post("/upload")
async def upload():
    pass

@app.post("/check")
async def check(input: Dict) -> JSONResponse:
    audio, _ = librosa.load(f"../data/videos/{input['filename']}", sr=_settings.model_sr)
    for i in tqdm(list(range(0, len(audio), _settings.model_sr * _settings.segment_duration))):
        segment = audio[i:i+_settings.model_sr * _settings.segment_duration]
        if len(segment) < _settings.model_sr * _settings.segment_duration:
            continue
        embedding = requests.post(
            "http://localhost:8000/encode",
            json = {
                "audio": segment.tolist()
            }
        ).json()
        milvus_client.insert_to_collection(
            collection_key=_settings.milvus_collection_name,
            insert_data=[
                {
                    "video_id": input["filename"],
                    "embedding": embedding[0],
                    "segment_start": i//_settings.model_sr
                }
            ]
        )
    return JSONResponse(
        content={
            "status": "ok"
        }
    )
        
    
    
# @app.post("/search")
# async def search(query: Query) -> Dict:
#     if query.image is None and query.id is None:
#         return {"error": "Base64 image or id is required"}
#     elif query.image is not None and query.id is not None:
#         return {"error": "Only one of base64 image or id is allowed"}
#     elif query.image is not None:
#         logger.warning("Search by image")
#         image = Image.open(BytesIO(base64.b64decode(query.image)))
#         _, embedding = ram_model.process(image)
#         result_idxs = client.search(
#             collection_name=_settings.milvus_collection_name,
#             data=[embedding],
#             limit=_settings.n_results,
#             search_params={"metric_type": _settings.milvus_sim_metric},
#             filter=get_filter(query)
#         )[0]
#         results = [elem["id"] for elem in result_idxs]
#         return {
#             "ids": results
#         }
#     elif query.id is not None:
#         logger.warning("Search by id")
#         embedding = client.get(
#             collection_name = _settings.milvus_collection_name,
#             ids=query.id
#         )[0]["embedding"]
#         result_idxs = client.search(
#             collection_name=_settings.milvus_collection_name,
#             data=[embedding],
#             limit=_settings.n_results,
#             search_params={"metric_type": _settings.milvus_sim_metric},
#             filter=get_filter(query)
#         )[0]
#         results = [elem["id"] for elem in result_idxs]
#         if int(query.id) in results:
#             results.remove(int(query.id))
#         return {
#             "ids": results
#         }
#     raise Exception("Something went wrong in if statements")

if __name__ == "__main__":
    uvicorn.run(app, port=_settings.port, host=_settings.host)
