# import nemo.collections.asr as nemo_asr
import requests

import librosa
import IPython.display as ipd
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
# from moviepy.editor import VideoFileClip, clips_array
from pathlib import Path
from pymilvus import MilvusClient
from tqdm import tqdm
from pymilvus import (
    FieldSchema,
    DataType
)

import logging

from typing import List, Dict
from pymilvus import (
    MilvusClient,
    FieldSchema,
    CollectionSchema,
    DataType,
    MilvusException
)

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(asctime)s %(msecs)d %(name)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)

class CustomMilvusClient(MilvusClient):
    def __init__(
        self,
        milvus_endpoint: str,
        db_name: str
    ):
        super().__init__(
            uri=milvus_endpoint,
            db_name=db_name
        )

    def get_or_create_collection(self, collection_key: str, fields: List[FieldSchema], index_params: Dict) -> None:
        if not self._get_connection().has_collection(collection_key):
            schema = CollectionSchema(fields=fields, description=f"{collection_key} collection")
            self._create_collection_with_schema(
                collection_name=collection_key,
                schema=schema,
                index_params=index_params
            )
            self.load_collection(collection_key)

    def insert_to_collection(self, collection_key: str, insert_data: List[Dict]) -> List[int | str]:
        primary_keys = self.insert(collection_name=collection_key, data=insert_data)
        if len(primary_keys["ids"]) != len(insert_data):
            bad_ids = list(set([data["id"] for data in insert_data]) - set(primary_keys["ids"]))
            logger.error(f"Inserted less objects in collection than expected: inserted {primary_keys}\nFailed: {bad_ids}")
            return bad_ids
        return []


if __name__ == "__main__":
    df = pd.read_csv("../data/piracy_val.csv")
    vidoses = df["ID_license"].tolist()

    
    for segment_duration in [10, 5]:
        for segment_shift in [0, segment_duration/2]:
            
            milvus_client = CustomMilvusClient(
                milvus_endpoint="http://127.0.0.1:19530",
                db_name="default"
            )
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, descrition="id", is_primary=True, auto_id=True),
                FieldSchema(name="video_id", dtype=DataType.VARCHAR, max_length=200, description="Video id"),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, description="Audio embedding", dim=1280),
                FieldSchema(name="segment_start", dtype=DataType.INT64, description="Start time of segment (seconds)"),
                FieldSchema(name="segment_duration", dtype=DataType.FLOAT, description="Length of segment (seconds)"),
            ]
            index_params = {
                "field_name": "embedding",
                "metric_type": "L2",
                "index_type": "IVF_FLAT",
                "params":{
                    "nlist": 1024
                }
            }
            milvus_client.get_or_create_collection(
                collection_key= f"whisper__{segment_duration}__{str(segment_shift).replace('.', '_')}",
                fields=fields,
                index_params=[index_params]
            )

            for video_id in tqdm(vidoses):
                response = requests.post(
                    'http://localhost:8123/upload',
                    json={
                        "filename": f"{video_id}",
                        "segment_duration": segment_duration,
                        "segment_shift": segment_shift,
                        "colllection_name": f"whisper__{segment_duration}__{str(segment_shift).replace('.', '_')}",
                    }
                )