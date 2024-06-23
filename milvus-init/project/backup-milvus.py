import requests
import json
import logging
from time import sleep
from pymilvus import (
    FieldSchema,
    DataType,
    MilvusClient,
    CollectionSchema
)
from pathlib import Path
from tqdm import tqdm
from project.settings import _settings
from project.storage.milvus import CustomMilvusClient

# requests.post(
#     "http://milvus-backup:8080/api/v1/restore",
#     json = {
#         "async": False,
#         "collection_names": [
#             "whisper_segments",
#         ],
#         "collection_suffix": "",
#         "backup_name":"milvus-backup"
#     }
# )

DIRECTORY = "/app/backup-data"

if __name__ == "__main__":

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
    milvus_client.load_collection(_settings.milvus_collection_name)
    for json_path in tqdm(list(Path(DIRECTORY).glob("*.json"))):
        logging.info(f"Loading new video...")
        with open(json_path, 'r') as f:
            data = json.load(f)
        for elem in data:
            milvus_client.insert(
                collection_name=_settings.milvus_collection_name,
                data=elem
            )
    logging.info("Done")
