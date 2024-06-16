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
    milvus_client = MilvusClient(
        uri="http://standalone:19530",
        db_name="default"
    )
    while not milvus_client._get_connection().has_collection("whisper_segments"):
        logging.warning("Can't find collection whisper_segments")
        sleep(5)
    # fields = [
    #     FieldSchema(name="id", dtype=DataType.INT64, descrition="id", is_primary=True, auto_id=True),
    #     FieldSchema(name="video_id", dtype=DataType.VARCHAR, max_length=200, description="Video id"),
    #     FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, description="Audio embedding", dim=1280),
    #     FieldSchema(name="segment_start", dtype=DataType.INT64, description="Start time of segment (seconds)"),
    #     FieldSchema(name="segment_duration", dtype=DataType.FLOAT, description="Length of segment (seconds)"),
    # ]
    # index_params = {
    #     "field_name": "embedding",
    #     "metric_type": "L2",
    #     "index_type": "IVF_FLAT",
    #     "params":{
    #         "nlist":1024
    #     }
    # }
    # schema = CollectionSchema(fields=fields, description=f"whisper_segments collection")
    # milvus_client._create_collection_with_schema(
    #     collection_name="whisper_segments",
    #     schema=schema,
    #     index_params=index_params
    # )
    # milvus_client.load_collection("whisper_segments")
    for json_path in tqdm(list(Path(DIRECTORY).glob("*.json"))):
        logging.info(f"Loading new video...")
        with open(json_path, 'r') as f:
            data = json.load(f)
        for elem in data:
            milvus_client.insert(
                collection_name="whisper_segments",
                data=elem
            )
    logging.info("Done")
