import requests

requests.post(
    "http://localhost:10101/api/v1/create",
    json = {
        "async": False,
        "backup_name": "milvus-backup",
        "collection_names": [
            "whisper_segments",
        ]
    }
)