import requests

# requests.post(
#     "http://localhost:10101/api/v1/create",
#     json = {
#         "async": False,
#         "backup_name": "test_backup",
#         "collection_names": [
#             "whisper__10__0",
#             "whisper__10__5_0"
#         ]
#     }
# )

requests.post(
    "http://localhost:10101/api/v1/restore",
    json = {
        "async": False,
        "collection_names": [
            "whisper__10__0",
            "whisper__10__5_0"
        ],
        "collection_suffix": "_bak",
        "backup_name":"test_backup"
    }
)
