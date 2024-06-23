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
