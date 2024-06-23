from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(extra="allow")
    
    milvus_endpoint: str = Field(..., env="MILVUS_ENDPOINT")
    milvus_db_name: str = Field(..., env="MILVUS_DB_NAME")
    milvus_collection_name: str = Field(..., env="MILVUS_COLLECTION_NAME")
    milvus_sim_metric: str = Field(..., env="MILVUS_SIM_METRIC")
    milvus_index_type: str = Field(..., env="MILVUS_INDEX_TYPE")
    milvus_index_nlist: int = 1024
    embedding_dim: int =  Field(..., env="EMBEDDING_DIM")

_settings = Settings()
