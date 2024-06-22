from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(extra="allow")

    host: str = "0.0.0.0"
    port: int = 8080
    n_workers: int  =  Field(..., env="N_WORKERS")

    model_sr: int = Field(..., env="MODEL_SR")

    milvus_endpoint: str = Field(..., env="MILVUS_ENDPOINT")
    milvus_db_name: str = Field(..., env="MILVUS_DB_NAME")
    milvus_collection_name: str = Field(..., env="MILVUS_COLLECTION_NAME")
    milvus_sim_metric: str = Field(..., env="MILVUS_SIM_METRIC")
    milvus_index_type: str = Field(..., env="MILVUS_INDEX_TYPE")
    milvus_index_nlist: int = 1024
    embedding_dim: int =  Field(..., env="EMBEDDING_DIM")

class DevSettings(BaseSettings):
    model_config = SettingsConfigDict(extra="allow")
    host: str = "localhost"
    port: int = 16483
    n_workers: int = 1

    model_sr: int = 16000

    milvus_endpoint: str = "http://standalone:19530"
    milvus_db_name: str = "default"
    milvus_collection_name: str = "audio_segments_whisper"
    milvus_sim_metric: str = "L2"
    milvus_index_type: str = "IVF_FLAT"
    milvus_index_nlist: int = 1024
    embedding_dim: int = 1280

_settings = Settings()
