from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(extra="allow")

    milvus_endpoint: str = Field(..., env="MILVUS_ENDPOINT")
    milvus_collection_name: str = Field(..., env="MILVUS_COLLECTION_NAME")
    milvus_sim_metric: str = Field(..., env="MILVUS_SIM_METRIC")

class DevSettings(BaseSettings):
    model_config = SettingsConfigDict(extra="allow")
    host: str = "localhost"
    port: int = 8123

    model_sr: int = 16000

    segment_duration: int = 10

    milvus_endpoint: str = "http://127.0.0.1:19530"
    milvus_db_name: str = "default"
    milvus_collection_name: str = "audio_segments_whisper"
    milvus_sim_metric: str = "L2"
    milvus_index_type: str = "IVF_FLAT"
    milvus_index_nlist: int = 1024
    embedding_dim: int = 1280


    milvus_writer_n_workers: int = 8


_settings = DevSettings()
