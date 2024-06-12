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
    port: int = 8001

    model_name: str =  "antony66/whisper-large-v3-russian"
    model_sr: int = 16000
    segment_duration: int = 10


_settings = DevSettings()
