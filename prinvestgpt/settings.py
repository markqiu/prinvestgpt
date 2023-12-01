from typing import Literal, List

from pydantic.networks import IPvAnyAddress, AnyUrl
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # ernie
    ernie_client_id: str  # 百度千帆的应用接入id
    ernie_client_secret: str  # 百度千帆的应用接入secret
    openai_api_key: str  # openai的api_key

    # MODELS:
    llm_model_config: List[Literal["Ernie", "OpenAI"]]  # 语言模型名称
    embedding_model_config: List[Literal["Ernie", "OpenAI"]]  # embedding模型名称

    # gradio服务配置
    concurrency_count: int = 32
    bind_address: IPvAnyAddress = "0.0.0.0"
    server_port: int = 7860
    root_path: str = "/"
    debug: bool = True

    # meilisearch配置
    meilisearch_url: AnyUrl = "http://127.0.0.1:7700"
    meilisearch_api_key: str  # meilisearch的api_key

    # baklib's key
    baklib_token: str
    baklib_tenant_id: str

    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8')


settings = Settings()
