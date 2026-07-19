from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """SAM3 配置管理"""

    model_config = SettingsConfigDict(env_prefix="")

    model_name: str = r"D:\code\tmp\sam3"
    device: Literal["auto", "cuda", "cpu"] = "auto"
    output_dir: str = "./out"
