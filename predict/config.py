from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    api_token: str = ""
    api_host: str = "localhost"
    api_port: int = 8000
    model_path: str = ""

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
