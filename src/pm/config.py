from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    app_name: str ="Predictive Maintenance API"
    api_version: str = "0.1.0"


settings = Settings()
