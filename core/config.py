from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    openrouter_api_key: str
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    
    agent_model: str = "google/gemma-4-31b-it:free"
    
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()