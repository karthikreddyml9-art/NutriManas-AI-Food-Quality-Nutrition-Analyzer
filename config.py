from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    supabase_url: str = ""
    supabase_anon_key: str = ""
    usda_api_key: str = "DEMO_KEY"
    gemini_api_key: str = ""
    ollama_base_url: str = "http://localhost:11434"
    ollama_vision_model: str = "llama3.2-vision"
    ollama_text_model: str = "llama3.1:8b"

    class Config:
        env_file = ".env"


settings = Settings()
