from pydantic_settings import BaseSettings
from typing import Optional
import os


class Settings(BaseSettings):
    google_api_key: str = ""
    secret_key: str = "default-secret"
    host: str = "0.0.0.0"
    port: int = 8000
    code_timeout_seconds: int = 120
    max_retries_per_question: int = 3
    chrome_headless: bool = True
    log_level: str = "INFO"
    temp_dir: str = "/tmp/quiz_sandbox" if os.name != 'nt' else os.path.join(os.environ.get('TEMP', 'C:\\Temp'), 'quiz_sandbox')
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"  # Ignore extra fields in .env file
settings = Settings()
