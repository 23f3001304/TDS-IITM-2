"""Application configuration with environment variable support."""
import os
from functools import lru_cache
from typing import Optional

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# Constants
DEFAULT_TIMEOUT = 120
DEFAULT_MAX_RETRIES = 3
DEFAULT_PORT = 8000
MIN_TIMEOUT = 10
MAX_TIMEOUT = 600


def _get_temp_dir() -> str:
    """Get platform-appropriate temp directory."""
    if os.name == 'nt':  # Windows
        return os.path.join(os.environ.get('TEMP', 'C:\\Temp'), 'quiz_sandbox')
    return '/tmp/quiz_sandbox'


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
    
    # API Keys
    google_api_key: str = ""
    secret_key: str = "default-secret"
    
    # Server Settings
    host: str = "0.0.0.0"
    port: int = DEFAULT_PORT
    
    # Execution Settings
    code_timeout_seconds: int = DEFAULT_TIMEOUT
    max_retries_per_question: int = DEFAULT_MAX_RETRIES
    
    # Browser Settings  
    chrome_headless: bool = True
    
    # Logging
    log_level: str = "INFO"
    
    # Paths
    temp_dir: str = _get_temp_dir()
    
    @field_validator('code_timeout_seconds')
    @classmethod
    def validate_timeout(cls, v: int) -> int:
        """Ensure timeout is within reasonable bounds."""
        return max(MIN_TIMEOUT, min(v, MAX_TIMEOUT))
    
    @field_validator('log_level')
    @classmethod  
    def validate_log_level(cls, v: str) -> str:
        """Ensure log level is valid."""
        valid_levels = {'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'}
        v = v.upper()
        if v not in valid_levels:
            return 'INFO'
        return v


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


settings = get_settings()
