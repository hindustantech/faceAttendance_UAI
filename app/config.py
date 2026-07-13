# app/config.py
from pydantic_settings import BaseSettings, SettingsConfigDict, NoDecode
from pydantic import Field, field_validator
from typing import List, Annotated


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""

    # Application
    APP_NAME: str = Field(default="Face Recognition Service")
    VERSION: str = Field(default="1.0.0")
    DEBUG: bool = Field(default=False)
    PORT: int = Field(default=8000, ge=1, le=65535)
    WORKERS: int = Field(default=4, ge=1, le=8)

    # Security
    API_KEY: str = Field(..., min_length=16)
    ALLOWED_ORIGINS: Annotated[List[str], NoDecode] = Field(
        default=["https://api.praecore.in"]
    )

    @field_validator("ALLOWED_ORIGINS", mode="before")
    @classmethod
    def parse_allowed_origins(cls, v):
        """Parse comma-separated origins string to list"""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",") if origin.strip()]
        return v

    # MongoDB - Your existing database
    MONGODB_URI: str = Field(...)
    MONGODB_DB_NAME: str = Field(...)

    # Redis
    REDIS_URL: str = Field(default="redis://localhost:6379/0")

    # Face Recognition Settings
    # app/config.py - Update this line
    FACE_MATCH_THRESHOLD: float = Field(default=0.45, ge=0.0, le=1.0)  # Changed from 0.65 to 0.45
    FACE_DETECTION_MODEL: str = Field(default="retinaface")
    FACE_RECOGNITION_MODEL: str = Field(default="arcface_r100_v1")

    # Anti-spoofing
    ENABLE_ANTI_SPOOFING: bool = Field(default=True)
    SPOOFING_THRESHOLD: float = Field(default=0.8, ge=0.0, le=1.0)

    # Image Processing
    MAX_IMAGE_SIZE_MB: int = Field(default=10)
    MIN_FACE_SIZE: int = Field(default=30)
    SUPPORTED_FORMATS: Annotated[List[str], NoDecode] = Field(
        default=["image/jpeg", "image/png", "image/webp"]
    )

    @field_validator("SUPPORTED_FORMATS", mode="before")
    @classmethod
    def parse_supported_formats(cls, v):
        """Parse comma-separated formats string to list (if set via env)"""
        if isinstance(v, str):
            return [fmt.strip() for fmt in v.split(",") if fmt.strip()]
        return v

    # Training
    MAX_TRAINING_IMAGES: int = Field(default=5)
    MIN_TRAINING_IMAGES: int = Field(default=2)

    # Cloudinary - Your existing Cloudinary
    CLOUDINARY_CLOUD_NAME: str = Field(...)
    CLOUDINARY_API_KEY: str = Field(...)
    CLOUDINARY_API_SECRET: str = Field(...)

    # Node.js Backend
    NODE_BACKEND_URL: str = Field(default="https://api.praecore.in")
    NODE_BACKEND_API_KEY: str = Field(...)

    # Cache
    FACE_DATABASE_CACHE_TTL: int = Field(default=1800)  # 30 minutes

    # Logging
    LOG_LEVEL: str = Field(default="INFO")
    LOG_FORMAT: str = Field(default="json")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )


# Create global settings instance
settings = Settings()