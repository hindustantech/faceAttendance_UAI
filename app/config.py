from typing import List

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


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
    ALLOWED_ORIGINS: List[str] = Field(
        default=["https://api.praecore.in"]
    )

    @field_validator("ALLOWED_ORIGINS", mode="before")
    @classmethod
    def parse_allowed_origins(cls, v):
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",") if origin.strip()]
        return v

    # MongoDB
    MONGODB_URI: str = Field(...)
    MONGODB_DB_NAME: str = Field(...)

    # Redis
    REDIS_URL: str = Field(default="redis://localhost:6379/0")

    # Face Recognition
    FACE_MATCH_THRESHOLD: float = Field(default=0.65, ge=0.0, le=1.0)
    FACE_DETECTION_MODEL: str = Field(default="retinaface")
    FACE_RECOGNITION_MODEL: str = Field(default="arcface_r100_v1")

    # Anti Spoofing
    ENABLE_ANTI_SPOOFING: bool = Field(default=True)
    SPOOFING_THRESHOLD: float = Field(default=0.8, ge=0.0, le=1.0)

    # Image Processing
    MAX_IMAGE_SIZE_MB: int = Field(default=10)
    MIN_FACE_SIZE: int = Field(default=30)

    SUPPORTED_FORMATS: List[str] = Field(
        default=[
            "image/jpeg",
            "image/png",
            "image/webp",
        ]
    )

    @field_validator("SUPPORTED_FORMATS", mode="before")
    @classmethod
    def parse_supported_formats(cls, v):
        if isinstance(v, str):
            return [fmt.strip() for fmt in v.split(",") if fmt.strip()]
        return v

    # Training
    MAX_TRAINING_IMAGES: int = Field(default=5)
    MIN_TRAINING_IMAGES: int = Field(default=2)

    # Cloudinary
    CLOUDINARY_CLOUD_NAME: str = Field(...)
    CLOUDINARY_API_KEY: str = Field(...)
    CLOUDINARY_API_SECRET: str = Field(...)

    # Backend
    NODE_BACKEND_URL: str = Field(default="https://api.praecore.in")
    NODE_BACKEND_API_KEY: str = Field(...)

    # Cache
    FACE_DATABASE_CACHE_TTL: int = Field(default=1800)

    # Logging
    LOG_LEVEL: str = Field(default="INFO")
    LOG_FORMAT: str = Field(default="json")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )


settings = Settings()