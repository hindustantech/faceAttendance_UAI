# app/models/face_models.py
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum

class FaceAngle(str, Enum):
    FRONT = "front"
    LEFT = "left"
    RIGHT = "right"
    UP = "up"
    DOWN = "down"
    OTHER = "other"

class FaceQuality(str, Enum):
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"

class EnrollmentStatus(str, Enum):
    NOT_STARTED = "not_started"
    PENDING_REVIEW = "pending_review"
    APPROVED = "approved"
    REJECTED = "rejected"
    NEEDS_REENROLLMENT = "needs_reenrollment"

class VerificationPurpose(str, Enum):
    PUNCH_IN = "punch_in"
    PUNCH_OUT = "punch_out"
    RE_VERIFICATION = "re_verification"
    SPOT_CHECK = "spot_check"

class FaceImage(BaseModel):
    """Face image schema matching your Node.js FaceImageSchema"""
    url: str
    cloudinary_public_id: Optional[str] = None
    embedding: List[float]  # 512-d vector
    det_score: float = Field(default=0, ge=0, le=1)
    angle: FaceAngle = Field(default=FaceAngle.FRONT)
    quality: FaceQuality = Field(default=FaceQuality.GOOD)
    captured_at: datetime = Field(default_factory=datetime.utcnow)
    is_active: bool = Field(default=True)
    
    @validator("embedding")
    def validate_embedding(cls, v):
        """Validate embedding vector"""
        if not v or len(v) == 0:
            raise ValueError("Embedding must be non-empty")
        if not all(isinstance(x, (int, float)) for x in v):
            raise ValueError("Embedding must contain numbers only")
        return v

class VerificationLog(BaseModel):
    """Verification log matching your Node.js VerificationLogSchema"""
    attempted_at: datetime = Field(default_factory=datetime.utcnow)
    matched: bool
    similarity: float = Field(ge=-1, le=1)
    det_score: float = Field(default=0, ge=0, le=1)
    image_url: Optional[str] = None
    purpose: VerificationPurpose = Field(default=VerificationPurpose.PUNCH_IN)
    device_info: Optional[Dict[str, Any]] = None
    ip: Optional[str] = None
    attendance_id: Optional[str] = None

class FaceProfile(BaseModel):
    """Main face profile matching your Node.js faceSchema"""
    company_id: str
    employee_id: str
    images: List[FaceImage] = Field(default_factory=list, max_length=5)
    is_enrolled: bool = False
    enrollment_status: EnrollmentStatus = Field(default=EnrollmentStatus.NOT_STARTED)
    min_required_images: int = Field(default=2, ge=1, le=5)
    enrolled_at: Optional[datetime] = None
    last_re_enrolled_at: Optional[datetime] = None
    enrolled_by: Optional[str] = None
    rejection_reason: Optional[str] = None
    match_threshold: float = Field(default=0.65, ge=0, le=1)
    model_version: str = "insightface-arcface-buffalo_l"
    verification_logs: List[VerificationLog] = Field(default_factory=list)
    last_verified_at: Optional[datetime] = None
    total_verification_attempts: int = Field(default=0, ge=0)
    total_failed_attempts: int = Field(default=0, ge=0)
    consecutive_failed_attempts: int = Field(default=0, ge=0)
    is_locked: bool = False
    locked_at: Optional[datetime] = None
    lock_reason: Optional[str] = None
    is_suspicious: bool = False
    flagged_for_review: bool = False
    consent_given: bool = False
    consent_given_at: Optional[datetime] = None
    consent_withdrawn_at: Optional[datetime] = None
    data_retention_expires_at: Optional[datetime] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

class TrainingJob(BaseModel):
    """Training job model"""
    employee_id: str
    company_id: str
    type: str = "single"  # single or batch
    status: str = "pending"  # pending, processing, completed, failed
    total_images: int = 1
    processed_images: int = 0
    failed_images: int = 0
    metadata: Optional[Dict[str, Any]] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None