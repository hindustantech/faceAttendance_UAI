# app/routes/detection.py
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
import numpy as np
import cv2
from datetime import datetime

from app.middleware.auth import verify_api_key
from app.services.face_detection import FaceDetectionService
from app.services.anti_spoofing import AntiSpoofingService
from app.config import settings
from app.utils.logger import setup_logger

logger = setup_logger(__name__)
router = APIRouter()

@router.post("/detect")
async def detect_face(
    file: UploadFile = File(..., description="Image to run face detection on"),
    check_spoofing: bool = Form(default=False, description="Also run anti-spoofing check"),
    api_key: str = Depends(verify_api_key)
):
    """
    Detect face(s) in an image WITHOUT matching against any enrolled database
    and WITHOUT training/storing anything.

    Use this as a lightweight pre-check on the client before calling
    /verify or /face-training/single, e.g. to give the user instant
    feedback like "move closer" or "only one face allowed" before you
    spend a network round trip on Cloudinary upload + DB matching.

    Returns:
    - face_count: number of faces found
    - faces: bbox, quality, angle, det_score for each face found
    - is_enrollment_ready: true if exactly 1 good-quality face is present
    """
    try:
        if file.content_type not in settings.SUPPORTED_FORMATS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {file.content_type}"
            )

        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")

        detector = FaceDetectionService()
        await detector.initialize()

        faces = await detector.detect_faces(image, min_confidence=0.5)

        spoof_result = None
        if check_spoofing and settings.ENABLE_ANTI_SPOOFING:
            anti_spoofing = AntiSpoofingService()
            spoof_result = await anti_spoofing.detect_spoofing(image)

        faces_out = []
        for f in faces:
            faces_out.append({
                "bbox": f["bbox"],
                "quality": f["quality"],
                "angle": f["angle"],
                "det_score": round(f["det_score"], 4),
                "face_size": f["face_size"],
                "has_embedding": f["embedding"] is not None
            })

        is_enrollment_ready = (
            len(faces_out) == 1
            and faces_out[0]["quality"] in ("good", "acceptable")
            and faces_out[0]["has_embedding"]
        )

        response_data = {
            "face_count": len(faces_out),
            "faces": faces_out,
            "is_enrollment_ready": is_enrollment_ready
        }

        if spoof_result is not None:
            response_data["spoofing"] = {
                "is_real": spoof_result["is_real"],
                "confidence": spoof_result["confidence"]
            }

        return {
            "success": True,
            "data": response_data,
            "timestamp": datetime.utcnow().isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Face detection failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))