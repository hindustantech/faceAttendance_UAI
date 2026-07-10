# app/routes/recognition.py
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
import numpy as np
import cv2
from datetime import datetime

from app.middleware.auth import verify_api_key
from app.services.face_recognition import FaceRecognitionService
from app.services.anti_spoofing import AntiSpoofingService
from app.config import settings
from app.utils.logger import setup_logger

logger = setup_logger(__name__)
router = APIRouter()

@router.post("/verify")
async def verify_face(
    company_id: str = Form(..., description="Company ID"),
    purpose: str = Form(default="punch_in", description="Verification purpose"),
    file: UploadFile = File(..., description="Face image for verification"),
    device_info: str = Form(default="{}", description="Device information (JSON)"),
    api_key: str = Depends(verify_api_key)
):
    """
    Verify face for attendance
    
    This endpoint:
    1. Validates the image
    2. Performs anti-spoofing check
    3. Extracts face embedding
    4. Matches against enrolled faces
    5. Returns match result
    """
    try:
        # Validate file
        if file.content_type not in settings.SUPPORTED_FORMATS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {file.content_type}"
            )
        
        # Read image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Parse device info
        import json
        try:
            device_data = json.loads(device_info)
        except:
            device_data = {}
        
        # Anti-spoofing check
        if settings.ENABLE_ANTI_SPOOFING:
            anti_spoofing = AntiSpoofingService()
            spoof_result = await anti_spoofing.detect_spoofing(image)
            
            if not spoof_result['is_real']:
                logger.warning(f"Spoofing detected: {spoof_result['details']}")
                
                raise HTTPException(
                    status_code=403,
                    detail={
                        "message": "Spoofing detected. Please use a real face.",
                        "confidence": spoof_result['confidence'],
                        "details": spoof_result['details']
                    }
                )
        
        # Verify face
        recognition_service = FaceRecognitionService()
        await recognition_service.initialize()
        
        result = await recognition_service.verify_face(
            image_data=image,
            company_id=company_id,
            purpose=purpose,
            device_info=device_data
        )
        
        return {
            "success": True,
            "data": result['data'],
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Verification failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/identify")
async def identify_face(
    company_id: str = Form(...),
    file: UploadFile = File(...),
    threshold: float = Form(default=None),
    api_key: str = Depends(verify_api_key)
):
    """
    Identify face from database (1:N matching)
    Returns top matches
    """
    try:
        # Read image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Override threshold if provided
        if threshold:
            settings.FACE_MATCH_THRESHOLD = threshold
        
        # Identify face
        recognition_service = FaceRecognitionService()
        await recognition_service.initialize()
        
        result = await recognition_service.verify_face(
            image_data=image,
            company_id=company_id,
            purpose="spot_check"
        )
        
        return {
            "success": True,
            "data": result['data'],
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Identification failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))