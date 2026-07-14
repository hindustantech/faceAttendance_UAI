# app/routes/recognition.py (STRENGTHENED VERSION)

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
import numpy as np
import cv2
from datetime import datetime
from typing import Optional
import json

from app.middleware.auth import verify_api_key
from app.services.face_recognition import FaceRecognitionService
from app.services.anti_spoofing import AntiSpoofingService
from app.config import settings
from app.utils.logger import setup_logger

logger = setup_logger(__name__)
router = APIRouter()


async def perform_anti_spoofing_check(image: np.ndarray, endpoint_name: str = "unknown"):
    """
    Centralized anti-spoofing check with strict fail-closed policy.
    Returns (passed: bool, result: dict)
    """
    try:
        anti_spoofing = AntiSpoofingService()
        await anti_spoofing.initialize()
        result = await anti_spoofing.detect_spoofing(image)
        
        is_real = result.get('is_real', False)
        confidence = result.get('confidence', 0.0)
        details = result.get('details', {})
        verdict = details.get('verdict', 'UNKNOWN')
        attack_type = details.get('attack_type', 'UNKNOWN')
        indicators = details.get('indicators', [])
        
        logger.info(
            f"[{endpoint_name}] Anti-spoofing: verdict={verdict}, "
            f"attack_type={attack_type}, confidence={confidence:.4f}, "
            f"is_real={is_real}"
        )
        
        if indicators:
            logger.warning(f"[{endpoint_name}] Spoof indicators: {indicators}")
        
        # STRICT REJECTION: Only accept if explicitly marked as REAL
        # with reasonable confidence
        if not is_real or confidence < 0.35:
            logger.warning(
                f"[{endpoint_name}] Spoofing detected: type={attack_type}, "
                f"confidence={confidence:.4f}"
            )
            raise HTTPException(
                status_code=403,
                detail={
                    "message": f"Spoofing detected: {attack_type}. Please use a real face.",
                    "confidence": confidence,
                    "details": {
                        "verdict": verdict,
                        "attack_type": attack_type,
                        "indicators": indicators
                    }
                }
            )
        
        logger.info(f"[{endpoint_name}] Anti-spoofing check passed")
        return True, result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[{endpoint_name}] Anti-spoofing error: {str(e)}")
        # FAIL-CLOSED: Reject on any anti-spoofing error
        raise HTTPException(
            status_code=403,
            detail={
                "message": "Security check failed. Please try again with better lighting.",
                "error": "Anti-spoofing service unavailable"
            }
        )


@router.post("/verify")
async def verify_face(
    company_id: str = Form(..., description="Company ID"),
    employee_id: Optional[str] = Form(None, description="Employee ID for 1:1 verification"),
    purpose: str = Form(default="punch_in", description="Verification purpose"),
    file: UploadFile = File(..., description="Face image for verification"),
    device_info: str = Form(default="{}", description="Device information (JSON)"),
    api_key: str = Depends(verify_api_key)
):
    """
    Verify face for attendance with MANDATORY anti-spoofing check.
    
    This endpoint ALWAYS performs anti-spoofing regardless of settings.
    """
    try:
        # Validate file
        if file.content_type not in settings.SUPPORTED_FORMATS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {file.content_type}"
            )
        
        # Read and validate image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        height, width = image.shape[:2]
        if height < 100 or width < 100:
            raise HTTPException(status_code=400, detail="Image too small")
        
        # Parse device info
        try:
            device_data = json.loads(device_info)
        except json.JSONDecodeError:
            device_data = {}
        
        # MANDATORY ANTI-SPOOFING CHECK
        # This runs regardless of settings - security first
        await perform_anti_spoofing_check(image, "verify")
        
        # Initialize recognition service
        recognition_service = FaceRecognitionService()
        await recognition_service.initialize()
        
        try:
            result = await recognition_service.verify_face(
                image_data=image,
                company_id=company_id,
                employee_id=employee_id,
                purpose=purpose,
                device_info=device_data
            )
            
            result['timestamp'] = datetime.utcnow().isoformat()
            result['security_checks'] = {
                'anti_spoofing': 'passed',
                'face_detection': 'passed'
            }
            
            if employee_id:
                result['data']['verification_type'] = '1:1'
                if result['data']['matched']:
                    result['message'] = f"Face verified as employee {employee_id}"
                else:
                    result['message'] = f"Face does not match employee {employee_id}"
            else:
                result['data']['verification_type'] = '1:N'
                if result['data']['matched']:
                    result['message'] = f"Face identified as employee {result['data']['employee_id']}"
                else:
                    result['message'] = "No matching employee found"
            
            return result
            
        finally:
            await recognition_service.cleanup()
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Verification failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Verification failed")


@router.post("/identify")
async def identify_face(
    company_id: str = Form(..., description="Company ID"),
    file: UploadFile = File(..., description="Face image for identification"),
    threshold: Optional[float] = Form(None),
    max_results: int = Form(default=10, ge=1, le=50),
    api_key: str = Depends(verify_api_key)
):
    """
    Identify face with MANDATORY anti-spoofing check.
    """
    try:
        # Validate file
        if file.content_type not in settings.SUPPORTED_FORMATS:
            raise HTTPException(status_code=400, detail="Unsupported file type")
        
        # Read and validate image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Validate threshold
        if threshold is not None and not (0 <= threshold <= 1):
            raise HTTPException(status_code=400, detail="Threshold must be between 0 and 1")
        
        # MANDATORY ANTI-SPOOFING CHECK
        await perform_anti_spoofing_check(image, "identify")
        
        # Initialize recognition service
        recognition_service = FaceRecognitionService()
        await recognition_service.initialize()
        
        try:
            result = await recognition_service.identify_face(
                image_data=image,
                company_id=company_id,
                threshold=threshold,
                max_results=max_results
            )
            
            result['timestamp'] = datetime.utcnow().isoformat()
            result['security_checks'] = {
                'anti_spoofing': 'passed',
                'face_detection': 'passed'
            }
            
            matches = result['data']['matches']
            if matches:
                result['message'] = f"Found {len(matches)} potential matches"
                if matches[0]['similarity'] >= (threshold or settings.FACE_MATCH_THRESHOLD):
                    result['message'] += f". Best: employee {matches[0]['employee_id']}"
            else:
                result['message'] = "No matches found"
            
            return result
            
        finally:
            await recognition_service.cleanup()
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Identification failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Identification failed")


# Alias endpoints
@router.post("/verify-employee")
async def verify_specific_employee(
    company_id: str = Form(...),
    employee_id: str = Form(...),
    file: UploadFile = File(...),
    purpose: str = Form(default="attendance"),
    device_info: str = Form(default="{}"),
    api_key: str = Depends(verify_api_key)
):
    """Verify if face belongs to specific employee (1:1) with anti-spoofing"""
    return await verify_face(
        company_id=company_id,
        employee_id=employee_id,
        purpose=purpose,
        file=file,
        device_info=device_info,
        api_key=api_key
    )


@router.post("/search")
async def search_employee_by_face(
    company_id: str = Form(...),
    file: UploadFile = File(...),
    threshold: Optional[float] = Form(None),
    limit: int = Form(default=5, ge=1, le=20),
    api_key: str = Depends(verify_api_key)
):
    """Search employee by face with anti-spoofing"""
    return await identify_face(
        company_id=company_id,
        file=file,
        threshold=threshold,
        max_results=limit,
        api_key=api_key
    )