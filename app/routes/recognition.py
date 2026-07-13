# app/routes/recognition.py (fixed version)

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends, Query
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
    Verify face for attendance
    
    Supports two modes:
    1. 1:1 Verification (employee_id provided): Verify if face matches specific employee
    2. 1:N Verification (employee_id not provided): Find matching employee in company
    
    This endpoint:
    1. Validates the image
    2. Performs anti-spoofing check (if enabled)
    3. Extracts face embedding
    4. Matches against enrolled faces
    5. Returns match result
    """
    try:
        # Validate file
        if file.content_type not in settings.SUPPORTED_FORMATS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {file.content_type}. Supported: {settings.SUPPORTED_FORMATS}"
            )
        
        # Read image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Validate image size
        height, width = image.shape[:2]
        if height < 100 or width < 100:
            raise HTTPException(status_code=400, detail="Image too small. Minimum 100x100 pixels required.")
        
        # Parse device info
        try:
            device_data = json.loads(device_info)
        except json.JSONDecodeError:
            device_data = {}
        
        # FIXED: Anti-spoofing check with better handling
        if settings.ENABLE_ANTI_SPOOFING:
            try:
                anti_spoofing = AntiSpoofingService()
                await anti_spoofing.initialize()
                spoof_result = await anti_spoofing.detect_spoofing(image)
                
                is_real = spoof_result.get('is_real', True)
                confidence = spoof_result.get('confidence', 1.0)
                details = spoof_result.get('details', {})
                verdict = details.get('verdict', 'UNKNOWN')
                
                logger.info(f"Anti-spoofing check: verdict={verdict}, confidence={confidence:.4f}, is_real={is_real}")
                
                # Only reject if clearly spoofed (very low confidence)
                if not is_real and confidence < 0.40:
                    logger.warning(f"Spoofing detected (high confidence): {details}")
                    raise HTTPException(
                        status_code=403,
                        detail={
                            "message": "Spoofing detected. Please use a real face.",
                            "confidence": confidence,
                            "details": details
                        }
                    )
                elif not is_real:
                    # Borderline case - log but allow
                    logger.warning(f"Borderline spoof check (confidence: {confidence:.4f}) - allowing verification")
                else:
                    logger.info(f"Anti-spoofing passed: confidence={confidence:.4f}")
                    
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Anti-spoofing check failed with error: {str(e)}")
                # If anti-spoofing itself fails, allow the verification to proceed
                logger.warning("Anti-spoofing service error - allowing verification to continue")
        
        # Initialize recognition service
        recognition_service = FaceRecognitionService()
        await recognition_service.initialize()
        
        try:
            # Perform verification
            result = await recognition_service.verify_face(
                image_data=image,
                company_id=company_id,
                employee_id=employee_id,
                purpose=purpose,
                device_info=device_data
            )
            
            # Add metadata
            result['timestamp'] = datetime.utcnow().isoformat()
            
            if employee_id:
                result['data']['verification_type'] = '1:1'
                if result['data']['matched']:
                    result['message'] = f"Face matched with employee {employee_id}"
                else:
                    result['message'] = f"Face did not match employee {employee_id}"
            else:
                result['data']['verification_type'] = '1:N'
                if result['data']['matched']:
                    result['message'] = f"Face matched with employee {result['data']['employee_id']}"
                else:
                    result['message'] = "No matching employee found"
            
            return result
            
        finally:
            await recognition_service.cleanup()
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Verification failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Verification failed: {str(e)}")


@router.post("/identify")
async def identify_face(
    company_id: str = Form(..., description="Company ID"),
    file: UploadFile = File(..., description="Face image for identification"),
    threshold: Optional[float] = Form(None, description="Match threshold (0-1). Default from settings."),
    max_results: int = Form(default=10, description="Maximum results to return", ge=1, le=50),
    api_key: str = Depends(verify_api_key)
):
    """
    Identify face from database (1:N matching)
    
    Returns top matches sorted by similarity score.
    This is useful for:
    - Spot checking
    - Finding an employee by face
    - Security verification
    
    The endpoint always returns results regardless of threshold,
    but only matches above threshold are considered valid.
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
        
        # Validate threshold if provided
        if threshold is not None and not (0 <= threshold <= 1):
            raise HTTPException(
                status_code=400,
                detail="Threshold must be between 0 and 1"
            )
        
        # Optional: Anti-spoofing check for identify endpoint
        if settings.ENABLE_ANTI_SPOOFING:
            try:
                anti_spoofing = AntiSpoofingService()
                await anti_spoofing.initialize()
                spoof_result = await anti_spoofing.detect_spoofing(image)
                
                is_real = spoof_result.get('is_real', True)
                confidence = spoof_result.get('confidence', 1.0)
                
                # Only reject if clearly spoofed
                if not is_real and confidence < 0.40:
                    logger.warning(f"Spoofing detected in identify: confidence={confidence:.4f}")
                    raise HTTPException(
                        status_code=403,
                        detail={
                            "message": "Spoofing detected. Please use a real face.",
                            "confidence": confidence
                        }
                    )
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Anti-spoofing error in identify: {str(e)}")
                # Continue if anti-spoofing fails
        
        # Initialize recognition service
        recognition_service = FaceRecognitionService()
        await recognition_service.initialize()
        
        try:
            # Perform identification
            result = await recognition_service.identify_face(
                image_data=image,
                company_id=company_id,
                threshold=threshold,
                max_results=max_results
            )
            
            # Add metadata
            result['timestamp'] = datetime.utcnow().isoformat()
            result['company_id'] = company_id
            result['query_info'] = {
                'threshold_used': threshold or settings.FACE_MATCH_THRESHOLD,
                'max_results': max_results
            }
            
            # Add summary message
            matches = result['data']['matches']
            if matches:
                result['message'] = f"Found {len(matches)} potential matches"
                if matches[0]['similarity'] >= (threshold or settings.FACE_MATCH_THRESHOLD):
                    result['message'] += f". Best match: employee {matches[0]['employee_id']} with {matches[0]['similarity']*100:.2f}% confidence"
            else:
                result['message'] = "No matches found"
            
            return result
            
        finally:
            await recognition_service.cleanup()
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Identification failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Identification failed: {str(e)}")


@router.post("/verify-employee")
async def verify_specific_employee(
    company_id: str = Form(..., description="Company ID"),
    employee_id: str = Form(..., description="Employee ID to verify"),
    file: UploadFile = File(..., description="Face image"),
    purpose: str = Form(default="attendance", description="Verification purpose"),
    device_info: str = Form(default="{}", description="Device information (JSON)"),
    api_key: str = Depends(verify_api_key)
):
    """
    Verify if face belongs to a specific employee (1:1 matching)
    
    This is a convenience endpoint for explicit 1:1 verification.
    It's equivalent to /verify with employee_id parameter.
    """
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
    company_id: str = Form(..., description="Company ID"),
    file: UploadFile = File(..., description="Face image to search"),
    threshold: Optional[float] = Form(None, description="Minimum similarity threshold"),
    limit: int = Form(default=5, ge=1, le=20),
    api_key: str = Depends(verify_api_key)
):
    """
    Search for employee by face image
    
    Returns list of potential matches sorted by similarity.
    This is an alias for /identify with different parameter names.
    """
    return await identify_face(
        company_id=company_id,
        file=file,
        threshold=threshold,
        max_results=limit,
        api_key=api_key
    )


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "face-recognition",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
        "features": {
            "anti_spoofing": settings.ENABLE_ANTI_SPOOFING,
            "face_detection": True,
            "verification_modes": ["1:1", "1:N"]
        }
    }