# app/routes/detection.py
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
import numpy as np
import cv2
from datetime import datetime
from typing import Optional
import json

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
    company_id: Optional[str] = Form(None, description="Company ID for contextual detection"),
    employee_id: Optional[str] = Form(None, description="Employee ID for employee-specific detection"),
    api_key: str = Depends(verify_api_key)
):
    """
    Detect face(s) in an image WITHOUT matching against any enrolled database
    and WITHOUT training/storing anything.

    Use this as a lightweight pre-check on the client before calling
    /verify or /face-training/single, e.g. to give the user instant
    feedback like "move closer" or "only one face allowed" before you
    spend a network round trip on Cloudinary upload + DB matching.

    Optional Parameters:
    - company_id: Provides company context for detection
    - employee_id: Provides employee context with quick match check

    Returns:
    - face_count: number of faces found
    - faces: bbox, quality, angle, det_score for each face found
    - is_enrollment_ready: true if exactly 1 good-quality face is present
    - context: company/employee specific information (if provided)
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
        
        # Validate image size
        height, width = image.shape[:2]
        if height < 50 or width < 50:
            raise HTTPException(
                status_code=400, 
                detail="Image too small. Minimum 50x50 pixels required."
            )

        detector = FaceDetectionService()
        await detector.initialize()

        try:
            # Perform detection based on context
            if employee_id and company_id:
                # Employee-specific detection with quick match
                result = await detector.detect_face_for_employee(
                    image_data=image,
                    company_id=company_id,
                    employee_id=employee_id
                )
                detection_type = "employee_specific"
            elif company_id:
                # Company-specific detection
                result = await detector.detect_face_for_company(
                    image_data=image,
                    company_id=company_id
                )
                detection_type = "company_specific"
            else:
                # Generic detection
                faces = await detector.detect_faces(image, min_confidence=0.5)
                result = {
                    'face_count': len(faces),
                    'faces': detector._format_faces_for_response(faces),
                    'is_enrollment_ready': len(faces) == 1 and faces[0].get('is_enrollment_ready', False),
                }
                detection_type = "generic"

            # Anti-spoofing check if requested
            spoof_result = None
            if check_spoofing and settings.ENABLE_ANTI_SPOOFING:
                anti_spoofing = AntiSpoofingService()
                spoof_result = await anti_spoofing.detect_spoofing(image)

            # Prepare response
            response_data = {
                "success": True,
                "data": {
                    **result,
                    "detection_type": detection_type,
                    "image_info": {
                        "width": width,
                        "height": height,
                        "format": file.content_type,
                        "file_size": len(contents)
                    }
                },
                "timestamp": datetime.utcnow().isoformat()
            }

            # Add spoofing results if available
            if spoof_result is not None:
                response_data["data"]["spoofing"] = {
                    "is_real": spoof_result["is_real"],
                    "confidence": spoof_result["confidence"],
                    "details": spoof_result.get("details", {})
                }

            # Add summary
            face_count = result.get('face_count', 0)
            if face_count == 0:
                response_data["message"] = "No face detected in the image"
            elif face_count == 1:
                response_data["message"] = "One face detected successfully"
                if result.get('is_enrollment_ready'):
                    response_data["message"] += " - Ready for enrollment/verification"
            else:
                response_data["message"] = f"Multiple faces detected ({face_count})"

            return response_data

        finally:
            await detector.cleanup()

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Face detection failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/detect-company")
async def detect_face_for_company(
    company_id: str = Form(..., description="Company ID"),
    file: UploadFile = File(..., description="Image to detect faces in"),
    check_spoofing: bool = Form(default=False, description="Also run anti-spoofing check"),
    api_key: str = Depends(verify_api_key)
):
    """
    Detect faces specifically for a company context.
    
    This endpoint provides:
    - Face detection with company context
    - Company enrollment statistics
    - Recent detection history for the company
    
    Useful for company-wide face detection needs.
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

        try:
            # Company-specific detection
            result = await detector.detect_face_for_company(
                image_data=image,
                company_id=company_id
            )

            # Anti-spoofing if requested
            spoof_result = None
            if check_spoofing and settings.ENABLE_ANTI_SPOOFING:
                anti_spoofing = AntiSpoofingService()
                spoof_result = await anti_spoofing.detect_spoofing(image)

            response_data = {
                "success": True,
                "data": {
                    **result,
                    "company_id": company_id,
                    "detection_type": "company_specific"
                },
                "timestamp": datetime.utcnow().isoformat()
            }

            if spoof_result:
                response_data["data"]["spoofing"] = {
                    "is_real": spoof_result["is_real"],
                    "confidence": spoof_result["confidence"]
                }

            return response_data

        finally:
            await detector.cleanup()

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Company detection failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/detect-employee")
async def detect_face_for_employee(
    company_id: str = Form(..., description="Company ID"),
    employee_id: str = Form(..., description="Employee ID"),
    file: UploadFile = File(..., description="Image to detect face in"),
    check_spoofing: bool = Form(default=False, description="Also run anti-spoofing check"),
    api_key: str = Depends(verify_api_key)
):
    """
    Detect face specifically for an employee with quick match check.
    
    This endpoint provides:
    - Face detection for specific employee
    - Quick preliminary match against employee's enrolled faces
    - Employee enrollment status
    - Employee-specific validation
    
    Perfect for employee-specific face validation before enrollment or verification.
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

        try:
            # Employee-specific detection with quick match
            result = await detector.detect_face_for_employee(
                image_data=image,
                company_id=company_id,
                employee_id=employee_id
            )

            # Anti-spoofing if requested
            spoof_result = None
            if check_spoofing and settings.ENABLE_ANTI_SPOOFING:
                anti_spoofing = AntiSpoofingService()
                spoof_result = await anti_spoofing.detect_spoofing(image)

            response_data = {
                "success": True,
                "data": {
                    **result,
                    "company_id": company_id,
                    "employee_id": employee_id,
                    "detection_type": "employee_specific"
                },
                "timestamp": datetime.utcnow().isoformat()
            }

            # Add match summary if available
            quick_match = result.get('quick_match')
            if quick_match:
                response_data["data"]["match_summary"] = {
                    "is_potential_match": quick_match.get('is_potential_match', False),
                    "confidence_level": quick_match.get('confidence_level', 'unknown'),
                    "similarity": quick_match.get('similarity', 0)
                }

            if spoof_result:
                response_data["data"]["spoofing"] = {
                    "is_real": spoof_result["is_real"],
                    "confidence": spoof_result["confidence"]
                }

            return response_data

        finally:
            await detector.cleanup()

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Employee detection failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/validate-enrollment")
async def validate_face_for_enrollment(
    file: UploadFile = File(..., description="Image to validate for enrollment"),
    company_id: Optional[str] = Form(None, description="Company ID for context"),
    employee_id: Optional[str] = Form(None, description="Employee ID to check existing images"),
    api_key: str = Depends(verify_api_key)
):
    """
    Validate if an image is suitable for face enrollment.
    
    Checks:
    - Single face detection
    - Face quality assessment
    - Face size requirements
    - Embedding generation capability
    - Existing enrollment limits (if employee_id provided)
    
    Returns detailed validation results with specific reasons if invalid.
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

        try:
            # Validate for enrollment
            validation_result = await detector.validate_face_for_enrollment(
                image_data=image,
                company_id=company_id,
                employee_id=employee_id
            )

            return {
                "success": True,
                "data": validation_result,
                "timestamp": datetime.utcnow().isoformat()
            }

        finally:
            await detector.cleanup()

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Enrollment validation failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check():
    """Health check for detection service"""
    return {
        "status": "healthy",
        "service": "face-detection",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0"
    }