# app/routes/recognition.py (FULL FIXED VERSION)

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

# Initialize as singleton - created once when module loads
anti_spoofing_service = AntiSpoofingService()
_recognition_service = None


@router.on_event("startup")
async def startup_event():
    """Initialize services on application startup"""
    global _recognition_service
    
    logger.info("=" * 50)
    logger.info("Starting Face Recognition Services...")
    logger.info("=" * 50)
    
    # Initialize anti-spoofing service
    try:
        logger.info("Initializing anti-spoofing service...")
        success = await anti_spoofing_service.initialize()
        if success:
            logger.info("✓ Anti-spoofing service initialized successfully")
        else:
            logger.error("✗ Failed to initialize anti-spoofing service")
    except Exception as e:
        logger.error(f"✗ Anti-spoofing initialization error: {e}", exc_info=True)
    
    # Initialize recognition service
    try:
        logger.info("Initializing face recognition service...")
        _recognition_service = FaceRecognitionService()
        await _recognition_service.initialize()
        logger.info("✓ Face recognition service initialized successfully")
    except Exception as e:
        logger.error(f"✗ Face recognition initialization error: {e}", exc_info=True)
        _recognition_service = None
    
    logger.info("=" * 50)


@router.on_event("shutdown")
async def shutdown_event():
    """Cleanup services on application shutdown"""
    global _recognition_service
    
    logger.info("Shutting down face recognition services...")
    
    if _recognition_service:
        try:
            await _recognition_service.cleanup()
            logger.info("✓ Face recognition service cleaned up")
        except Exception as e:
            logger.error(f"Error cleaning up recognition service: {e}")
    
    logger.info("Shutdown complete")


async def get_recognition_service() -> FaceRecognitionService:
    """Dependency to get or create recognition service"""
    global _recognition_service
    
    if _recognition_service is None or not _recognition_service.initialized:
        logger.info("Recognition service not initialized, creating new instance...")
        _recognition_service = FaceRecognitionService()
        await _recognition_service.initialize()
    
    return _recognition_service


async def perform_anti_spoofing_check(image: np.ndarray, endpoint_name: str = "unknown"):
    """
    Centralized anti-spoofing check with fail-closed policy.
    
    Args:
        image: numpy array of the face image
        endpoint_name: name of the calling endpoint for logging
    
    Returns:
        tuple: (passed: bool, result: dict)
    
    Raises:
        HTTPException: If spoofing detected or service unavailable
    """
    try:
        # Check if anti-spoofing service is initialized
        if not anti_spoofing_service.is_initialized():
            logger.warning(
                f"[{endpoint_name}] Anti-spoofing service not initialized. "
                f"Attempting re-initialization..."
            )
            success = await anti_spoofing_service.initialize()
            if not success:
                logger.error(f"[{endpoint_name}] Failed to re-initialize anti-spoofing service")
                raise HTTPException(
                    status_code=503,
                    detail={
                        "message": "Security service temporarily unavailable. Please try again later.",
                        "error": "Anti-spoofing service initialization failed"
                    }
                )
        
        # Validate image
        if image is None or image.size == 0:
            logger.error(f"[{endpoint_name}] Invalid image provided")
            raise HTTPException(
                status_code=400,
                detail={
                    "message": "Invalid image provided",
                    "error": "Empty or corrupt image"
                }
            )
        
        # Perform anti-spoofing detection
        logger.info(f"[{endpoint_name}] Running anti-spoofing detection...")
        result = await anti_spoofing_service.detect_spoofing(image)
        
        is_real = result.get('is_real', False)
        confidence = result.get('confidence', 0.0)
        attack_type = result.get('attack_type', 'UNKNOWN')
        details = result.get('details', {})
        verdict = details.get('verdict', 'UNKNOWN')
        
        logger.info(
            f"[{endpoint_name}] Anti-spoofing results: "
            f"is_real={is_real}, confidence={confidence:.4f}, "
            f"attack_type={attack_type}, verdict={verdict}"
        )
        
        # STRICT CHECK: Only accept if explicitly marked as REAL with sufficient confidence
        min_confidence = getattr(settings, "FAS_MIN_CONFIDENCE_THRESHOLD", 0.35)
        
        if not is_real:
            logger.warning(
                f"[{endpoint_name}] ✗ SPOOFING DETECTED: "
                f"type={attack_type}, confidence={confidence:.4f}, verdict={verdict}"
            )
            raise HTTPException(
                status_code=403,
                detail={
                    "message": "Face verification failed. Please ensure you're using a real face in good lighting.",
                    "confidence": confidence,
                    "details": {
                        "verdict": verdict,
                        "attack_type": attack_type
                    }
                }
            )
        
        if confidence < min_confidence:
            logger.warning(
                f"[{endpoint_name}] ✗ LOW CONFIDENCE: confidence={confidence:.4f} < threshold={min_confidence}"
            )
            raise HTTPException(
                status_code=403,
                detail={
                    "message": "Face verification failed due to low confidence. Please try again with better lighting.",
                    "confidence": confidence,
                    "details": {
                        "verdict": verdict,
                        "attack_type": attack_type,
                        "min_required_confidence": min_confidence
                    }
                }
            )
        
        logger.info(f"[{endpoint_name}] ✓ Anti-spoofing check PASSED")
        return True, result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            f"[{endpoint_name}] Unexpected anti-spoofing error: {str(e)}", 
            exc_info=True
        )
        raise HTTPException(
            status_code=503,
            detail={
                "message": "Security verification failed. Please try again later.",
                "error": "Internal security error"
            }
        )


async def validate_and_decode_image(
    file: UploadFile, 
    endpoint_name: str = "unknown"
) -> np.ndarray:
    """
    Validate and decode uploaded image file.
    
    Args:
        file: Uploaded file object
        endpoint_name: name of the calling endpoint for logging
    
    Returns:
        numpy array of decoded image
    
    Raises:
        HTTPException: If validation fails
    """
    # Validate content type
    if file.content_type not in settings.SUPPORTED_FORMATS:
        logger.warning(
            f"[{endpoint_name}] Unsupported file type: {file.content_type}"
        )
        raise HTTPException(
            status_code=400,
            detail={
                "message": f"Unsupported file type: {file.content_type}",
                "supported_formats": settings.SUPPORTED_FORMATS
            }
        )
    
    # Validate file size
    contents = await file.read()
    max_size = getattr(settings, "MAX_UPLOAD_SIZE", 10 * 1024 * 1024)  # Default 10MB
    
    if len(contents) > max_size:
        logger.warning(
            f"[{endpoint_name}] File too large: {len(contents)} bytes (max: {max_size})"
        )
        raise HTTPException(
            status_code=400,
            detail={
                "message": f"File too large. Maximum size: {max_size / (1024*1024):.1f}MB",
                "max_size_mb": max_size / (1024*1024)
            }
        )
    
    if len(contents) == 0:
        logger.warning(f"[{endpoint_name}] Empty file uploaded")
        raise HTTPException(
            status_code=400,
            detail={"message": "Empty file uploaded"}
        )
    
    # Decode image
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if image is None:
        logger.warning(f"[{endpoint_name}] Failed to decode image")
        raise HTTPException(
            status_code=400,
            detail={"message": "Invalid or corrupt image file"}
        )
    
    # Validate image dimensions
    height, width = image.shape[:2]
    min_dimension = getattr(settings, "MIN_IMAGE_DIMENSION", 100)
    
    if height < min_dimension or width < min_dimension:
        logger.warning(
            f"[{endpoint_name}] Image too small: {width}x{height} "
            f"(min: {min_dimension}x{min_dimension})"
        )
        raise HTTPException(
            status_code=400,
            detail={
                "message": f"Image too small. Minimum dimensions: {min_dimension}x{min_dimension} pixels",
                "current_dimensions": f"{width}x{height}"
            }
        )
    
    max_dimension = getattr(settings, "MAX_IMAGE_DIMENSION", 4096)
    if height > max_dimension or width > max_dimension:
        logger.warning(
            f"[{endpoint_name}] Image too large: {width}x{height} "
            f"(max: {max_dimension}x{max_dimension})"
        )
        raise HTTPException(
            status_code=400,
            detail={
                "message": f"Image too large. Maximum dimensions: {max_dimension}x{max_dimension} pixels",
                "current_dimensions": f"{width}x{height}"
            }
        )
    
    logger.info(
        f"[{endpoint_name}] Image validated: {width}x{height}, "
        f"{len(contents)} bytes, {file.content_type}"
    )
    
    return image


def parse_device_info(device_info_str: str, endpoint_name: str = "unknown") -> dict:
    """
    Parse device info JSON string.
    
    Args:
        device_info_str: JSON string of device info
        endpoint_name: name of the calling endpoint for logging
    
    Returns:
        dict: Parsed device info
    """
    try:
        device_data = json.loads(device_info_str) if device_info_str else {}
        logger.debug(f"[{endpoint_name}] Device info parsed: {device_data}")
        return device_data
    except json.JSONDecodeError as e:
        logger.warning(
            f"[{endpoint_name}] Invalid device info JSON: {e}"
        )
        return {}


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
    Supports both 1:1 verification (with employee_id) and 1:N identification.
    """
    endpoint_name = "verify"
    
    try:
        logger.info("=" * 60)
        logger.info(f"[{endpoint_name}] Starting face verification")
        logger.info(f"[{endpoint_name}] Company: {company_id}, Employee: {employee_id}, Purpose: {purpose}")
        logger.info("=" * 60)
        
        # Step 1: Validate and decode image
        image = await validate_and_decode_image(file, endpoint_name)
        
        # Step 2: Parse device info
        device_data = parse_device_info(device_info, endpoint_name)
        
        # Step 3: MANDATORY ANTI-SPOOFING CHECK
        # This runs regardless of any settings - security first
        logger.info(f"[{endpoint_name}] Running mandatory anti-spoofing check...")
        await perform_anti_spoofing_check(image, endpoint_name)
        logger.info(f"[{endpoint_name}] Anti-spoofing check passed ✓")
        
        # Step 4: Initialize recognition service
        recognition_service = await get_recognition_service()
        
        try:
            # Step 5: Perform face verification
            result = await recognition_service.verify_face(
                image_data=image,
                company_id=company_id,
                employee_id=employee_id,
                purpose=purpose,
                device_info=device_data
            )
            
            # Step 6: Add metadata to result
            result['timestamp'] = datetime.utcnow().isoformat()
            result['security_checks'] = {
                'anti_spoofing': 'passed',
                'face_detection': 'passed',
                'image_validation': 'passed'
            }
            
            # Step 7: Add human-readable message
            if employee_id:
                result['data']['verification_type'] = '1:1'
                if result['data'].get('matched'):
                    result['message'] = f"Face successfully verified as employee {employee_id}"
                else:
                    result['message'] = f"Face does not match employee {employee_id}"
            else:
                result['data']['verification_type'] = '1:N'
                if result['data'].get('matched'):
                    matched_id = result['data'].get('employee_id', 'unknown')
                    result['message'] = f"Face identified as employee {matched_id}"
                else:
                    result['message'] = "No matching employee found in the system"
            
            logger.info("=" * 60)
            logger.info(f"[{endpoint_name}] Verification completed successfully")
            logger.info(f"[{endpoint_name}] Result: {result.get('message')}")
            logger.info("=" * 60)
            
            return result
            
        finally:
            # Don't cleanup recognition service here - it's a singleton
            pass
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            f"[{endpoint_name}] Verification failed: {str(e)}", 
            exc_info=True
        )
        raise HTTPException(
            status_code=500,
            detail={
                "message": "Face verification failed due to an internal error",
                "error": str(e) if getattr(settings, "DEBUG", False) else "Internal server error"
            }
        )


@router.post("/identify")
async def identify_face(
    company_id: str = Form(..., description="Company ID"),
    file: UploadFile = File(..., description="Face image for identification"),
    threshold: Optional[float] = Form(None, description="Custom matching threshold (0-1)"),
    max_results: int = Form(default=10, ge=1, le=50, description="Maximum number of results"),
    api_key: str = Depends(verify_api_key)
):
    """
    Identify face with MANDATORY anti-spoofing check.
    
    Always returns top matches regardless of threshold.
    Anti-spoofing is enforced before identification.
    """
    endpoint_name = "identify"
    
    try:
        logger.info("=" * 60)
        logger.info(f"[{endpoint_name}] Starting face identification")
        logger.info(f"[{endpoint_name}] Company: {company_id}, Max results: {max_results}")
        if threshold:
            logger.info(f"[{endpoint_name}] Custom threshold: {threshold}")
        logger.info("=" * 60)
        
        # Step 1: Validate and decode image
        image = await validate_and_decode_image(file, endpoint_name)
        
        # Step 2: Validate threshold if provided
        if threshold is not None:
            if not (0 <= threshold <= 1):
                raise HTTPException(
                    status_code=400,
                    detail={
                        "message": "Threshold must be between 0 and 1",
                        "invalid_value": threshold
                    }
                )
        
        # Step 3: MANDATORY ANTI-SPOOFING CHECK
        logger.info(f"[{endpoint_name}] Running mandatory anti-spoofing check...")
        await perform_anti_spoofing_check(image, endpoint_name)
        logger.info(f"[{endpoint_name}] Anti-spoofing check passed ✓")
        
        # Step 4: Initialize recognition service
        recognition_service = await get_recognition_service()
        
        try:
            # Step 5: Perform face identification
            result = await recognition_service.identify_face(
                image_data=image,
                company_id=company_id,
                threshold=threshold,
                max_results=max_results
            )
            
            # Step 6: Add metadata to result
            result['timestamp'] = datetime.utcnow().isoformat()
            result['security_checks'] = {
                'anti_spoofing': 'passed',
                'face_detection': 'passed',
                'image_validation': 'passed'
            }
            
            # Step 7: Add human-readable message
            matches = result['data'].get('matches', [])
            if matches:
                match_threshold = threshold or settings.FACE_MATCH_THRESHOLD
                good_matches = [m for m in matches if m['similarity'] >= match_threshold]
                
                if good_matches:
                    best_match = good_matches[0]
                    result['message'] = (
                        f"Found {len(good_matches)} potential matches. "
                        f"Best match: Employee {best_match['employee_id']} "
                        f"({best_match['similarity']:.2%} similarity)"
                    )
                else:
                    result['message'] = (
                        f"Found {len(matches)} potential matches, "
                        f"but none above the {match_threshold:.0%} threshold"
                    )
            else:
                result['message'] = "No matching employees found in the system"
            
            logger.info("=" * 60)
            logger.info(f"[{endpoint_name}] Identification completed successfully")
            logger.info(f"[{endpoint_name}] Found {len(matches)} potential matches")
            logger.info("=" * 60)
            
            return result
            
        finally:
            # Don't cleanup recognition service here - it's a singleton
            pass
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            f"[{endpoint_name}] Identification failed: {str(e)}", 
            exc_info=True
        )
        raise HTTPException(
            status_code=500,
            detail={
                "message": "Face identification failed due to an internal error",
                "error": str(e) if getattr(settings, "DEBUG", False) else "Internal server error"
            }
        )


@router.post("/verify-employee")
async def verify_specific_employee(
    company_id: str = Form(..., description="Company ID"),
    employee_id: str = Form(..., description="Employee ID to verify against"),
    file: UploadFile = File(..., description="Face image for verification"),
    purpose: str = Form(default="attendance", description="Verification purpose"),
    device_info: str = Form(default="{}", description="Device information (JSON)"),
    api_key: str = Depends(verify_api_key)
):
    """
    Alias endpoint: Verify if face belongs to specific employee (1:1).
    Includes mandatory anti-spoofing check.
    """
    logger.info(f"[/verify-employee] Alias called for employee {employee_id}")
    
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
    file: UploadFile = File(..., description="Face image for search"),
    threshold: Optional[float] = Form(None, description="Custom matching threshold (0-1)"),
    limit: int = Form(default=5, ge=1, le=20, description="Maximum number of results"),
    api_key: str = Depends(verify_api_key)
):
    """
    Alias endpoint: Search employee by face with anti-spoofing.
    Includes mandatory anti-spoofing check.
    """
    logger.info(f"[/search] Alias called for company {company_id}, limit {limit}")
    
    return await identify_face(
        company_id=company_id,
        file=file,
        threshold=threshold,
        max_results=limit,
        api_key=api_key
    )


@router.get("/health")
async def health_check():
    """
    Health check endpoint for face recognition services.
    """
    health_status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "services": {
            "anti_spoofing": {
                "initialized": anti_spoofing_service.is_initialized() if anti_spoofing_service else False,
                "status": "ok" if (anti_spoofing_service and anti_spoofing_service.is_initialized()) else "error"
            },
            "face_recognition": {
                "initialized": _recognition_service.initialized if _recognition_service else False,
                "status": "ok" if (_recognition_service and _recognition_service.initialized) else "error"
            }
        }
    }
    
    # Determine overall status
    services_ok = all(
        service["status"] == "ok" 
        for service in health_status["services"].values()
    )
    health_status["status"] = "healthy" if services_ok else "degraded"
    
    return health_status


@router.post("/enroll")
async def enroll_face(
    company_id: str = Form(..., description="Company ID"),
    employee_id: str = Form(..., description="Employee ID"),
    file: UploadFile = File(..., description="Face image for enrollment"),
    api_key: str = Depends(verify_api_key)
):
    """
    Enroll a new face for an employee.
    Includes mandatory anti-spoofing check.
    """
    endpoint_name = "enroll"
    
    try:
        logger.info("=" * 60)
        logger.info(f"[{endpoint_name}] Starting face enrollment")
        logger.info(f"[{endpoint_name}] Company: {company_id}, Employee: {employee_id}")
        logger.info("=" * 60)
        
        # Step 1: Validate and decode image
        image = await validate_and_decode_image(file, endpoint_name)
        
        # Step 2: MANDATORY ANTI-SPOOFING CHECK
        logger.info(f"[{endpoint_name}] Running mandatory anti-spoofing check...")
        await perform_anti_spoofing_check(image, endpoint_name)
        logger.info(f"[{endpoint_name}] Anti-spoofing check passed ✓")
        
        # Step 3: Initialize recognition service
        recognition_service = await get_recognition_service()
        
        try:
            # Step 4: Detect face and generate embedding
            faces = await recognition_service.face_detector.detect_faces(image)
            
            if not faces:
                raise HTTPException(
                    status_code=400,
                    detail={"message": "No face detected in image"}
                )
            
            if len(faces) > 1:
                raise HTTPException(
                    status_code=400,
                    detail={"message": "Multiple faces detected. Please provide image with single face."}
                )
            
            face_data = faces[0]
            
            if not face_data.get('embedding'):
                raise HTTPException(
                    status_code=400,
                    detail={"message": "Failed to generate face embedding"}
                )
            
            # Step 5: Store face embedding in database
            collection = recognition_service.db['faces']
            
            enrollment_data = {
                'employeeId': employee_id,
                'companyId': company_id,
                'isEnrolled': True,
                'enrollmentStatus': 'completed',
                'enrolledAt': datetime.utcnow(),
                'images': [{
                    'embedding': face_data['embedding'],
                    'isActive': True,
                    'quality': 'good',
                    'detScore': face_data.get('det_score', 0),
                    'enrolledAt': datetime.utcnow()
                }]
            }
            
            # Update or insert the document
            result = await collection.update_one(
                {
                    'employeeId': employee_id,
                    'companyId': company_id
                },
                {
                    '$set': {
                        'isEnrolled': True,
                        'enrollmentStatus': 'completed',
                        'enrolledAt': datetime.utcnow(),
                        'lastUpdated': datetime.utcnow()
                    },
                    '$push': {
                        'images': {
                            'embedding': face_data['embedding'],
                            'isActive': True,
                            'quality': 'good',
                            'detScore': face_data.get('det_score', 0),
                            'enrolledAt': datetime.utcnow()
                        }
                    }
                },
                upsert=True
            )
            
            logger.info(f"[{endpoint_name}] Face enrolled successfully")
            
            return {
                'success': True,
                'message': f"Face enrolled successfully for employee {employee_id}",
                'data': {
                    'employee_id': employee_id,
                    'company_id': company_id,
                    'enrollment_status': 'completed',
                    'timestamp': datetime.utcnow().isoformat(),
                    'security_checks': {
                        'anti_spoofing': 'passed',
                        'face_detection': 'passed'
                    }
                }
            }
            
        finally:
            pass
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            f"[{endpoint_name}] Enrollment failed: {str(e)}", 
            exc_info=True
        )
        raise HTTPException(
            status_code=500,
            detail={
                "message": "Face enrollment failed due to an internal error",
                "error": str(e) if getattr(settings, "DEBUG", False) else "Internal server error"
            }
        )