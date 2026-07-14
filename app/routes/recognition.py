# app/routes/recognition.py (spoofing gate fixed)
#
# FIX (2026-07-14): The anti-spoofing gate previously re-derived its own
# pass/fail decision from `confidence` (the average of ALL check scores,
# critical + secondary) instead of trusting `is_real` (which is already the
# correct, stricter verdict computed by AntiSpoofingService - ALL critical
# checks - moire/glare/print - must pass). This meant a request the service
# explicitly flagged as `is_real: False, verdict: SPOOF` could still sail
# through if the noisy `confidence` average happened to land above 0.40.
# This was observed live in production logs: critical=2/3, is_real=False,
# confidence=0.8167 -> "Borderline spoof check - allowing verification" ->
# verify_face proceeded -> matched: true -> 200 OK. That is the bug this
# file fixes.
#
# FIX (2026-07-14): the anti-spoofing check previously failed OPEN on any
# internal error (service exception -> log a warning -> let the request
# through as if it were real). It now fails CLOSED: if the spoof check
# itself errors out, the request is rejected with 503 rather than silently
# treated as a verified real face. For an attendance/payroll system, an
# unverifiable request should never be treated as equivalent to a verified
# real face.
#
# FIX (2026-07-14): the spoofing-gate logic was duplicated almost
# identically in /verify and /identify. Extracted into a single
# `_run_spoof_check()` helper so there is exactly one place this decision
# is made, instead of two copies that can silently drift apart.

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


async def _run_spoof_check(image: np.ndarray, context: str = "") -> None:
    """
    Run the anti-spoofing check and enforce it.

    Raises HTTPException(403) if the image is flagged as a spoof.
    Raises HTTPException(503) if the spoof check itself fails (fail-closed -
    we do NOT treat an unverifiable check as "real").

    Trusts `is_real` from AntiSpoofingService directly. Do NOT re-derive a
    separate pass/fail decision from `confidence` here - that field is just
    an average across all checks (critical + secondary) for diagnostics/
    logging, and re-thresholding on it is what caused real spoofs to slip
    through previously. `is_real` is already the correct, stricter verdict
    (it requires ALL critical checks - screen moire, specular glare, print
    halftone - to pass).
    """
    if not settings.ENABLE_ANTI_SPOOFING:
        return

    try:
        anti_spoofing = AntiSpoofingService()
        await anti_spoofing.initialize()
        spoof_result = await anti_spoofing.detect_spoofing(image)

        is_real = spoof_result.get('is_real', False)  # default False: unverifiable != real
        confidence = spoof_result.get('confidence', 0.0)
        details = spoof_result.get('details', {})
        verdict = details.get('verdict', 'UNKNOWN')

        logger.info(
            f"Anti-spoofing check{f' ({context})' if context else ''}: "
            f"verdict={verdict}, confidence={confidence:.4f}, is_real={is_real}"
        )

        if not is_real:
            logger.warning(f"Spoofing detected{f' ({context})' if context else ''}: {details}")
            raise HTTPException(
                status_code=403,
                detail={
                    "message": "Spoofing detected. Please use a real face.",
                    "confidence": confidence,
                    "details": details
                }
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Anti-spoofing check failed with error: {str(e)}", exc_info=True)
        # FAIL CLOSED: an error means we could not verify the image is real.
        # That is not the same thing as "verified real" - do not let it
        # through. Ask the caller to retry instead.
        raise HTTPException(
            status_code=503,
            detail="Anti-spoofing check unavailable. Please try again."
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
    Verify face for attendance

    Supports two modes:
    1. 1:1 Verification (employee_id provided): Verify if face matches specific employee
    2. 1:N Verification (employee_id not provided): Find matching employee in company

    This endpoint:
    1. Validates the image
    2. Performs anti-spoofing check (if enabled) - rejects with 403 if spoofed,
       fails closed (503) if the check itself errors
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

        # Anti-spoofing gate - raises HTTPException(403) on spoof,
        # HTTPException(503) if the check itself fails. No bypass.
        await _run_spoof_check(image, context="verify")

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

            # If verify_face itself returned an error (no face detected,
            # locked account, etc.), don't overwrite it with a generic
            # matched/not-matched message.
            if 'error' in result.get('data', {}):
                result['message'] = result['data']['error']
            elif employee_id:
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

        # Anti-spoofing gate - same enforcement as /verify, no bypass.
        await _run_spoof_check(image, context="identify")

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
            effective_threshold = threshold if threshold is not None else settings.FACE_MATCH_THRESHOLD
            result['query_info'] = {
                'threshold_used': effective_threshold,
                'max_results': max_results
            }

            # Add summary message. Use .get() since identify_face's error
            # path (e.g. "no face detected") does not include a 'matches' key.
            matches = result['data'].get('matches', [])
            if 'error' in result.get('data', {}):
                result['message'] = result['data']['error']
            elif matches:
                result['message'] = f"Found {len(matches)} potential matches"
                if matches[0]['similarity'] >= effective_threshold:
                    result['message'] += (
                        f". Best match: employee {matches[0]['employee_id']} "
                        f"with {matches[0]['similarity']*100:.2f}% confidence"
                    )
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