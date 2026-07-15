# app/routes/training.py
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends, Query
from typing import List, Optional
import numpy as np
import cv2
from datetime import datetime

from app.middleware.auth import verify_api_key
from app.services.face_training import FaceTrainingService
from app.services.anti_spoofing import AntiSpoofingService
from app.config import settings
from app.utils.logger import setup_logger

logger = setup_logger(__name__)
router = APIRouter()

@router.post("/single")
async def train_single_face(
    employee_id: str = Form(..., description="Employee ID"),
    company_id: str = Form(..., description="Company ID"),
    file: UploadFile = File(..., description="Face image file"),
    api_key: str = Depends(verify_api_key)
):
    """
    Train a single face image for an employee

    This endpoint:
    1. Validates the face image
    2. Extracts face embedding
    3. Uploads to Cloudinary
    4. Stores in MongoDB
    5. Checks enrollment status
    """
    try:
        # Validate file type
        if file.content_type not in settings.SUPPORTED_FORMATS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {file.content_type}. Supported: {', '.join(settings.SUPPORTED_FORMATS)}"
            )

        # Validate file size
        file_size_mb = file.size / (1024 * 1024) if file.size else 0
        if file_size_mb > settings.MAX_IMAGE_SIZE_MB:
            raise HTTPException(
                status_code=400,
                detail=f"File too large: {file_size_mb:.1f}MB. Maximum: {settings.MAX_IMAGE_SIZE_MB}MB"
            )

        # Read image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")

        # Anti-spoofing check (optional for training)
        if settings.ENABLE_ANTI_SPOOFING:
            anti_spoofing = AntiSpoofingService()
            spoof_result = await anti_spoofing.detect_spoofing(image)

            if not spoof_result['is_real']:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "message": "Spoofing detected. Please use a real face for enrollment.",
                        "spoof_details": spoof_result['details']
                    }
                )

        # Train face
        training_service = FaceTrainingService()
        await training_service.initialize()

        result = await training_service.train_single_face(
            employee_id=employee_id,
            company_id=company_id,
            image_data=image,
            metadata={
                'filename': file.filename,
                'content_type': file.content_type,
                'file_size': file.size
            }
        )

        if not result['success']:
            raise HTTPException(
                status_code=400,
                detail=result
            )

        return {
            "success": True,
            "message": result['message'],
            "data": result['data'],
            "timestamp": datetime.utcnow().isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Training failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/batch")
async def train_batch_faces(
    employee_id: str = Form(...),
    company_id: str = Form(...),
    files: List[UploadFile] = File(...),
    api_key: str = Depends(verify_api_key)
):
    """
    Train multiple face images in batch
    Maximum 5 images allowed
    """
    try:
        # Validate number of files
        if len(files) > settings.MAX_TRAINING_IMAGES:
            raise HTTPException(
                status_code=400,
                detail=f"Maximum {settings.MAX_TRAINING_IMAGES} images allowed. Received: {len(files)}"
            )

        if len(files) == 0:
            raise HTTPException(status_code=400, detail="No files provided")

        training_service = FaceTrainingService()
        await training_service.initialize()

        results = []
        successful = 0
        failed = 0

        for file in files:
            try:
                # Read image
                contents = await file.read()
                nparr = np.frombuffer(contents, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                if image is None:
                    results.append({
                        'filename': file.filename,
                        'success': False,
                        'error': 'Invalid image file'
                    })
                    failed += 1
                    continue

                # Train face
                result = await training_service.train_single_face(
                    employee_id=employee_id,
                    company_id=company_id,
                    image_data=image,
                    metadata={'filename': file.filename}
                )

                results.append({
                    'filename': file.filename,
                    'success': result['success'],
                    'message': result.get('message'),
                    'data': result.get('data')
                })

                if result['success']:
                    successful += 1
                else:
                    failed += 1

            except Exception as e:
                results.append({
                    'filename': file.filename,
                    'success': False,
                    'error': str(e)
                })
                failed += 1

        return {
            "success": True,
            "message": f"Batch training completed: {successful} successful, {failed} failed",
            "data": {
                "total": len(files),
                "successful": successful,
                "failed": failed,
                "results": results
            },
            "timestamp": datetime.utcnow().isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch training failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status/{employee_id}")
async def get_training_status(
    employee_id: str,
    company_id: str = Query(..., description="Company ID"),
    api_key: str = Depends(verify_api_key)
):
    """
    Get face training status for an employee
    """
    try:
        training_service = FaceTrainingService()
        await training_service.initialize()

        result = await training_service.get_training_status(employee_id, company_id)

        return {
            "success": True,
            "data": result['data'],
            "timestamp": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"Failed to get status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/image")
async def delete_training_image(
    employee_id: str = Query(..., description="Employee ID"),
    company_id: str = Query(..., description="Company ID"),
    cloudinary_public_id: str = Query(..., description="Cloudinary public_id of the image to delete"),
    api_key: str = Depends(verify_api_key)
):
    """
    Delete a single training image.
    Removes it from Cloudinary and from the employee's face profile.
    Enrollment status auto-downgrades if remaining images fall below the minimum.
    """
    try:
        training_service = FaceTrainingService()
        await training_service.initialize()

        result = await training_service.delete_face_image(
            employee_id=employee_id,
            company_id=company_id,
            cloudinary_public_id=cloudinary_public_id
        )

        if not result['success']:
            status_code = 404 if result.get('error_code') in ('PROFILE_NOT_FOUND', 'IMAGE_NOT_FOUND') else 400
            raise HTTPException(status_code=status_code, detail=result)

        return {
            "success": True,
            "message": result['message'],
            "data": result['data'],
            "timestamp": datetime.utcnow().isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete training image: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/{employee_id}")
async def delete_all_training_data(
    employee_id: str,
    company_id: str = Query(..., description="Company ID"),
    api_key: str = Depends(verify_api_key)
):
    """
    Delete ALL training data for an employee.
    Wipes every Cloudinary image and the entire face profile document.
    Use for full re-enrollment resets or offboarding / consent withdrawal.
    This is irreversible.
    """
    try:
        training_service = FaceTrainingService()
        await training_service.initialize()

        result = await training_service.delete_all_training_data(
            employee_id=employee_id,
            company_id=company_id
        )

        if not result['success']:
            status_code = 404 if result.get('error_code') == 'PROFILE_NOT_FOUND' else 400
            raise HTTPException(status_code=status_code, detail=result)

        return {
            "success": True,
            "message": result['message'],
            "data": result['data'],
            "timestamp": datetime.utcnow().isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete all training data: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))