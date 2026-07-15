# app/services/face_training.py
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime
from motor.motor_asyncio import AsyncIOMotorClient
from bson import ObjectId
import httpx

from app.config import settings
from app.services.face_detection import FaceDetectionService
from app.services.cloudinary_service import CloudinaryService
from app.utils.logger import setup_logger

logger = setup_logger(__name__)


class FaceTrainingService:
    """
    Face Training Service
    Handles face enrollment and training
    """

    def __init__(self):
        self.face_detector = FaceDetectionService()
        self.cloudinary = CloudinaryService()
        self.mongo_client = None
        self.db = None
        self.http_client = None
        self.initialized = False

    async def initialize(self):
        """Initialize connections"""
        try:
            # Initialize MongoDB
            self.mongo_client = AsyncIOMotorClient(
                settings.MONGODB_URI,
                maxPoolSize=10
            )
            self.db = self.mongo_client[settings.MONGODB_DB_NAME]

            # Initialize HTTP client for Node.js backend communication
            self.http_client = httpx.AsyncClient(
                timeout=30.0,
                headers={
                    'X-API-Key': settings.NODE_BACKEND_API_KEY,
                    'Content-Type': 'application/json'
                }
            )

            # Initialize face detector
            await self.face_detector.initialize()

            self.initialized = True
            logger.info("Face training service initialized")

        except Exception as e:
            logger.error(f"Failed to initialize training service: {str(e)}")
            raise

    async def cleanup(self):
        """Cleanup connections"""
        if self.mongo_client:
            self.mongo_client.close()
        if self.http_client:
            await self.http_client.aclose()

    async def train_single_face(
        self,
        employee_id: str,
        company_id: str,
        image_data: np.ndarray,
        metadata: Dict = None
    ) -> Dict:
        """
        Train a single face image for an employee

        Args:
            employee_id: Employee ID
            company_id: Company ID
            image_data: Face image
            metadata: Additional metadata

        Returns:
            Training result
        """
        try:
            # Step 1: Validate face
            validation = await self.face_detector.validate_face_for_enrollment(image_data)

            if not validation['is_valid']:
                return {
                    'success': False,
                    'message': validation['reason'],
                    'error_code': 'INVALID_FACE',
                    'details': validation['details']
                }

            # Step 2: Get face embedding
            faces = await self.face_detector.detect_faces(image_data, min_confidence=0.6)

            if not faces or not faces[0].get('embedding'):
                return {
                    'success': False,
                    'message': 'Failed to generate face embedding',
                    'error_code': 'EMBEDDING_FAILED'
                }

            face = faces[0]

            # Step 3: Upload to Cloudinary
            cloudinary_result = await self.cloudinary.upload_image(
                image_data,
                folder=f"face-recognition/{company_id}/{employee_id}",
                public_id=f"employee_{employee_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            )

            if not cloudinary_result['success']:
                return {
                    'success': False,
                    'message': 'Failed to upload image to Cloudinary',
                    'error_code': 'UPLOAD_FAILED'
                }

            # Step 4: Prepare face image document
            face_image_doc = {
                'url': cloudinary_result['url'],
                'cloudinaryPublicId': cloudinary_result['public_id'],
                'embedding': face['embedding'],
                'detScore': validation['details']['det_score'],
                'angle': validation['details']['angle'],
                'quality': validation['details']['quality'],
                'capturedAt': datetime.utcnow(),
                'isActive': True
            }

            # Step 5: Update MongoDB (matching your Node.js Face model)
            collection = self.db['faces']

            # Check if face profile exists
            existing = await collection.find_one({
                'employeeId': ObjectId(employee_id),
                'companyId': ObjectId(company_id)
            })

            if existing:
                # Check image count
                active_images = [img for img in existing.get('images', []) if img.get('isActive')]

                if len(active_images) >= settings.MAX_TRAINING_IMAGES:
                    return {
                        'success': False,
                        'message': f'Maximum {settings.MAX_TRAINING_IMAGES} images already enrolled',
                        'error_code': 'MAX_IMAGES_REACHED'
                    }

                # Add new image
                await collection.update_one(
                    {
                        'employeeId': ObjectId(employee_id),
                        'companyId': ObjectId(company_id)
                    },
                    {
                        '$push': {'images': face_image_doc},
                        '$set': {'updatedAt': datetime.utcnow()}
                    }
                )
            else:
                # Create new face profile
                face_profile = {
                    'companyId': ObjectId(company_id),
                    'employeeId': ObjectId(employee_id),
                    'images': [face_image_doc],
                    'isEnrolled': False,
                    'enrollmentStatus': 'pending_review',
                    'minRequiredImages': settings.MIN_TRAINING_IMAGES,
                    'matchThreshold': settings.FACE_MATCH_THRESHOLD,
                    'modelVersion': 'insightface-arcface-buffalo_l',
                    'consentGiven': True,
                    'consentGivenAt': datetime.utcnow(),
                    'createdAt': datetime.utcnow(),
                    'updatedAt': datetime.utcnow()
                }

                await collection.insert_one(face_profile)

            # Step 6: Check enrollment status
            await self._check_enrollment_status(employee_id, company_id)

            # Step 7: Notify Node.js backend
            await self._notify_backend(employee_id, company_id, 'training_completed')

            # Step 8: Invalidate recognition cache
            from app.services.face_recognition import FaceRecognitionService
            recognition_service = FaceRecognitionService()
            await recognition_service.initialize()
            await recognition_service.invalidate_cache(company_id)

            return {
                'success': True,
                'message': 'Face trained successfully',
                'data': {
                    'image_url': cloudinary_result['url'],
                    'quality': face['quality'],
                    'angle': face['angle'],
                    'det_score': round(face['det_score'], 4)
                }
            }

        except Exception as e:
            logger.error(f"Face training failed: {str(e)}", exc_info=True)
            return {
                'success': False,
                'message': f'Training failed: {str(e)}',
                'error_code': 'TRAINING_ERROR'
            }

    async def delete_face_image(
        self,
        employee_id: str,
        company_id: str,
        cloudinary_public_id: str
    ) -> Dict:
        """
        Delete a single training image — removes it from Cloudinary
        AND from the face profile's images array in MongoDB.
        """
        try:
            collection = self.db['faces']

            face_doc = await collection.find_one({
                'employeeId': ObjectId(employee_id),
                'companyId': ObjectId(company_id)
            })

            if not face_doc:
                return {
                    'success': False,
                    'message': 'Face profile not found',
                    'error_code': 'PROFILE_NOT_FOUND'
                }

            # Confirm the image actually belongs to this profile
            image_exists = any(
                img.get('cloudinaryPublicId') == cloudinary_public_id
                for img in face_doc.get('images', [])
            )

            if not image_exists:
                return {
                    'success': False,
                    'message': 'Image not found in this profile',
                    'error_code': 'IMAGE_NOT_FOUND'
                }

            # Step 1: Delete from Cloudinary first
            deleted_from_cloudinary = await self.cloudinary.delete_image(cloudinary_public_id)

            if not deleted_from_cloudinary:
                logger.warning(
                    f"Cloudinary deletion failed for {cloudinary_public_id}, "
                    f"proceeding with DB removal anyway"
                )

            # Step 2: Remove the image entry from MongoDB
            await collection.update_one(
                {
                    'employeeId': ObjectId(employee_id),
                    'companyId': ObjectId(company_id)
                },
                {
                    '$pull': {'images': {'cloudinaryPublicId': cloudinary_public_id}},
                    '$set': {'updatedAt': datetime.utcnow()}
                }
            )

            # Step 3: Re-check enrollment status since image count changed
            updated_doc = await collection.find_one({
                'employeeId': ObjectId(employee_id),
                'companyId': ObjectId(company_id)
            })
            active_images = [img for img in updated_doc.get('images', []) if img.get('isActive')]
            min_required = updated_doc.get('minRequiredImages', settings.MIN_TRAINING_IMAGES)

            if len(active_images) < min_required and updated_doc.get('isEnrolled'):
                await collection.update_one(
                    {'_id': updated_doc['_id']},
                    {
                        '$set': {
                            'isEnrolled': False,
                            'enrollmentStatus': 'needs_reenrollment'
                        }
                    }
                )
                await self._notify_backend(employee_id, company_id, 'enrollment_downgraded')

            # Step 4: Invalidate recognition cache — stale embedding must not be matched against
            from app.services.face_recognition import FaceRecognitionService
            recognition_service = FaceRecognitionService()
            await recognition_service.initialize()
            await recognition_service.invalidate_cache(company_id)

            # Step 5: Notify Node.js backend
            await self._notify_backend(employee_id, company_id, 'training_image_deleted')

            return {
                'success': True,
                'message': 'Training image deleted successfully',
                'data': {
                    'cloudinary_deleted': deleted_from_cloudinary,
                    'remaining_active_images': len(active_images)
                }
            }

        except Exception as e:
            logger.error(f"Failed to delete face image: {str(e)}", exc_info=True)
            return {
                'success': False,
                'message': f'Deletion failed: {str(e)}',
                'error_code': 'DELETE_ERROR'
            }

    async def delete_all_training_data(
        self,
        employee_id: str,
        company_id: str
    ) -> Dict:
        """
        Delete ALL training data for an employee — every Cloudinary image
        plus the entire face profile document. Use for full re-enrollment
        or offboarding (data retention / consent withdrawal).
        """
        try:
            collection = self.db['faces']

            face_doc = await collection.find_one({
                'employeeId': ObjectId(employee_id),
                'companyId': ObjectId(company_id)
            })

            if not face_doc:
                return {
                    'success': False,
                    'message': 'Face profile not found',
                    'error_code': 'PROFILE_NOT_FOUND'
                }

            images = face_doc.get('images', [])
            public_ids = [img['cloudinaryPublicId'] for img in images if img.get('cloudinaryPublicId')]

            # Step 1: Delete every image from Cloudinary in one batch call
            batch_result = await self.cloudinary.delete_images_batch(public_ids)

            # Step 2: Delete the whole profile document from MongoDB
            await collection.delete_one({'_id': face_doc['_id']})

            # Step 3: Invalidate recognition cache
            from app.services.face_recognition import FaceRecognitionService
            recognition_service = FaceRecognitionService()
            await recognition_service.initialize()
            await recognition_service.invalidate_cache(company_id)

            # Step 4: Notify Node.js backend
            await self._notify_backend(employee_id, company_id, 'training_data_deleted')

            if batch_result.get('failed'):
                logger.warning(
                    f"Some Cloudinary images failed to delete for employee {employee_id}: "
                    f"{batch_result['failed']}"
                )

            return {
                'success': True,
                'message': 'All training data deleted successfully',
                'data': {
                    'images_deleted_from_cloudinary': len(batch_result.get('deleted', [])),
                    'total_images': len(images),
                    'failed_cloudinary_deletions': batch_result.get('failed', [])
                }
            }

        except Exception as e:
            logger.error(f"Failed to delete all training data: {str(e)}", exc_info=True)
            return {
                'success': False,
                'message': f'Deletion failed: {str(e)}',
                'error_code': 'DELETE_ERROR'
            }

    async def _check_enrollment_status(self, employee_id: str, company_id: str):
        """Check and update enrollment status"""
        try:
            collection = self.db['faces']

            face_doc = await collection.find_one({
                'employeeId': ObjectId(employee_id),
                'companyId': ObjectId(company_id)
            })

            if not face_doc:
                return

            active_images = [
                img for img in face_doc.get('images', [])
                if img.get('isActive')
            ]

            min_required = face_doc.get('minRequiredImages', settings.MIN_TRAINING_IMAGES)

            if len(active_images) >= min_required:
                await collection.update_one(
                    {'_id': face_doc['_id']},
                    {
                        '$set': {
                            'isEnrolled': True,
                            'enrollmentStatus': 'approved',
                            'enrolledAt': face_doc.get('enrolledAt') or datetime.utcnow()
                        }
                    }
                )
                logger.info(f"Employee {employee_id} enrollment approved")

                # Notify Node.js backend
                await self._notify_backend(employee_id, company_id, 'enrollment_approved')

        except Exception as e:
            logger.error(f"Failed to check enrollment status: {str(e)}")

    async def _notify_backend(
        self,
        employee_id: str,
        company_id: str,
        event: str
    ):
        """Notify Node.js backend about events"""
        try:
            if not self.http_client:
                return

            url = f"{settings.NODE_BACKEND_URL}/api/face-training/callback"

            payload = {
                'employeeId': employee_id,
                'companyId': company_id,
                'event': event,
                'timestamp': datetime.utcnow().isoformat()
            }

            response = await self.http_client.post(url, json=payload)

            if response.status_code == 200:
                logger.info(f"Backend notified: {event} for employee {employee_id}")
            else:
                logger.warning(f"Backend notification failed: {response.status_code}")

        except Exception as e:
            logger.warning(f"Failed to notify backend: {str(e)}")

    async def get_training_status(self, employee_id: str, company_id: str) -> Dict:
        """Get training status for an employee"""
        try:
            collection = self.db['faces']

            face_doc = await collection.find_one({
                'employeeId': ObjectId(employee_id),
                'companyId': ObjectId(company_id)
            })

            if not face_doc:
                return {
                    'success': True,
                    'data': {
                        'exists': False,
                        'is_enrolled': False,
                        'enrollment_status': 'not_started',
                        'active_images': 0,
                        'total_images': 0
                    }
                }

            active_images = [
                img for img in face_doc.get('images', [])
                if img.get('isActive')
            ]

            return {
                'success': True,
                'data': {
                    'exists': True,
                    'is_enrolled': face_doc.get('isEnrolled', False),
                    'enrollment_status': face_doc.get('enrollmentStatus', 'not_started'),
                    'active_images': len(active_images),
                    'total_images': len(face_doc.get('images', [])),
                    'min_required': face_doc.get('minRequiredImages', settings.MIN_TRAINING_IMAGES),
                    'enrolled_at': face_doc.get('enrolledAt'),
                    'last_updated': face_doc.get('updatedAt'),
                    'is_locked': face_doc.get('isLocked', False)
                }
            }

        except Exception as e:
            logger.error(f"Failed to get training status: {str(e)}")
            raise