# app/services/cloudinary_service.py
import cloudinary
import cloudinary.uploader
import cloudinary.api
from typing import Optional, Dict
import numpy as np
import cv2
import tempfile
import os
from app.config import settings
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

class CloudinaryService:
    """
    Cloudinary Service for image upload and management
    Integrates with your existing Cloudinary setup
    """
    
    def __init__(self):
        """Initialize Cloudinary configuration"""
        cloudinary.config(
            cloud_name=settings.CLOUDINARY_CLOUD_NAME,
            api_key=settings.CLOUDINARY_API_KEY,
            api_secret=settings.CLOUDINARY_API_SECRET,
            secure=True
        )
        logger.info("Cloudinary service configured")
    
    async def upload_image(
        self,
        image_data: np.ndarray,
        folder: str = "face-recognition",
        public_id: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> Dict:
        """
        Upload image to Cloudinary
        
        Args:
            image_data: Image as numpy array
            folder: Cloudinary folder
            public_id: Custom public ID
            metadata: Additional metadata
            
        Returns:
            Upload result with URL and public ID
        """
        temp_file_path = None
        
        try:
            # Save numpy array to temporary file
            with tempfile.NamedTemporaryFile(
                suffix='.jpg',
                delete=False
            ) as temp_file:
                # Convert BGR to RGB for saving
                if len(image_data.shape) == 3 and image_data.shape[2] == 3:
                    rgb_image = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)
                else:
                    rgb_image = image_data
                
                cv2.imwrite(temp_file.name, rgb_image, [cv2.IMWRITE_JPEG_QUALITY, 95])
                temp_file_path = temp_file.name
            
            # Upload options
            upload_options = {
                'folder': folder,
                'resource_type': 'image',
                'quality': 'auto:best',
                'fetch_format': 'auto',
                'overwrite': True,
                'invalidation': True,
                'tags': ['face-recognition', 'attendance-system']
            }
            
            if public_id:
                upload_options['public_id'] = public_id
            
            if metadata:
                upload_options['context'] = metadata
            
            # Upload to Cloudinary
            result = cloudinary.uploader.upload(
                temp_file_path,
                **upload_options
            )
            
            logger.info(f"Image uploaded to Cloudinary: {result['public_id']}")
            
            return {
                'success': True,
                'url': result['secure_url'],
                'public_id': result['public_id'],
                'format': result.get('format'),
                'size': result.get('bytes'),
                'width': result.get('width'),
                'height': result.get('height')
            }
            
        except Exception as e:
            logger.error(f"Failed to upload to Cloudinary: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
            
        finally:
            # Clean up temp file
            if temp_file_path and os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
    
    async def delete_image(self, public_id: str) -> bool:
        """Delete image from Cloudinary"""
        try:
            result = cloudinary.uploader.destroy(
                public_id,
                invalidate=True
            )
            
            success = result.get('result') == 'ok'
            
            if success:
                logger.info(f"Image deleted from Cloudinary: {public_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to delete from Cloudinary: {str(e)}")
            return False
    
    async def get_image_info(self, public_id: str) -> Optional[Dict]:
        """Get image information from Cloudinary"""
        try:
            result = cloudinary.api.resource(public_id)
            
            return {
                'url': result['secure_url'],
                'public_id': result['public_id'],
                'format': result.get('format'),
                'size': result.get('bytes'),
                'width': result.get('width'),
                'height': result.get('height'),
                'created_at': result.get('created_at')
            }
            
        except Exception as e:
            logger.error(f"Failed to get image info: {str(e)}")
            return None
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
            result = await collection.update_one(
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

            # Step 1: Delete every image from Cloudinary
            deleted_count = 0
            failed_deletions = []

            for img in images:
                public_id = img.get('cloudinaryPublicId')
                if not public_id:
                    continue
                success = await self.cloudinary.delete_image(public_id)
                if success:
                    deleted_count += 1
                else:
                    failed_deletions.append(public_id)

            # Step 2: Delete the whole profile document from MongoDB
            await collection.delete_one({'_id': face_doc['_id']})

            # Step 3: Invalidate recognition cache
            from app.services.face_recognition import FaceRecognitionService
            recognition_service = FaceRecognitionService()
            await recognition_service.initialize()
            await recognition_service.invalidate_cache(company_id)

            # Step 4: Notify Node.js backend
            await self._notify_backend(employee_id, company_id, 'training_data_deleted')

            if failed_deletions:
                logger.warning(
                    f"Some Cloudinary images failed to delete for employee {employee_id}: "
                    f"{failed_deletions}"
                )

            return {
                'success': True,
                'message': 'All training data deleted successfully',
                'data': {
                    'images_deleted_from_cloudinary': deleted_count,
                    'total_images': len(images),
                    'failed_cloudinary_deletions': failed_deletions
                }
            }

        except Exception as e:
            logger.error(f"Failed to delete all training data: {str(e)}", exc_info=True)
            return {
                'success': False,
                'message': f'Deletion failed: {str(e)}',
                'error_code': 'DELETE_ERROR'
            }