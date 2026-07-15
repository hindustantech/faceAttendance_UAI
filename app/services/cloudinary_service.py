# app/services/cloudinary_service.py
import cloudinary
import cloudinary.uploader
import cloudinary.api
from typing import Optional, Dict, List
import numpy as np
import cv2
import tempfile
import os
from app.config import settings
from app.utils.logger import setup_logger

logger = setup_logger(__name__)


class CloudinaryService:
    """
    Cloudinary Service for image upload and management.
    This class is intentionally storage-only — it knows nothing about
    MongoDB, employees, or face profiles. That logic lives in
    FaceTrainingService, which calls into this class.
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
        """Delete a single image from Cloudinary"""
        try:
            result = cloudinary.uploader.destroy(
                public_id,
                invalidate=True
            )

            success = result.get('result') == 'ok'

            if success:
                logger.info(f"Image deleted from Cloudinary: {public_id}")
            else:
                logger.warning(
                    f"Cloudinary destroy did not return 'ok' for {public_id}: {result}"
                )

            return success

        except Exception as e:
            logger.error(f"Failed to delete from Cloudinary: {str(e)}")
            return False

    async def delete_images_batch(self, public_ids: List[str]) -> Dict:
        """
        Delete multiple images from Cloudinary in as few API calls as
        possible, using the Admin API's bulk delete_resources.

        Cloudinary's delete_resources endpoint accepts at most 100
        public_ids per call, so this chunks larger lists automatically.

        Args:
            public_ids: List of Cloudinary public IDs to delete

        Returns:
            Dict with 'deleted' (list of public_ids successfully deleted)
            and 'failed' (list of public_ids that could not be deleted)
        """
        deleted: List[str] = []
        failed: List[str] = []

        if not public_ids:
            return {'deleted': deleted, 'failed': failed}

        # Drop empty/None entries defensively
        clean_ids = [pid for pid in public_ids if pid]

        chunk_size = 100
        for i in range(0, len(clean_ids), chunk_size):
            chunk = clean_ids[i:i + chunk_size]
            try:
                result = cloudinary.api.delete_resources(chunk, invalidate=True)
                deleted_map = result.get('deleted', {})

                for public_id in chunk:
                    status = deleted_map.get(public_id)
                    if status == 'deleted' or status == 'not_found':
                        # 'not_found' means it's already gone — treat as deleted
                        deleted.append(public_id)
                    else:
                        failed.append(public_id)

            except Exception as e:
                logger.error(
                    f"Batch delete failed for chunk starting at index {i}: {str(e)}"
                )
                failed.extend(chunk)

        if failed:
            logger.warning(f"Cloudinary batch delete had failures: {failed}")
        logger.info(f"Cloudinary batch delete: {len(deleted)} deleted, {len(failed)} failed")

        return {'deleted': deleted, 'failed': failed}

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