# app/services/face_recognition.py
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime
from motor.motor_asyncio import AsyncIOMotorClient
from redis import asyncio as aioredis
import json

from app.config import settings
from app.services.face_detection import FaceDetectionService
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

class FaceRecognitionService:
    """Face Recognition Service"""
    
    def __init__(self):
        self.face_detector = FaceDetectionService()
        self.mongo_client = None
        self.redis_client = None
        self.db = None
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
            
            # Try Redis, but continue if not available
            try:
                self.redis_client = await aioredis.from_url(
                    settings.REDIS_URL,
                    encoding="utf-8",
                    decode_responses=True
                )
                await self.redis_client.ping()
                logger.info("Redis connection established")
            except Exception as e:
                logger.warning(f"Redis not available: {str(e)}")
                self.redis_client = None
            
            # Initialize face detector
            await self.face_detector.initialize()
            
            self.initialized = True
            logger.info("Face recognition service initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize recognition service: {str(e)}")
            raise
    
    async def cleanup(self):
        """Cleanup connections"""
        if self.mongo_client:
            self.mongo_client.close()
        if self.redis_client:
            await self.redis_client.close()
    
    async def verify_face(
        self,
        image_data: np.ndarray,
        company_id: str,
        purpose: str = "punch_in",
        device_info: Dict = None
    ) -> Dict:
        """Verify a face against enrolled faces"""
        start_time = datetime.utcnow()
        
        try:
            # Detect face
            faces = await self.face_detector.detect_faces(image_data)
            
            if not faces:
                return self._create_response(False, "No face detected", start_time)
            
            if len(faces) > 1:
                return self._create_response(False, "Multiple faces detected", start_time)
            
            query_face = faces[0]
            
            if not query_face['embedding']:
                return self._create_response(False, "Failed to generate embedding", start_time)
            
            # Get enrolled faces
            enrolled_faces = await self._get_enrolled_faces(company_id)
            
            if not enrolled_faces:
                return self._create_response(False, "No enrolled faces found", start_time)
            
            # Find best match
            best_match = await self._find_best_match(
                query_face['embedding'],
                enrolled_faces
            )
            
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            
            if best_match and best_match['similarity'] >= settings.FACE_MATCH_THRESHOLD:
                await self._log_verification(
                    best_match['employee_id'],
                    company_id,
                    True,
                    best_match['similarity'],
                    query_face['det_score'],
                    purpose,
                    device_info
                )
                
                return {
                    'success': True,
                    'data': {
                        'matched': True,
                        'employee_id': best_match['employee_id'],
                        'similarity': round(best_match['similarity'], 4),
                        'confidence': round(best_match['similarity'] * 100, 2),
                        'det_score': round(query_face['det_score'], 4),
                        'processing_time': round(processing_time, 3)
                    }
                }
            else:
                similarity = best_match['similarity'] if best_match else 0
                return {
                    'success': True,
                    'data': {
                        'matched': False,
                        'similarity': round(similarity, 4),
                        'processing_time': round(processing_time, 3)
                    }
                }
                
        except Exception as e:
            logger.error(f"Verification failed: {str(e)}", exc_info=True)
            raise
    
    async def _get_enrolled_faces(self, company_id: str) -> List[Dict]:
        """Get enrolled faces from cache or database"""
        try:
            # Try cache first
            if self.redis_client:
                cache_key = f"enrolled_faces:{company_id}"
                cached_data = await self.redis_client.get(cache_key)
                
                if cached_data:
                    logger.debug(f"Loaded from cache for company {company_id}")
                    return json.loads(cached_data)
            
            # Get from MongoDB
            collection = self.db['faces']
            
            cursor = collection.find({
                'companyId': company_id,
                'isEnrolled': True,
                'enrollmentStatus': 'approved',
                'isLocked': {'$ne': True}
            })
            
            enrolled_faces = []
            
            async for document in cursor:
                for image in document.get('images', []):
                    if image.get('isActive') and image.get('embedding'):
                        enrolled_faces.append({
                            'employee_id': str(document['employeeId']),
                            'embedding': image['embedding'],
                            'image_id': str(image.get('_id', '')),
                            'quality': image.get('quality', 'good')
                        })
            
            # Cache if Redis is available
            if self.redis_client and enrolled_faces:
                await self.redis_client.setex(
                    cache_key,
                    settings.FACE_DATABASE_CACHE_TTL,
                    json.dumps(enrolled_faces)
                )
            
            logger.info(f"Loaded {len(enrolled_faces)} enrolled faces for company {company_id}")
            return enrolled_faces
            
        except Exception as e:
            logger.error(f"Failed to get enrolled faces: {str(e)}")
            return []
    
    async def _find_best_match(
        self,
        query_embedding: List[float],
        enrolled_faces: List[Dict]
    ) -> Optional[Dict]:
        """Find best matching face using Euclidean distance"""
        try:
            if not query_embedding or not enrolled_faces:
                return None
            
            query_vec = np.array(query_embedding, dtype=np.float64)
            
            best_match = None
            best_similarity = -1
            
            for face in enrolled_faces:
                try:
                    enrolled_vec = np.array(face['embedding'], dtype=np.float64)
                    
                    # Calculate Euclidean distance
                    distance = np.linalg.norm(query_vec - enrolled_vec)
                    
                    # Convert to similarity (0-1)
                    similarity = 1.0 / (1.0 + distance)
                    
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_match = {
                            'employee_id': face['employee_id'],
                            'similarity': float(similarity),
                            'distance': float(distance),
                            'image_id': face['image_id']
                        }
                        
                except Exception as e:
                    logger.warning(f"Error comparing face: {str(e)}")
                    continue
            
            return best_match
            
        except Exception as e:
            logger.error(f"Face matching error: {str(e)}")
            return None
    
    async def _log_verification(
        self,
        employee_id: str,
        company_id: str,
        matched: bool,
        similarity: float,
        det_score: float,
        purpose: str,
        device_info: Dict = None
    ):
        """Log verification attempt"""
        try:
            collection = self.db['faces']
            
            log_entry = {
                'attemptedAt': datetime.utcnow(),
                'matched': matched,
                'similarity': similarity,
                'detScore': det_score,
                'purpose': purpose,
                'deviceInfo': device_info or {}
            }
            
            await collection.update_one(
                {'employeeId': employee_id, 'companyId': company_id},
                {
                    '$push': {
                        'verificationLogs': {
                            '$each': [log_entry],
                            '$slice': -100
                        }
                    },
                    '$set': {'lastVerifiedAt': datetime.utcnow()},
                    '$inc': {
                        'totalVerificationAttempts': 1,
                        'totalFailedAttempts': 0 if matched else 1,
                        'consecutiveFailedAttempts': 0 if matched else 1
                    }
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to log verification: {str(e)}")
    
    async def invalidate_cache(self, company_id: str = None):
        """Invalidate cache"""
        if self.redis_client:
            try:
                if company_id:
                    await self.redis_client.delete(f"enrolled_faces:{company_id}")
                else:
                    keys = await self.redis_client.keys("enrolled_faces:*")
                    if keys:
                        await self.redis_client.delete(*keys)
            except Exception as e:
                logger.error(f"Failed to invalidate cache: {str(e)}")
    
    def _create_response(self, matched: bool, error: str, start_time: datetime) -> Dict:
        """Create standardized response"""
        process_time = (datetime.utcnow() - start_time).total_seconds()
        
        return {
            'success': True,
            'data': {
                'matched': matched,
                'error': error,
                'processing_time': round(process_time, 3)
            }
        }