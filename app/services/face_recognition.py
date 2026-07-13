# app/services/face_recognition.py (improved version)

import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from motor.motor_asyncio import AsyncIOMotorClient
from redis import asyncio as aioredis
import json
from bson import ObjectId

from app.config import settings
from app.services.face_detection import FaceDetectionService
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

class FaceRecognitionService:
    """Face Recognition Service with 1:1 and 1:N matching"""
    
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
        employee_id: str = None,  # Added optional employee_id for 1:1 verification
        purpose: str = "punch_in",
        device_info: Dict = None
    ) -> Dict:
        """
        Verify a face - supports both 1:1 and 1:N matching
        
        If employee_id is provided: 1:1 verification (verify specific employee)
        If employee_id is None: 1:N verification (find employee in company)
        """
        start_time = datetime.utcnow()
        
        try:
            # Detect face
            faces = await self.face_detector.detect_faces(image_data)
            
            if not faces:
                return self._create_error_response(
                    "No face detected in image",
                    start_time
                )
            
            if len(faces) > 1:
                return self._create_error_response(
                    "Multiple faces detected. Please provide image with single face.",
                    start_time
                )
            
            query_face = faces[0]
            
            if not query_face.get('embedding'):
                return self._create_error_response(
                    "Failed to generate face embedding",
                    start_time
                )
            
            # If employee_id provided, do 1:1 verification
            if employee_id:
                result = await self._verify_specific_employee(
                    query_face,
                    company_id,
                    employee_id,
                    purpose,
                    device_info,
                    start_time
                )
            else:
                # Do 1:N matching against all company employees
                result = await self._verify_against_company(
                    query_face,
                    company_id,
                    purpose,
                    device_info,
                    start_time
                )
            
            return result
                
        except Exception as e:
            logger.error(f"Verification failed: {str(e)}", exc_info=True)
            raise
    
    async def _verify_specific_employee(
        self,
        query_face: Dict,
        company_id: str,
        employee_id: str,
        purpose: str,
        device_info: Dict,
        start_time: datetime
    ) -> Dict:
        """1:1 verification - verify if face matches specific employee"""
        try:
            logger.info(f"Performing 1:1 verification for employee {employee_id}")
            
            # Get specific employee's enrolled faces
            employee_faces = await self._get_employee_faces(company_id, employee_id)
            
            if not employee_faces:
                return self._create_error_response(
                    f"No enrolled faces found for employee {employee_id}",
                    start_time
                )
            
            # Check if employee is locked
            if employee_faces.get('isLocked'):
                return self._create_error_response(
                    f"Employee {employee_id} account is locked: {employee_faces.get('lockReason', 'Unknown reason')}",
                    start_time
                )
            
            # Extract embeddings from employee's images
            enrolled_embeddings = []
            for image in employee_faces.get('images', []):
                if image.get('isActive') and image.get('embedding'):
                    enrolled_embeddings.append({
                        'embedding': image['embedding'],
                        'image_id': str(image.get('_id', '')),
                        'quality': image.get('quality', 'good')
                    })
            
            if not enrolled_embeddings:
                return self._create_error_response(
                    "No valid face embeddings found for employee",
                    start_time
                )
            
            # Compare with all employee's enrolled faces
            best_match = await self._find_best_match(
                query_face['embedding'],
                enrolled_embeddings
            )
            
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            
            # Check if match meets threshold
            is_matched = best_match and best_match['similarity'] >= settings.FACE_MATCH_THRESHOLD
            
            # Log the verification attempt
            await self._log_verification(
                employee_id,
                company_id,
                is_matched,
                best_match['similarity'] if best_match else 0,
                query_face['det_score'],
                purpose,
                device_info
            )
            
            # Update consecutive failed attempts
            if not is_matched:
                await self._handle_failed_attempt(company_id, employee_id)
            
            return {
                'success': True,
                'data': {
                    'matched': is_matched,
                    'employee_id': employee_id,
                    'verification_type': '1:1',
                    'similarity': round(best_match['similarity'], 4) if best_match else 0,
                    'confidence': round(best_match['similarity'] * 100, 2) if best_match else 0,
                    'det_score': round(query_face['det_score'], 4),
                    'processing_time': round(processing_time, 3),
                    'threshold_used': settings.FACE_MATCH_THRESHOLD,
                    'best_match_image_id': best_match.get('image_id') if best_match else None
                }
            }
            
        except Exception as e:
            logger.error(f"1:1 verification failed: {str(e)}", exc_info=True)
            raise
    
    async def _verify_against_company(
        self,
        query_face: Dict,
        company_id: str,
        purpose: str,
        device_info: Dict,
        start_time: datetime
    ) -> Dict:
        """1:N verification - find best match in company"""
        try:
            logger.info(f"Performing 1:N verification for company {company_id}")
            
            # Get all enrolled faces for company
            enrolled_faces = await self._get_enrolled_faces(company_id)
            
            if not enrolled_faces:
                return self._create_error_response(
                    "No enrolled faces found in company",
                    start_time
                )
            
            # Find best match
            best_match = await self._find_best_match(
                query_face['embedding'],
                enrolled_faces
            )
            
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            
            is_matched = best_match and best_match['similarity'] >= settings.FACE_MATCH_THRESHOLD
            
            if is_matched:
                # Log successful verification
                await self._log_verification(
                    best_match['employee_id'],
                    company_id,
                    True,
                    best_match['similarity'],
                    query_face['det_score'],
                    purpose,
                    device_info
                )
                
                # Get top matches for identification
                top_matches = await self._get_top_matches(
                    query_face['embedding'],
                    enrolled_faces,
                    top_k=5
                )
                
                return {
                    'success': True,
                    'data': {
                        'matched': True,
                        'employee_id': best_match['employee_id'],
                        'verification_type': '1:N',
                        'similarity': round(best_match['similarity'], 4),
                        'confidence': round(best_match['similarity'] * 100, 2),
                        'det_score': round(query_face['det_score'], 4),
                        'processing_time': round(processing_time, 3),
                        'threshold_used': settings.FACE_MATCH_THRESHOLD,
                        'top_matches': top_matches
                    }
                }
            else:
                # Return top matches even if no match found
                top_matches = await self._get_top_matches(
                    query_face['embedding'],
                    enrolled_faces,
                    top_k=5
                )
                
                similarity = best_match['similarity'] if best_match else 0
                return {
                    'success': True,
                    'data': {
                        'matched': False,
                        'verification_type': '1:N',
                        'similarity': round(similarity, 4),
                        'processing_time': round(processing_time, 3),
                        'threshold_used': settings.FACE_MATCH_THRESHOLD,
                        'top_matches': top_matches
                    }
                }
                
        except Exception as e:
            logger.error(f"1:N verification failed: {str(e)}", exc_info=True)
            raise
    
    async def identify_face(
        self,
        image_data: np.ndarray,
        company_id: str,
        threshold: float = None,
        max_results: int = 10
    ) -> Dict:
        """
        Identify face - find top matches in company
        Always returns top matches regardless of threshold
        """
        start_time = datetime.utcnow()
        
        try:
            # Detect face
            faces = await self.face_detector.detect_faces(image_data)
            
            if not faces:
                return self._create_error_response(
                    "No face detected in image",
                    start_time
                )
            
            query_face = faces[0]
            
            if not query_face.get('embedding'):
                return self._create_error_response(
                    "Failed to generate face embedding",
                    start_time
                )
            
            # Get enrolled faces
            enrolled_faces = await self._get_enrolled_faces(company_id)
            
            if not enrolled_faces:
                return {
                    'success': True,
                    'data': {
                        'matches': [],
                        'total_enrolled': 0,
                        'processing_time': round(
                            (datetime.utcnow() - start_time).total_seconds(), 3
                        )
                    }
                }
            
            # Use provided threshold or default
            match_threshold = threshold if threshold is not None else settings.FACE_MATCH_THRESHOLD
            
            # Find all matches
            all_matches = await self._find_all_matches(
                query_face['embedding'],
                enrolled_faces,
                match_threshold
            )
            
            # Get top matches
            top_matches = all_matches[:max_results]
            
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            
            return {
                'success': True,
                'data': {
                    'matches': top_matches,
                    'total_matches': len(all_matches),
                    'total_enrolled': len(set(f['employee_id'] for f in enrolled_faces)),
                    'threshold_used': match_threshold,
                    'processing_time': round(processing_time, 3)
                }
            }
            
        except Exception as e:
            logger.error(f"Identification failed: {str(e)}", exc_info=True)
            raise
    
    async def _get_employee_faces(self, company_id: str, employee_id: str) -> Optional[Dict]:
        """Get specific employee's enrolled faces"""
        try:
            # Try cache first
            if self.redis_client:
                cache_key = f"employee_faces:{company_id}:{employee_id}"
                cached_data = await self.redis_client.get(cache_key)
                
                if cached_data:
                    return json.loads(cached_data)
            
            # Get from MongoDB
            collection = self.db['faces']
            
            document = await collection.find_one({
                'companyId': company_id,
                'employeeId': employee_id,
                'isEnrolled': True
            })
            
            if not document:
                return None
            
            # Convert ObjectId to string for JSON serialization
            result = self._serialize_document(document)
            
            # Cache if Redis is available
            if self.redis_client and result:
                await self.redis_client.setex(
                    cache_key,
                    settings.FACE_DATABASE_CACHE_TTL,
                    json.dumps(result)
                )
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to get employee faces: {str(e)}")
            return None
    
    async def _get_enrolled_faces(self, company_id: str) -> List[Dict]:
        """Get all enrolled faces for company"""
        try:
            # Try cache first
            if self.redis_client:
                cache_key = f"enrolled_faces:{company_id}"
                cached_data = await self.redis_client.get(cache_key)
                
                if cached_data:
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
                employee_id = str(document['employeeId'])
                for image in document.get('images', []):
                    if image.get('isActive') and image.get('embedding'):
                        enrolled_faces.append({
                            'employee_id': employee_id,
                            'embedding': image['embedding'],
                            'image_id': str(image.get('_id', '')),
                            'quality': image.get('quality', 'good'),
                            'det_score': image.get('detScore', 0)
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
        """Find best matching face using cosine similarity"""
        try:
            if not query_embedding or not enrolled_faces:
                return None
            
            query_vec = np.array(query_embedding, dtype=np.float64)
            # Normalize query vector
            query_vec = query_vec / np.linalg.norm(query_vec)
            
            best_match = None
            best_similarity = -1
            
            for face in enrolled_faces:
                try:
                    enrolled_vec = np.array(face['embedding'], dtype=np.float64)
                    # Normalize enrolled vector
                    enrolled_vec = enrolled_vec / np.linalg.norm(enrolled_vec)
                    
                    # Calculate cosine similarity
                    similarity = np.dot(query_vec, enrolled_vec)
                    
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_match = {
                            'employee_id': face['employee_id'],
                            'similarity': float(similarity),
                            'image_id': face['image_id'],
                            'quality': face.get('quality', 'good')
                        }
                        
                except Exception as e:
                    logger.warning(f"Error comparing face: {str(e)}")
                    continue
            
            return best_match
            
        except Exception as e:
            logger.error(f"Face matching error: {str(e)}")
            return None
    
    async def _find_all_matches(
        self,
        query_embedding: List[float],
        enrolled_faces: List[Dict],
        threshold: float
    ) -> List[Dict]:
        """Find all matching faces above threshold"""
        try:
            if not query_embedding or not enrolled_faces:
                return []
            
            query_vec = np.array(query_embedding, dtype=np.float64)
            query_vec = query_vec / np.linalg.norm(query_vec)
            
            # Group similarities by employee
            employee_matches = {}
            
            for face in enrolled_faces:
                try:
                    enrolled_vec = np.array(face['embedding'], dtype=np.float64)
                    enrolled_vec = enrolled_vec / np.linalg.norm(enrolled_vec)
                    
                    similarity = float(np.dot(query_vec, enrolled_vec))
                    
                    if similarity >= threshold:
                        employee_id = face['employee_id']
                        
                        if employee_id not in employee_matches:
                            employee_matches[employee_id] = {
                                'employee_id': employee_id,
                                'similarity': similarity,
                                'best_similarity': similarity,
                                'num_images_matched': 0,
                                'matched_images': []
                            }
                        
                        # Update best similarity
                        if similarity > employee_matches[employee_id]['best_similarity']:
                            employee_matches[employee_id]['best_similarity'] = similarity
                            employee_matches[employee_id]['similarity'] = similarity
                        
                        employee_matches[employee_id]['num_images_matched'] += 1
                        employee_matches[employee_id]['matched_images'].append({
                            'image_id': face['image_id'],
                            'similarity': similarity,
                            'quality': face.get('quality', 'good')
                        })
                        
                except Exception as e:
                    logger.warning(f"Error comparing face: {str(e)}")
                    continue
            
            # Convert to list and sort by similarity
            matches_list = list(employee_matches.values())
            matches_list.sort(key=lambda x: x['similarity'], reverse=True)
            
            # Round similarities
            for match in matches_list:
                match['similarity'] = round(match['similarity'], 4)
                match['best_similarity'] = round(match['best_similarity'], 4)
                for img in match['matched_images']:
                    img['similarity'] = round(img['similarity'], 4)
            
            return matches_list
            
        except Exception as e:
            logger.error(f"Find all matches error: {str(e)}")
            return []
    
    async def _get_top_matches(
        self,
        query_embedding: List[float],
        enrolled_faces: List[Dict],
        top_k: int = 5
    ) -> List[Dict]:
        """Get top K matches"""
        all_matches = await self._find_all_matches(
            query_embedding,
            enrolled_faces,
            threshold=0.0  # Get all matches
        )
        
        return all_matches[:top_k]
    
    async def _handle_failed_attempt(self, company_id: str, employee_id: str):
        """Handle failed verification attempt - may lock account"""
        try:
            collection = self.db['faces']
            
            # Get current failed attempts
            document = await collection.find_one({
                'companyId': company_id,
                'employeeId': employee_id
            })
            
            if document:
                consecutive_fails = document.get('consecutiveFailedAttempts', 0) + 1
                
                # Lock account after too many failed attempts
                if consecutive_fails >= settings.MAX_FAILED_ATTEMPTS:
                    await collection.update_one(
                        {'companyId': company_id, 'employeeId': employee_id},
                        {
                            '$set': {
                                'isLocked': True,
                                'lockedAt': datetime.utcnow(),
                                'lockReason': f'Too many failed attempts ({consecutive_fails})'
                            }
                        }
                    )
                    logger.warning(f"Account locked for employee {employee_id}")
                    
        except Exception as e:
            logger.error(f"Failed to handle failed attempt: {str(e)}")
    
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
                },
                upsert=True
            )
            
        except Exception as e:
            logger.error(f"Failed to log verification: {str(e)}")
    
    def _serialize_document(self, document: Dict) -> Dict:
        """Convert MongoDB document to JSON-serializable format"""
        if document is None:
            return None
            
        result = {}
        for key, value in document.items():
            if isinstance(value, ObjectId):
                result[key] = str(value)
            elif isinstance(value, datetime):
                result[key] = value.isoformat()
            elif isinstance(value, dict):
                result[key] = self._serialize_document(value)
            elif isinstance(value, list):
                result[key] = [
                    self._serialize_document(item) if isinstance(item, dict)
                    else str(item) if isinstance(item, ObjectId)
                    else item.isoformat() if isinstance(item, datetime)
                    else item
                    for item in value
                ]
            else:
                result[key] = value
        return result
    
    async def invalidate_cache(self, company_id: str = None, employee_id: str = None):
        """Invalidate cache"""
        if self.redis_client:
            try:
                if company_id and employee_id:
                    await self.redis_client.delete(f"employee_faces:{company_id}:{employee_id}")
                elif company_id:
                    await self.redis_client.delete(f"enrolled_faces:{company_id}")
                    # Also delete all employee-specific caches for this company
                    keys = await self.redis_client.keys(f"employee_faces:{company_id}:*")
                    if keys:
                        await self.redis_client.delete(*keys)
                else:
                    keys = await self.redis_client.keys("enrolled_faces:*")
                    keys += await self.redis_client.keys("employee_faces:*")
                    if keys:
                        await self.redis_client.delete(*keys)
            except Exception as e:
                logger.error(f"Failed to invalidate cache: {str(e)}")
    
    def _create_error_response(self, error: str, start_time: datetime) -> Dict:
        """Create standardized error response"""
        process_time = (datetime.utcnow() - start_time).total_seconds()
        
        return {
            'success': True,
            'data': {
                'matched': False,
                'error': error,
                'processing_time': round(process_time, 3)
            }
        }