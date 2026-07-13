# app/services/face_detection.py
import cv2
import numpy as np
import face_recognition
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from motor.motor_asyncio import AsyncIOMotorClient
from redis import asyncio as aioredis
import json

from app.config import settings
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

class FaceDetectionService:
    """Enhanced Face Detection Service with company/employee context"""
    
    def __init__(self):
        self.initialized = False
        self.mongo_client = None
        self.redis_client = None
        self.db = None
        
    async def initialize(self):
        """Initialize the face detection model and database connections"""
        try:
            logger.info("Initializing face detection service...")
            
            # Initialize MongoDB connection (optional, for contextual checks)
            try:
                self.mongo_client = AsyncIOMotorClient(
                    settings.MONGODB_URI,
                    maxPoolSize=5,
                    serverSelectionTimeoutMS=5000
                )
                self.db = self.mongo_client[settings.MONGODB_DB_NAME]
                # Test connection
                await self.mongo_client.admin.command('ping')
                logger.info("MongoDB connection established for detection service")
            except Exception as e:
                logger.warning(f"MongoDB not available for detection service: {str(e)}")
                self.mongo_client = None
                self.db = None
            
            # Try Redis connection (optional)
            try:
                self.redis_client = await aioredis.from_url(
                    settings.REDIS_URL,
                    encoding="utf-8",
                    decode_responses=True
                )
                await self.redis_client.ping()
                logger.info("Redis connection established for detection service")
            except Exception as e:
                logger.warning(f"Redis not available for detection service: {str(e)}")
                self.redis_client = None
            
            # Test face_recognition library
            test_image = np.zeros((100, 100, 3), dtype=np.uint8)
            face_recognition.face_locations(test_image)
            
            self.initialized = True
            logger.info("Face detection service initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize face detector: {str(e)}")
            raise RuntimeError(f"Face detector initialization failed: {str(e)}")
    
    async def cleanup(self):
        """Cleanup connections"""
        if self.mongo_client:
            self.mongo_client.close()
        if self.redis_client:
            await self.redis_client.close()
    
    async def detect_faces(
        self,
        image_data: np.ndarray,
        min_confidence: float = 0.5,
        company_id: Optional[str] = None,
        employee_id: Optional[str] = None
    ) -> List[Dict]:
        """
        Detect all faces in an image with optional company/employee context
        
        Args:
            image_data: Input image as numpy array
            min_confidence: Minimum confidence threshold
            company_id: Optional company ID for contextual checks
            employee_id: Optional employee ID for employee-specific checks
        
        Returns:
            List of detected faces with metadata
        """
        if not self.initialized:
            await self.initialize()
        
        try:
            # Convert BGR to RGB
            if len(image_data.shape) == 3 and image_data.shape[2] == 3:
                rgb_image = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)
            else:
                rgb_image = image_data
            
            # Ensure uint8
            if rgb_image.dtype != np.uint8:
                rgb_image = (rgb_image * 255).astype(np.uint8) if rgb_image.max() <= 1.0 else rgb_image.astype(np.uint8)
            
            # Detect face locations
            face_locations = face_recognition.face_locations(
                rgb_image,
                number_of_times_to_upsample=1,
                model="hog"
            )
            
            if not face_locations:
                logger.debug("No faces detected")
                return []
            
            # Get face encodings
            face_encodings = face_recognition.face_encodings(
                rgb_image,
                face_locations,
                num_jitters=1
            )
            
            # Get face landmarks
            face_landmarks_list = face_recognition.face_landmarks(
                rgb_image,
                face_locations
            )
            
            # Get employee info if context provided
            employee_info = None
            if company_id and employee_id and self.db is not None:
                employee_info = await self._get_employee_info(company_id, employee_id)
            
            detected_faces = []
            
            for idx, (location, encoding, landmarks) in enumerate(
                zip(face_locations, face_encodings, face_landmarks_list)
            ):
                top, right, bottom, left = location
                
                # Ensure bounds
                top = max(0, top)
                right = min(image_data.shape[1], right)
                bottom = min(image_data.shape[0], bottom)
                left = max(0, left)
                
                face_width = right - left
                face_height = bottom - top
                
                # Assess quality
                quality = self._assess_face_quality(face_width, face_height, encoding)
                
                # Calculate angle
                angle = self._calculate_face_angle(landmarks)
                
                # Estimate confidence
                det_score = self._estimate_confidence(face_width, face_height, encoding)
                
                # Extract face region
                face_image = image_data[top:bottom, left:right]
                
                # Quick match check if employee context provided
                match_info = None
                if employee_info and encoding is not None:
                    match_info = await self._quick_match_check(
                        encoding, 
                        employee_info,
                        company_id,
                        employee_id
                    )
                
                face_info = {
                    'face_id': idx,
                    'bbox': [left, top, right, bottom],
                    'landmarks': self._format_landmarks(landmarks),
                    'det_score': det_score,
                    'angle': angle,
                    'quality': quality,
                    'embedding': encoding.tolist() if encoding is not None and len(encoding) > 0 else None,
                    'face_image': face_image,
                    'face_size': (face_width, face_height),
                    'is_enrollment_ready': self._check_enrollment_readiness(
                        face_width, face_height, quality, encoding
                    ),
                }
                
                # Add match info if available
                if match_info:
                    face_info['match_info'] = match_info
                
                # Add context
                if company_id:
                    face_info['company_id'] = company_id
                if employee_id:
                    face_info['employee_id'] = employee_id
                
                detected_faces.append(face_info)
            
            # Log detection if context provided
            if company_id and self.db is not None:
                await self._log_detection(company_id, employee_id, len(detected_faces))
            
            logger.debug(f"Detected {len(detected_faces)} faces")
            return detected_faces
            
        except Exception as e:
            logger.error(f"Face detection error: {str(e)}")
            return []
    
    async def detect_face_for_company(
        self,
        image_data: np.ndarray,
        company_id: str,
        min_confidence: float = 0.5
    ) -> Dict:
        """
        Detect faces specifically for a company context
        
        This includes additional company-specific checks like:
        - Whether the company has enrolled employees
        - Quick match against company employees
        - Company-specific thresholds
        """
        try:
            # Perform detection with company context
            faces = await self.detect_faces(
                image_data, 
                min_confidence, 
                company_id=company_id
            )
            
            # Get company-specific info
            company_info = {}
            if self.db is not None:
                company_info = await self._get_company_detection_info(company_id)
            
            return {
                'face_count': len(faces),
                'faces': self._format_faces_for_response(faces),
                'company_context': company_info,
                'is_enrollment_ready': len(faces) == 1 and faces[0].get('is_enrollment_ready', False),
            }
            
        except Exception as e:
            logger.error(f"Company-specific detection error: {str(e)}")
            raise
    
    async def detect_face_for_employee(
        self,
        image_data: np.ndarray,
        company_id: str,
        employee_id: str,
        min_confidence: float = 0.5
    ) -> Dict:
        """
        Detect faces for a specific employee with quick match check
        
        This includes:
        - Face detection
        - Quick comparison with employee's enrolled faces
        - Employee-specific validation
        """
        try:
            # Perform detection with employee context
            faces = await self.detect_faces(
                image_data, 
                min_confidence, 
                company_id=company_id,
                employee_id=employee_id
            )
            
            # Get employee-specific info
            employee_info = {}
            if self.db is not None:
                employee_info = await self._get_employee_detection_info(
                    company_id, 
                    employee_id
                )
            
            return {
                'face_count': len(faces),
                'faces': self._format_faces_for_response(faces),
                'employee_context': employee_info,
                'is_enrollment_ready': len(faces) == 1 and faces[0].get('is_enrollment_ready', False),
                'quick_match': faces[0].get('match_info') if faces else None,
            }
            
        except Exception as e:
            logger.error(f"Employee-specific detection error: {str(e)}")
            raise
    
    async def _quick_match_check(
        self,
        query_encoding: np.ndarray,
        employee_info: Dict,
        company_id: str,
        employee_id: str
    ) -> Optional[Dict]:
        """Quick preliminary match check against employee's enrolled faces"""
        try:
            if not employee_info or 'images' not in employee_info:
                return None
            
            best_similarity = 0
            matched_image_id = None
            
            query_vec = np.array(query_encoding, dtype=np.float64)
            query_vec = query_vec / np.linalg.norm(query_vec)
            
            for image in employee_info.get('images', []):
                if image.get('isActive') and image.get('embedding'):
                    enrolled_vec = np.array(image['embedding'], dtype=np.float64)
                    enrolled_vec = enrolled_vec / np.linalg.norm(enrolled_vec)
                    
                    similarity = float(np.dot(query_vec, enrolled_vec))
                    
                    if similarity > best_similarity:
                        best_similarity = similarity
                        matched_image_id = str(image.get('_id', ''))
            
            if best_similarity > 0:
                return {
                    'employee_id': employee_id,
                    'similarity': round(best_similarity, 4),
                    'matched_image_id': matched_image_id,
                    'is_potential_match': best_similarity >= settings.FACE_MATCH_THRESHOLD,
                    'confidence_level': 'high' if best_similarity > 0.8 
                                      else 'medium' if best_similarity > 0.6 
                                      else 'low'
                }
            
            return None
            
        except Exception as e:
            logger.warning(f"Quick match check failed: {str(e)}")
            return None
    
    async def _get_employee_info(
        self, 
        company_id: str, 
        employee_id: str
    ) -> Optional[Dict]:
        """Get employee face info from database"""
        try:
            if not self.db:
                return None
            
            # Try cache first
            if self.redis_client:
                cache_key = f"employee_info:{company_id}:{employee_id}"
                cached = await self.redis_client.get(cache_key)
                if cached:
                    return json.loads(cached)
            
            # Get from MongoDB
            collection = self.db['faces']
            document = await collection.find_one({
                'companyId': company_id,
                'employeeId': employee_id
            })
            
            if document:
                # Serialize for caching
                result = self._serialize_document(document)
                
                # Cache if Redis available
                if self.redis_client:
                    await self.redis_client.setex(
                        cache_key,
                        300,  # 5 minutes cache
                        json.dumps(result)
                    )
                
                return result
            
            return None
            
        except Exception as e:
            logger.warning(f"Failed to get employee info: {str(e)}")
            return None
    
    async def _get_company_detection_info(self, company_id: str) -> Dict:
        """Get company-specific information for detection context"""
        try:
            if not self.db:
                return {'company_id': company_id}
            
            # Try cache
            if self.redis_client:
                cache_key = f"company_detection_info:{company_id}"
                cached = await self.redis_client.get(cache_key)
                if cached:
                    return json.loads(cached)
            
            collection = self.db['faces']
            
            # Get counts
            total_enrolled = await collection.count_documents({
                'companyId': company_id,
                'isEnrolled': True
            })
            
            active_employees = await collection.count_documents({
                'companyId': company_id,
                'isEnrolled': True,
                'isLocked': {'$ne': True}
            })
            
            # Get recent detections
            pipeline = [
                {'$match': {'companyId': company_id}},
                {'$unwind': '$verificationLogs'},
                {'$sort': {'verificationLogs.attemptedAt': -1}},
                {'$limit': 5},
                {'$project': {
                    'employeeId': 1,
                    'matched': '$verificationLogs.matched',
                    'attemptedAt': '$verificationLogs.attemptedAt'
                }}
            ]
            
            recent_detections = []
            async for doc in collection.aggregate(pipeline):
                recent_detections.append({
                    'employee_id': doc['employeeId'],
                    'matched': doc['matched'],
                    'attempted_at': doc['attemptedAt'].isoformat() if doc.get('attemptedAt') else None
                })
            
            info = {
                'company_id': company_id,
                'total_enrolled': total_enrolled,
                'active_employees': active_employees,
                'has_enrolled_faces': total_enrolled > 0,
                'recent_detections': recent_detections[:5]
            }
            
            # Cache
            if self.redis_client:
                await self.redis_client.setex(
                    cache_key,
                    60,  # 1 minute cache
                    json.dumps(info, default=str)
                )
            
            return info
            
        except Exception as e:
            logger.warning(f"Failed to get company info: {str(e)}")
            return {'company_id': company_id, 'error': str(e)}
    
    async def _get_employee_detection_info(
        self, 
        company_id: str, 
        employee_id: str
    ) -> Dict:
        """Get employee-specific detection information"""
        try:
            if not self.db:
                return {'employee_id': employee_id}
            
            collection = self.db['faces']
            document = await collection.find_one({
                'companyId': company_id,
                'employeeId': employee_id
            })
            
            if not document:
                return {
                    'employee_id': employee_id,
                    'is_enrolled': False,
                    'message': 'Employee not found in face database'
                }
            
            return {
                'employee_id': employee_id,
                'is_enrolled': document.get('isEnrolled', False),
                'enrollment_status': document.get('enrollmentStatus', 'not_started'),
                'total_images': len(document.get('images', [])),
                'active_images': sum(1 for img in document.get('images', []) if img.get('isActive')),
                'is_locked': document.get('isLocked', False),
                'last_verified_at': document.get('lastVerifiedAt').isoformat() if document.get('lastVerifiedAt') else None,
                'total_verification_attempts': document.get('totalVerificationAttempts', 0),
                'consecutive_failed_attempts': document.get('consecutiveFailedAttempts', 0),
            }
            
        except Exception as e:
            logger.warning(f"Failed to get employee info: {str(e)}")
            return {'employee_id': employee_id, 'error': str(e)}
    
    async def _log_detection(
        self, 
        company_id: str, 
        employee_id: Optional[str], 
        face_count: int
    ):
        """Log detection event"""
        try:
            if not self.db:
                return
            
            collection = self.db['detection_logs']
            
            log_entry = {
                'companyId': company_id,
                'employeeId': employee_id,
                'face_count': face_count,
                'timestamp': datetime.utcnow(),
                'detection_type': 'employee_specific' if employee_id else 'company_wide'
            }
            
            await collection.insert_one(log_entry)
            
        except Exception as e:
            logger.warning(f"Failed to log detection: {str(e)}")
    
    def _check_enrollment_readiness(
        self,
        width: int,
        height: int,
        quality: str,
        encoding: np.ndarray
    ) -> bool:
        """Check if face is suitable for enrollment"""
        min_size = getattr(settings, 'MIN_FACE_SIZE', 100)
        
        if width < min_size or height < min_size:
            return False
        
        if quality == 'poor':
            return False
        
        if encoding is None or len(encoding) == 0:
            return False
        
        return True
    
    def _format_faces_for_response(self, faces: List[Dict]) -> List[Dict]:
        """Format face data for API response"""
        formatted = []
        for f in faces:
            face_data = {
                "bbox": f["bbox"],
                "quality": f["quality"],
                "angle": f["angle"],
                "det_score": round(f["det_score"], 4),
                "face_size": f["face_size"],
                "has_embedding": f["embedding"] is not None,
                "is_enrollment_ready": f.get("is_enrollment_ready", False),
            }
            
            if f.get("match_info"):
                face_data["match_info"] = f["match_info"]
            
            formatted.append(face_data)
        
        return formatted
    
    def _assess_face_quality(self, width: int, height: int, encoding: np.ndarray) -> str:
        """Assess face quality"""
        try:
            score = 0
            
            # Size factor
            if width >= 150 and height >= 150:
                score += 3
            elif width >= 100 and height >= 100:
                score += 2
            elif width >= 60 and height >= 60:
                score += 1
            
            # Encoding quality
            if encoding is not None and len(encoding) == 128:
                encoding_norm = np.linalg.norm(encoding)
                if encoding_norm > 0.9:
                    score += 3
                elif encoding_norm > 0.7:
                    score += 2
                elif encoding_norm > 0.5:
                    score += 1
            
            if score >= 5:
                return "good"
            elif score >= 3:
                return "acceptable"
            else:
                return "poor"
                
        except Exception:
            return "acceptable"
    
    def _calculate_face_angle(self, landmarks: Dict) -> str:
        """Calculate approximate face angle from landmarks"""
        try:
            if not landmarks or 'nose_bridge' not in landmarks:
                return "front"
            
            nose_bridge = landmarks['nose_bridge']
            if len(nose_bridge) < 3:
                return "front"
            
            top_nose = np.array(nose_bridge[0])
            bottom_nose = np.array(nose_bridge[-1])
            nose_vector = bottom_nose - top_nose
            horizontal_deviation = abs(nose_vector[0])
            
            if horizontal_deviation > 25:
                return "left" if nose_vector[0] < 0 else "right"
            else:
                return "front"
                
        except Exception:
            return "front"
    
    def _estimate_confidence(self, width: int, height: int, encoding: np.ndarray) -> float:
        """Estimate detection confidence"""
        confidence = 0.7
        
        if width > 120 and height > 120:
            confidence += 0.15
        elif width > 80 and height > 80:
            confidence += 0.1
        
        if encoding is not None:
            encoding_norm = np.linalg.norm(encoding)
            if encoding_norm > 0.9:
                confidence += 0.15
            elif encoding_norm > 0.7:
                confidence += 0.1
        
        return max(0.0, min(0.99, confidence))
    
    def _format_landmarks(self, landmarks: Dict) -> Optional[List]:
        """Convert landmarks to list format"""
        if not landmarks:
            return None
        
        try:
            formatted = []
            for feature, points in landmarks.items():
                formatted.extend([[p[0], p[1]] for p in points])
            return formatted
        except Exception:
            return None
    
    def _serialize_document(self, document: Dict) -> Dict:
        """Convert MongoDB document to JSON-serializable format"""
        from bson import ObjectId
        
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
    
    async def validate_face_for_enrollment(
        self, 
        image_data: np.ndarray,
        company_id: Optional[str] = None,
        employee_id: Optional[str] = None
    ) -> Dict:
        """Validate if image is suitable for face enrollment"""
        try:
            faces = await self.detect_faces(
                image_data, 
                company_id=company_id,
                employee_id=employee_id
            )
            
            if len(faces) == 0:
                return {
                    'is_valid': False,
                    'reason': 'No face detected in the image. Please ensure good lighting.',
                    'details': {}
                }
            
            if len(faces) > 1:
                return {
                    'is_valid': False,
                    'reason': f'Multiple faces detected ({len(faces)}). Only one face allowed.',
                    'details': {'face_count': len(faces)}
                }
            
            face = faces[0]
            
            if face['quality'] == 'poor':
                return {
                    'is_valid': False,
                    'reason': 'Face quality is poor. Please provide a clearer image.',
                    'details': {'quality': face['quality']}
                }
            
            if face['embedding'] is None:
                return {
                    'is_valid': False,
                    'reason': 'Could not generate face encoding. Please try a different image.',
                    'details': {}
                }
            
            face_width, face_height = face['face_size']
            min_size = settings.MIN_FACE_SIZE
            
            if face_width < min_size or face_height < min_size:
                return {
                    'is_valid': False,
                    'reason': f'Face too small ({face_width}x{face_height}px). Minimum: {min_size}px.',
                    'details': {'face_size': f'{face_width}x{face_height}'}
                }
            
            # Check if employee already has enough images
            if employee_id and company_id and self.db is not None:
                existing_count = await self._get_employee_image_count(company_id, employee_id)
                if existing_count >= getattr(settings, 'MAX_ENROLLMENT_IMAGES', 5):
                    return {
                        'is_valid': False,
                        'reason': f'Employee already has maximum allowed images ({existing_count}).',
                        'details': {'current_images': existing_count}
                    }
            
            return {
                'is_valid': True,
                'reason': None,
                'details': {
                    'quality': face['quality'],
                    'angle': face['angle'],
                    'det_score': round(face['det_score'], 4),
                    'face_size': face['face_size'],
                    'has_embedding': True,
                    'is_enrollment_ready': True
                }
            }
            
        except Exception as e:
            logger.error(f"Face validation error: {str(e)}")
            return {
                'is_valid': False,
                'reason': f'Validation error: {str(e)}',
                'details': {}
            }
    
    async def _get_employee_image_count(self, company_id: str, employee_id: str) -> int:
        """Get count of images for an employee"""
        try:
            if not self.db:
                return 0
            
            collection = self.db['faces']
            document = await collection.find_one({
                'companyId': company_id,
                'employeeId': employee_id
            })
            
            if document:
                return len([img for img in document.get('images', []) if img.get('isActive')])
            
            return 0
            
        except Exception as e:
            logger.warning(f"Failed to get image count: {str(e)}")
            return 0
    
    async def get_embedding(self, image_data: np.ndarray) -> Optional[List[float]]:
        """Get face embedding from image"""
        try:
            faces = await self.detect_faces(image_data)
            
            if faces and faces[0]['embedding']:
                return faces[0]['embedding']
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get embedding: {str(e)}")
            return None