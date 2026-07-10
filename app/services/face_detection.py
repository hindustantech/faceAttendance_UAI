# app/services/face_detection.py
import cv2
import numpy as np
import face_recognition
from typing import List, Dict, Optional
from app.config import settings
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

class FaceDetectionService:
    """
    Face Detection Service using face_recognition library
    Works offline without downloading models
    """
    
    def __init__(self):
        self.initialized = False
        
    async def initialize(self):
        """Initialize the face detection model"""
        try:
            logger.info("Initializing face detection model...")
            
            # Test with a simple operation to ensure everything works
            test_image = np.zeros((100, 100, 3), dtype=np.uint8)
            face_recognition.face_locations(test_image)
            
            self.initialized = True
            logger.info("Face detection model initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize face detector: {str(e)}")
            raise RuntimeError(f"Face detector initialization failed: {str(e)}")
    
    async def detect_faces(
        self,
        image_data: np.ndarray,
        min_confidence: float = 0.5
    ) -> List[Dict]:
        """Detect all faces in an image"""
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
                
                face_info = {
                    'face_id': idx,
                    'bbox': [left, top, right, bottom],
                    'landmarks': self._format_landmarks(landmarks),
                    'det_score': det_score,
                    'angle': angle,
                    'quality': quality,
                    'embedding': encoding.tolist() if encoding is not None and len(encoding) > 0 else None,
                    'face_image': face_image,
                    'face_size': (face_width, face_height)
                }
                
                detected_faces.append(face_info)
            
            logger.debug(f"Detected {len(detected_faces)} faces")
            return detected_faces
            
        except Exception as e:
            logger.error(f"Face detection error: {str(e)}")
            return []
    
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
    
    async def validate_face_for_enrollment(self, image_data: np.ndarray) -> Dict:
        """Validate if image is suitable for face enrollment"""
        try:
            faces = await self.detect_faces(image_data)
            
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
            
            return {
                'is_valid': True,
                'reason': None,
                'details': {
                    'quality': face['quality'],
                    'angle': face['angle'],
                    'det_score': round(face['det_score'], 4),
                    'face_size': face['face_size'],
                    'has_embedding': True
                }
            }
            
        except Exception as e:
            logger.error(f"Face validation error: {str(e)}")
            return {
                'is_valid': False,
                'reason': f'Validation error: {str(e)}',
                'details': {}
            }
    
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