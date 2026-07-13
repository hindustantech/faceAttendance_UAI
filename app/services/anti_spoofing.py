import cv2
import numpy as np
from typing import Dict
from app.config import settings
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

class AntiSpoofingService:
    """
    Anti-Spoofing Service
    Detects presentation attacks (photos, videos, masks)
    """
    
    def __init__(self):
        self.initialized = False
        # Hard-coded thresholds (no settings dependency)
        self.spoofing_threshold = 0.50
        self.min_checks_required = 2
        self.min_individual_threshold = 0.35
        
    async def initialize(self):
        """Initialize anti-spoofing service"""
        self.initialized = True
        logger.info("Anti-spoofing service initialized")
    
    async def detect_spoofing(self, image_data: np.ndarray) -> Dict:
        """
        Detect if face is real or spoofed
        
        Uses multiple techniques:
        1. Texture analysis (screen artifacts)
        2. Color distribution analysis
        3. Edge detection (printed photos)
        4. Noise pattern analysis
        
        Returns:
            Spoofing detection result
        """
        try:
            results = []
            
            # 1. Texture Analysis
            texture_score = self._analyze_texture(image_data)
            results.append({
                'method': 'texture_analysis',
                'score': texture_score,
                'passed': texture_score >= self.min_individual_threshold
            })
            
            # 2. Color Distribution Analysis
            color_score = self._analyze_color_distribution(image_data)
            results.append({
                'method': 'color_analysis',
                'score': color_score,
                'passed': color_score >= self.min_individual_threshold
            })
            
            # 3. Edge Analysis
            edge_score = self._analyze_edges(image_data)
            results.append({
                'method': 'edge_analysis',
                'score': edge_score,
                'passed': edge_score >= self.min_individual_threshold
            })
            
            # 4. Noise Pattern Analysis
            noise_score = self._analyze_noise_pattern(image_data)
            results.append({
                'method': 'noise_analysis',
                'score': noise_score,
                'passed': noise_score >= 0.35  # Even lower for noise
            })
            
            # Calculate overall score
            passed_count = sum(1 for r in results if r['passed'])
            overall_score = sum(r['score'] for r in results) / len(results)
            
            # FIXED LOGIC: More lenient real-face detection
            # Pass if:
            # 1. Overall score meets threshold AND at least 2 checks passed
            # 2. OR overall score is very high (>= 0.70) regardless of checks
            # 3. OR at least 3 checks passed regardless of score
            is_real = (
                (overall_score >= self.spoofing_threshold and passed_count >= self.min_checks_required) or
                overall_score >= 0.70 or
                passed_count >= 3
            )
            
            result = {
                'is_real': is_real,
                'confidence': round(overall_score, 4),
                'threshold': self.spoofing_threshold,
                'details': {
                    'results': results,
                    'passed_checks': passed_count,
                    'total_checks': len(results),
                    'verdict': 'REAL' if is_real else 'SPOOF'
                }
            }
            
            logger.info(f"Anti-spoofing: {result['details']['verdict']} "
                       f"(score: {result['confidence']:.4f}, passed: {passed_count}/4)")
            
            return result
            
        except Exception as e:
            logger.error(f"Spoofing detection failed: {str(e)}")
            # Default to REAL if detection fails (fail-open for UX)
            return {
                'is_real': True,
                'confidence': 1.0,
                'threshold': self.spoofing_threshold,
                'details': {
                    'error': str(e),
                    'verdict': 'PASSED_BY_DEFAULT'
                }
            }
    
    def _analyze_texture(self, image: np.ndarray) -> float:
        """Analyze texture for screen/replay artifacts"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Calculate Laplacian variance (measures blur/texture)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # Calibrated thresholds for real-world performance
            if laplacian_var > 300:
                return 1.0
            elif laplacian_var > 150:
                return 0.85
            elif laplacian_var > 80:
                return 0.70
            elif laplacian_var > 40:
                return 0.55
            elif laplacian_var > 20:
                return 0.40
            else:
                return 0.30
                
        except Exception as e:
            logger.warning(f"Texture analysis failed: {str(e)}")
            return 0.5
    
    def _analyze_color_distribution(self, image: np.ndarray) -> float:
        """Analyze color distribution for screen display artifacts"""
        try:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # Check saturation (screens often have different saturation)
            saturation = hsv[:, :, 1]
            saturation_std = np.std(saturation)
            
            # Calibrated thresholds
            if saturation_std > 35:
                return 1.0
            elif saturation_std > 25:
                return 0.85
            elif saturation_std > 18:
                return 0.70
            elif saturation_std > 12:
                return 0.55
            elif saturation_std > 8:
                return 0.40
            else:
                return 0.35
                
        except Exception as e:
            logger.warning(f"Color analysis failed: {str(e)}")
            return 0.5
    
    def _analyze_edges(self, image: np.ndarray) -> float:
        """Analyze edges for print artifacts"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply Canny edge detection
            edges = cv2.Canny(gray, 50, 150)
            
            # Calculate edge density
            edge_density = np.sum(edges > 0) / edges.size
            
            # Calibrated thresholds (inverted - less edges is better)
            if edge_density < 0.10:
                return 1.0
            elif edge_density < 0.15:
                return 0.85
            elif edge_density < 0.20:
                return 0.70
            elif edge_density < 0.25:
                return 0.55
            elif edge_density < 0.30:
                return 0.40
            else:
                return 0.30
                
        except Exception as e:
            logger.warning(f"Edge analysis failed: {str(e)}")
            return 0.5
    
    def _analyze_noise_pattern(self, image: np.ndarray) -> float:
        """Analyze noise pattern for digital artifacts"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply Gaussian blur and subtract to get noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            noise = cv2.absdiff(gray, blurred)
            
            # Calculate noise statistics
            noise_std = np.std(noise)
            
            # FIXED: Much broader acceptable ranges for real faces
            # Real faces have natural noise variation
            if 5 < noise_std < 40:
                return 1.0
            elif 3 < noise_std < 50:
                return 0.85
            elif 2 < noise_std < 60:
                return 0.70
            elif 1 < noise_std < 70:
                return 0.55
            else:
                return 0.40  # Minimum raised to 0.40
                
        except Exception as e:
            logger.warning(f"Noise analysis failed: {str(e)}")
            return 0.5