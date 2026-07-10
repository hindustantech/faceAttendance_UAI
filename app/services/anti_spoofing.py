# app/services/anti_spoofing.py
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
                'passed': texture_score >= 0.5
            })
            
            # 2. Color Distribution Analysis
            color_score = self._analyze_color_distribution(image_data)
            results.append({
                'method': 'color_analysis',
                'score': color_score,
                'passed': color_score >= 0.5
            })
            
            # 3. Edge Analysis
            edge_score = self._analyze_edges(image_data)
            results.append({
                'method': 'edge_analysis',
                'score': edge_score,
                'passed': edge_score >= 0.5
            })
            
            # 4. Noise Pattern Analysis
            noise_score = self._analyze_noise_pattern(image_data)
            results.append({
                'method': 'noise_analysis',
                'score': noise_score,
                'passed': noise_score >= 0.5
            })
            
            # Calculate overall score
            passed_count = sum(1 for r in results if r['passed'])
            overall_score = sum(r['score'] for r in results) / len(results)
            
            # Determine if real based on threshold
            is_real = overall_score >= settings.SPOOFING_THRESHOLD and passed_count >= 2
            
            return {
                'is_real': is_real,
                'confidence': round(overall_score, 4),
                'threshold': settings.SPOOFING_THRESHOLD,
                'details': {
                    'results': results,
                    'passed_checks': passed_count,
                    'total_checks': len(results),
                    'verdict': 'REAL' if is_real else 'SPOOF'
                }
            }
            
        except Exception as e:
            logger.error(f"Spoofing detection failed: {str(e)}")
            # Default to allow if detection fails
            return {
                'is_real': True,
                'confidence': 0.5,
                'threshold': settings.SPOOFING_THRESHOLD,
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
            
            # Normalize score
            if laplacian_var > 500:
                return 1.0
            elif laplacian_var > 100:
                return 0.7
            elif laplacian_var > 50:
                return 0.5
            else:
                return 0.3
                
        except Exception as e:
            logger.warning(f"Texture analysis failed: {str(e)}")
            return 0.5
    
    def _analyze_color_distribution(self, image: np.ndarray) -> float:
        """Analyze color distribution for screen display artifacts"""
        try:
            # Convert to HSV
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # Check saturation (screens often have different saturation)
            saturation = hsv[:, :, 1]
            saturation_mean = np.mean(saturation)
            saturation_std = np.std(saturation)
            
            # Real faces typically have higher saturation variation
            if saturation_std > 40:
                return 1.0
            elif saturation_std > 30:
                return 0.8
            elif saturation_std > 20:
                return 0.6
            else:
                return 0.4
                
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
            
            # Printed photos often have higher edge density
            if edge_density < 0.08:
                return 1.0
            elif edge_density < 0.12:
                return 0.8
            elif edge_density < 0.15:
                return 0.6
            else:
                return 0.3
                
        except Exception as e:
            logger.warning(f"Edge analysis failed: {str(e)}")
            return 0.5
    
    def _analyze_noise_pattern(self, image: np.ndarray) -> float:
        """Analyze noise pattern for digital artifacts"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply Gaussian blur and subtract
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            noise = cv2.absdiff(gray, blurred)
            
            # Calculate noise statistics
            noise_mean = np.mean(noise)
            noise_std = np.std(noise)
            
            # Real faces have natural noise patterns
            if 10 < noise_std < 30:
                return 1.0
            elif 5 < noise_std < 40:
                return 0.7
            else:
                return 0.4
                
        except Exception as e:
            logger.warning(f"Noise analysis failed: {str(e)}")
            return 0.5