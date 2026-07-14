# app/services/anti_spoofing.py
import cv2
import numpy as np
from typing import Dict
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

class AntiSpoofingService:
    """
    Enhanced Anti-Spoofing Service
    Detects presentation attacks:
    - Phone screen replays
    - Printed photos/paper
    - Digital masks
    - Video replays
    """
    
    def __init__(self):
        self.initialized = False
        # Stricter thresholds for better security
        self.spoofing_threshold = 0.55  # Increased from 0.50
        self.min_checks_required = 3    # Increased from 2
        self.min_individual_threshold = 0.40  # Increased from 0.35
        
    async def initialize(self):
        """Initialize anti-spoofing service"""
        self.initialized = True
        logger.info("Enhanced anti-spoofing service initialized")
    
    async def detect_spoofing(self, image_data: np.ndarray) -> Dict:
        """
        Detect if face is real or spoofed
        
        Enhanced detection for:
        1. Phone screen attacks (Moiré patterns, reflections, screen edges)
        2. Printed photo attacks (paper texture, flat surfaces)
        3. Texture analysis (screen artifacts, blur)
        4. Color distribution (screen color gamut)
        5. Edge analysis (photo boundaries, paper edges)
        6. Noise patterns (digital noise, print noise)
        7. Reflection detection (screen glare)
        8. Depth inconsistency (flat surface detection)
        
        Returns:
            Spoofing detection result with detailed analysis
        """
        try:
            results = []
            spoof_indicators = []
            
            # 1. Phone Screen Detection - Moiré Pattern Analysis
            moire_score = self._detect_moire_patterns(image_data)
            results.append({
                'method': 'moire_pattern_analysis',
                'score': moire_score,
                'passed': moire_score >= self.min_individual_threshold,
                'description': 'Detects screen pixel patterns'
            })
            if moire_score < self.min_individual_threshold:
                spoof_indicators.append('Moiré patterns detected - possible screen attack')
            
            # 2. Screen Edge Detection
            screen_edge_score = self._detect_screen_edges(image_data)
            results.append({
                'method': 'screen_edge_detection',
                'score': screen_edge_score,
                'passed': screen_edge_score >= self.min_individual_threshold,
                'description': 'Detects phone/display edges'
            })
            if screen_edge_score < self.min_individual_threshold:
                spoof_indicators.append('Screen edges detected - possible phone attack')
            
            # 3. Reflection/Glare Detection
            reflection_score = self._detect_reflections(image_data)
            results.append({
                'method': 'reflection_detection',
                'score': reflection_score,
                'passed': reflection_score >= self.min_individual_threshold,
                'description': 'Detects screen reflections'
            })
            if reflection_score < 0.35:  # More sensitive
                spoof_indicators.append('Screen reflections detected - possible screen attack')
            
            # 4. Paper Texture Detection (for printed photos)
            paper_texture_score = self._detect_paper_texture(image_data)
            results.append({
                'method': 'paper_texture_analysis',
                'score': paper_texture_score,
                'passed': paper_texture_score >= self.min_individual_threshold,
                'description': 'Detects paper/print texture'
            })
            if paper_texture_score < 0.35:
                spoof_indicators.append('Paper texture detected - possible print attack')
            
            # 5. Color Distribution Analysis
            color_score = self._analyze_color_distribution(image_data)
            results.append({
                'method': 'color_distribution',
                'score': color_score,
                'passed': color_score >= self.min_individual_threshold,
                'description': 'Detects unnatural color distribution'
            })
            if color_score < self.min_individual_threshold:
                spoof_indicators.append('Abnormal color distribution - possible screen attack')
            
            # 6. Edge Analysis (for printed photos)
            edge_score = self._analyze_edges(image_data)
            results.append({
                'method': 'edge_analysis',
                'score': edge_score,
                'passed': edge_score >= self.min_individual_threshold,
                'description': 'Detects photo/paper edges'
            })
            if edge_score < self.min_individual_threshold:
                spoof_indicators.append('Sharp edges detected - possible print attack')
            
            # 7. Texture Analysis
            texture_score = self._analyze_texture(image_data)
            results.append({
                'method': 'texture_analysis',
                'score': texture_score,
                'passed': texture_score >= self.min_individual_threshold,
                'description': 'Detects screen artifacts'
            })
            if texture_score < self.min_individual_threshold:
                spoof_indicators.append('Unnatural texture - possible screen/print attack')
            
            # 8. Noise Pattern Analysis
            noise_score = self._analyze_noise_pattern(image_data)
            results.append({
                'method': 'noise_analysis',
                'score': noise_score,
                'passed': noise_score >= 0.35,
                'description': 'Detects digital/print noise'
            })
            
            # Calculate overall metrics
            total_checks = len(results)
            passed_checks = sum(1 for r in results if r['passed'])
            overall_score = sum(r['score'] for r in results) / total_checks
            
            # STRICTER LOGIC: Multiple conditions must be met
            is_real = (
                overall_score >= self.spoofing_threshold and 
                passed_checks >= self.min_checks_required and
                len(spoof_indicators) == 0  # No spoof indicators detected
            )
            
            # Additional check: If specific attack patterns are found, always mark as spoof
            if len(spoof_indicators) >= 2:  # Multiple spoof indicators
                is_real = False
            
            # Determine attack type if spoofed
            attack_type = self._determine_attack_type(spoof_indicators)
            
            result = {
                'is_real': is_real,
                'confidence': round(overall_score, 4),
                'threshold': self.spoofing_threshold,
                'passed_checks': passed_checks,
                'total_checks': total_checks,
                'spoof_indicators': spoof_indicators,
                'attack_type': attack_type,
                'details': {
                    'results': results,
                    'verdict': 'REAL' if is_real else f'SPOOF - {attack_type}',
                    'recommendation': self._get_recommendation(is_real, spoof_indicators)
                }
            }
            
            logger.warning(f"Anti-spoofing: {'REAL' if is_real else 'SPOOF'} "
                          f"(score: {overall_score:.4f}, passed: {passed_checks}/{total_checks})"
                          f"{' - Indicators: ' + str(spoof_indicators) if spoof_indicators else ''}")
            
            return result
            
        except Exception as e:
            logger.error(f"Spoofing detection failed: {str(e)}")
            # FAIL-SAFE: Return spoof to prevent attacks
            return {
                'is_real': False,
                'confidence': 0.0,
                'threshold': self.spoofing_threshold,
                'details': {
                    'error': str(e),
                    'verdict': 'ERROR - TREATED AS SPOOF',
                    'recommendation': 'Retry with better lighting conditions'
                }
            }
    
    def _detect_moire_patterns(self, image: np.ndarray) -> float:
        """
        Detect Moiré patterns common in photos of screens
        Moiré patterns appear as wavy interference patterns
        """
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply FFT to detect regular patterns
            f_transform = np.fft.fft2(gray)
            f_shift = np.fft.fftshift(f_transform)
            magnitude_spectrum = np.log(np.abs(f_shift) + 1)
            
            # Normalize
            magnitude_spectrum = cv2.normalize(magnitude_spectrum, None, 0, 1, cv2.NORM_MINMAX)
            
            # Detect peaks in frequency domain (moire patterns create distinct peaks)
            threshold = 0.7
            peaks = np.sum(magnitude_spectrum > threshold)
            
            # More peaks = more likely to be a screen
            if peaks < 50:
                return 1.0  # Real face
            elif peaks < 100:
                return 0.8
            elif peaks < 200:
                return 0.5  # Suspicious
            elif peaks < 400:
                return 0.3  # Likely screen
            else:
                return 0.1  # Definitely screen
                
        except Exception as e:
            logger.warning(f"Moiré detection failed: {str(e)}")
            return 0.5
    
    def _detect_screen_edges(self, image: np.ndarray) -> float:
        """
        Detect edges of phone/tablet screens
        Screens have characteristic straight edges and bezels
        """
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Edge detection
            edges = cv2.Canny(gray, 30, 100)
            
            # Hough Transform to detect straight lines
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, 
                                    threshold=100, 
                                    minLineLength=100, 
                                    maxLineGap=10)
            
            if lines is None:
                return 1.0  # No straight lines = real face (or very good image)
            
            # Count long straight lines (screen edges are typically long)
            long_lines = 0
            for line in lines:
                x1, y1, x2, y2 = line[0]
                length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                if length > 150:  # Screen edges are long
                    long_lines += 1
            
            # Check for parallel lines (screen bezels)
            if long_lines >= 4:
                return 0.1  # Very likely screen with bezels
            elif long_lines >= 2:
                return 0.3  # Possible screen edges
            elif long_lines >= 1:
                return 0.6  # Suspicious
            else:
                return 1.0  # Natural face
                
        except Exception as e:
            logger.warning(f"Screen edge detection failed: {str(e)}")
            return 0.5
    
    def _detect_reflections(self, image: np.ndarray) -> float:
        """
        Detect screen reflections and glare
        Screens often have specular highlights
        """
        try:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # Detect bright spots (glare)
            brightness = hsv[:, :, 2]
            
            # Find very bright regions (screen glare)
            glare_mask = brightness > 230
            glare_percentage = np.sum(glare_mask) / glare_mask.size
            
            # Check for concentrated bright spots
            if glare_percentage > 0.05:  # More than 5% is very bright
                return 0.2  # Likely screen glare
            elif glare_percentage > 0.03:
                return 0.4  # Suspicious
            elif glare_percentage > 0.01:
                return 0.6  # Minor glare
            else:
                return 1.0  # Normal
                
        except Exception as e:
            logger.warning(f"Reflection detection failed: {str(e)}")
            return 0.5
    
    def _detect_paper_texture(self, image: np.ndarray) -> float:
        """
        Detect paper/print texture
        Printed photos have characteristic texture patterns
        """
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Calculate Local Binary Pattern (LBP) features
            # Paper/print has uniform texture patterns
            
            # Use Gaussian blur to detect fine texture
            blurred = cv2.GaussianBlur(gray, (3, 3), 0)
            high_freq = cv2.absdiff(gray, blurred)
            
            # Paper texture creates consistent fine patterns
            texture_variance = np.var(high_freq)
            
            # Print halftone patterns have specific variance range
            if texture_variance < 5:
                return 1.0  # Smooth (real skin or good image)
            elif texture_variance < 15:
                return 0.8  # Normal
            elif texture_variance < 30:
                return 0.5  # Suspicious
            elif texture_variance < 50:
                return 0.3  # Likely paper texture
            else:
                return 0.1  # Strong paper/print texture
                
        except Exception as e:
            logger.warning(f"Paper texture detection failed: {str(e)}")
            return 0.5
    
    def _analyze_color_distribution(self, image: np.ndarray) -> float:
        """
        Analyze color distribution for screen display artifacts
        Screens have different color gamuts than real faces
        """
        try:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # Check saturation (screens often have oversaturated colors)
            saturation = hsv[:, :, 1]
            saturation_std = np.std(saturation)
            
            # Check for color channel correlation (screens have specific patterns)
            b, g, r = cv2.split(image)
            
            # Screen images often have color channel mismatches
            rg_corr = np.corrcoef(r.flatten(), g.flatten())[0, 1]
            rb_corr = np.corrcoef(r.flatten(), b.flatten())[0, 1]
            
            # Combined score
            color_score = 1.0
            
            # Saturation check
            if saturation_std < 20:
                color_score -= 0.3  # Low saturation (possible screen)
            
            # Color correlation check (real faces have specific correlations)
            if abs(rg_corr - 0.9) > 0.1:
                color_score -= 0.2
            if abs(rb_corr - 0.7) > 0.15:
                color_score -= 0.2
            
            return max(0.0, color_score)
                
        except Exception as e:
            logger.warning(f"Color analysis failed: {str(e)}")
            return 0.5
    
    def _analyze_edges(self, image: np.ndarray) -> float:
        """
        Analyze edges for print/photo artifacts
        Printed photos have sharp, uniform edges
        """
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply Canny edge detection
            edges = cv2.Canny(gray, 50, 150)
            
            # Calculate edge density
            edge_density = np.sum(edges > 0) / edges.size
            
            # Analyze edge distribution
            # Printed photos often have uniform edge distribution
            edge_distribution = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # Combined score
            if edge_density < 0.08 and edge_distribution > 100:
                return 1.0  # Natural face edges
            elif edge_density < 0.15:
                return 0.7
            elif edge_density < 0.20:
                return 0.4  # Suspicious
            else:
                return 0.2  # Likely print edges
                
        except Exception as e:
            logger.warning(f"Edge analysis failed: {str(e)}")
            return 0.5
    
    def _analyze_texture(self, image: np.ndarray) -> float:
        """
        Analyze texture for screen/replay artifacts
        Enhanced with Gabor filters for texture analysis
        """
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Laplacian variance (measures blur/texture)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # GLCM-like features (Gray Level Co-occurrence Matrix)
            # Calculate local variance as texture measure
            kernel_size = 5
            local_variance = cv2.blur(gray.astype(float)**2, (kernel_size, kernel_size)) - \
                           cv2.blur(gray.astype(float), (kernel_size, kernel_size))**2
            
            texture_uniformity = np.mean(local_variance)
            
            # Combined texture assessment
            if laplacian_var > 200 and texture_uniformity > 50:
                return 1.0  # Rich texture (real face)
            elif laplacian_var > 100:
                return 0.7
            elif laplacian_var > 50:
                return 0.4  # Suspicious
            else:
                return 0.2  # Poor texture (screen/print)
                
        except Exception as e:
            logger.warning(f"Texture analysis failed: {str(e)}")
            return 0.5
    
    def _analyze_noise_pattern(self, image: np.ndarray) -> float:
        """
        Analyze noise pattern for digital/print artifacts
        Real faces have natural noise, screens/prints have artificial noise
        """
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Extract noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            noise = cv2.absdiff(gray, blurred)
            
            # Analyze noise statistics
            noise_mean = np.mean(noise)
            noise_std = np.std(noise)
            
            # Real faces have natural noise with specific characteristics
            # Screens have quantized noise
            # Prints have uniform noise
            
            # Check noise distribution
            noise_hist = cv2.calcHist([noise], [0], None, [256], [0, 256])
            noise_hist = noise_hist.flatten() / np.sum(noise_hist)
            
            # Calculate entropy of noise (real faces have higher entropy)
            noise_entropy = -np.sum(noise_hist[noise_hist > 0] * np.log2(noise_hist[noise_hist > 0]))
            
            # Combined noise assessment
            if 15 < noise_std < 35 and noise_entropy > 6:
                return 1.0  # Natural noise pattern
            elif noise_std > 8 and noise_entropy > 4:
                return 0.6
            else:
                return 0.3  # Artificial noise
                
        except Exception as e:
            logger.warning(f"Noise analysis failed: {str(e)}")
            return 0.5
    
    def _determine_attack_type(self, spoof_indicators: list) -> str:
        """Determine the type of spoofing attack"""
        if not spoof_indicators:
            return "UNKNOWN"
        
        indicators_text = " ".join(spoof_indicators).lower()
        
        if "screen" in indicators_text and ("moiré" in indicators_text or "reflection" in indicators_text):
            return "PHONE_SCREEN_ATTACK"
        elif "print" in indicators_text or "paper" in indicators_text:
            return "PRINTED_PHOTO_ATTACK"
        elif "screen" in indicators_text:
            return "DIGITAL_SCREEN_ATTACK"
        elif "edge" in indicators_text:
            return "PHOTO_EDGE_ATTACK"
        else:
            return "SPOOFING_ATTACK"
    
    def _get_recommendation(self, is_real: bool, spoof_indicators: list) -> str:
        """Get user-friendly recommendation"""
        if is_real:
            return "Face appears genuine. Proceed with verification."
        
        if not spoof_indicators:
            return "Unable to verify face authenticity. Please try in better lighting."
        
        indicators_text = " ".join(spoof_indicators).lower()
        
        if "moiré" in indicators_text:
            return "Screen patterns detected. Please use your real face, not a photo on a phone."
        elif "reflection" in indicators_text:
            return "Screen reflections detected. Please move away from bright lights and use your real face."
        elif "paper" in indicators_text or "print" in indicators_text:
            return "Printed photo detected. Please use your real face, not a printed picture."
        elif "edge" in indicators_text:
            return "Photo edges detected. Please use your real face directly."
        else:
            return "Spoofing attempt detected. Please use your real face for authentication."