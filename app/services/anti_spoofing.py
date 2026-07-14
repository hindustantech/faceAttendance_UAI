# app/services/anti_spoofing.py

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
from app.config import settings
from app.utils.logger import setup_logger

logger = setup_logger(__name__)


class AntiSpoofingService:
    """
    Enhanced Anti-Spoofing Service
    
    Multi-layer detection for presentation attacks:
    1. Screen/Phone Detection - Moiré patterns, pixel grid, refresh artifacts
    2. Print/Paper Detection - Halftone dots, paper texture, lack of depth
    3. Liveness Detection - Micro-texture, natural skin characteristics
    """

    def __init__(self):
        self.initialized = False
        
        # Primary detector thresholds
        self.screen_reject_threshold = 0.40
        self.print_reject_threshold = 0.40
        
        # Secondary corroborating signals
        self.min_secondary_threshold = 0.30
        
        # Weighting factors
        self.primary_weight = 0.70
        self.secondary_weight = 0.30

    async def initialize(self):
        """Initialize the anti-spoofing service"""
        self.initialized = True
        logger.info("Enhanced anti-spoofing service initialized")

    async def detect_spoofing(self, image_data: np.ndarray) -> Dict:
        """
        Multi-layer spoofing detection.
        
        Returns detailed analysis with per-check scores and final verdict.
        """
        try:
            # Validate image
            if image_data is None or image_data.size == 0:
                raise ValueError("Invalid image data")
            
            # Resize for consistent processing if too large
            h, w = image_data.shape[:2]
            if h > 1000 or w > 1000:
                scale = 1000 / max(h, w)
                new_h, new_w = int(h * scale), int(w * scale)
                image_data = cv2.resize(image_data, (new_w, new_h))
            
            results = []
            
            # =================================================================
            # LAYER 1: SCREEN/PHONE DETECTION (PRIMARY)
            # =================================================================
            
            # 1a. Moiré pattern detection
            moire_score = self._detect_screen_moire_enhanced(image_data)
            results.append({
                'method': 'screen_moire',
                'score': moire_score,
                'passed': moire_score >= self.screen_reject_threshold,
                'critical': True,
                'weight': 0.25
            })
            
            # 1b. Specular glare (glass reflection)
            glare_score = self._detect_specular_glare_enhanced(image_data)
            results.append({
                'method': 'specular_glare',
                'score': glare_score,
                'passed': glare_score >= self.screen_reject_threshold,
                'critical': True,
                'weight': 0.20
            })
            
            # 1c. Screen edge/border detection
            border_score = self._detect_screen_borders(image_data)
            results.append({
                'method': 'screen_borders',
                'score': border_score,
                'passed': border_score >= 0.35,
                'critical': True,
                'weight': 0.15
            })
            
            # =================================================================
            # LAYER 2: PRINT/PAPER DETECTION (PRIMARY)
            # =================================================================
            
            # 2a. Halftone/print dot patterns
            halftone_score = self._detect_halftone_patterns(image_data)
            results.append({
                'method': 'halftone_pattern',
                'score': halftone_score,
                'passed': halftone_score >= self.print_reject_threshold,
                'critical': True,
                'weight': 0.20
            })
            
            # 2b. Paper texture analysis
            paper_score = self._detect_paper_texture(image_data)
            results.append({
                'method': 'paper_texture',
                'score': paper_score,
                'passed': paper_score >= self.print_reject_threshold,
                'critical': True,
                'weight': 0.20
            })
            
            # =================================================================
            # LAYER 3: SECONDARY CHECKS
            # =================================================================
            
            # 3a. Micro-texture analysis
            texture_score = self._analyze_micro_texture(image_data)
            results.append({
                'method': 'micro_texture',
                'score': texture_score,
                'passed': texture_score >= self.min_secondary_threshold,
                'critical': False,
                'weight': 0.40
            })
            
            # 3b. Color distribution
            color_score = self._analyze_color_distribution(image_data)
            results.append({
                'method': 'color_distribution',
                'score': color_score,
                'passed': color_score >= self.min_secondary_threshold,
                'critical': False,
                'weight': 0.30
            })
            
            # 3c. Noise pattern analysis
            noise_score = self._analyze_noise_pattern(image_data)
            results.append({
                'method': 'noise_pattern',
                'score': noise_score,
                'passed': noise_score >= self.min_secondary_threshold,
                'critical': False,
                'weight': 0.30
            })
            
            # Calculate weighted scores
            critical_results = [r for r in results if r['critical']]
            secondary_results = [r for r in results if not r['critical']]
            
            # Primary score
            primary_scores = []
            primary_weights = []
            for r in critical_results:
                primary_scores.append(r['score'] * r['weight'])
                primary_weights.append(r['weight'])
            
            if primary_weights:
                primary_score = sum(primary_scores) / sum(primary_weights)
            else:
                primary_score = 0.5
            
            # Secondary score
            secondary_scores = []
            secondary_weights = []
            for r in secondary_results:
                secondary_scores.append(r['score'] * r['weight'])
                secondary_weights.append(r['weight'])
            
            if secondary_weights:
                secondary_score = sum(secondary_scores) / sum(secondary_weights)
            else:
                secondary_score = 0.5
            
            # Final combined score
            final_score = (primary_score * self.primary_weight + 
                         secondary_score * self.secondary_weight)
            
            # Decision logic
            screen_checks = [r for r in critical_results 
                           if r['method'] in ['screen_moire', 'specular_glare', 'screen_borders']]
            print_checks = [r for r in critical_results 
                          if r['method'] in ['halftone_pattern', 'paper_texture']]
            
            screen_spoof = any(r['score'] < 0.25 for r in screen_checks)
            print_spoof = any(r['score'] < 0.25 for r in print_checks)
            
            if screen_spoof:
                is_real = False
                attack_type = "SCREEN_REPLAY"
                confidence = 0.1
            elif print_spoof:
                is_real = False
                attack_type = "PRINT_PHOTO"
                confidence = 0.1
            elif primary_score < self.screen_reject_threshold:
                is_real = False
                attack_type = "SUSPECTED_SPOOF"
                confidence = primary_score
            else:
                secondary_passed = sum(1 for r in secondary_results if r['passed'])
                if secondary_passed >= 2:
                    is_real = True
                    attack_type = "NONE"
                    confidence = min(final_score, 0.95)
                else:
                    is_real = False
                    attack_type = "SUSPECTED_SPOOF"
                    confidence = final_score * 0.7
            
            result = {
                'is_real': is_real,
                'confidence': round(confidence, 4),
                'details': {
                    'results': results,
                    'primary_score': round(primary_score, 4),
                    'secondary_score': round(secondary_score, 4),
                    'final_score': round(final_score, 4),
                    'attack_type': attack_type,
                    'verdict': 'REAL' if is_real else 'SPOOF',
                    'indicators': self._get_spoof_indicators(results)
                }
            }
            
            logger.info(
                f"Anti-spoofing: {result['details']['verdict']} "
                f"(type: {attack_type}, primary: {primary_score:.3f}, "
                f"final: {final_score:.3f})"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Spoofing detection error: {str(e)}", exc_info=True)
            # FAIL-CLOSED: Reject on any error
            return {
                'is_real': False,
                'confidence': 0.0,
                'details': {
                    'error': str(e),
                    'verdict': 'ERROR'
                }
            }

    # =====================================================================
    # PRIMARY DETECTORS
    # =====================================================================

    def _detect_screen_moire_enhanced(self, image: np.ndarray) -> float:
        """
        Detect moiré interference patterns from screen photography.
        Screens create periodic patterns at specific frequencies.
        """
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (512, 512))
            
            # FFT analysis
            f = np.fft.fft2(gray.astype(np.float64))
            fshift = np.fft.fftshift(f)
            magnitude = np.log(np.abs(fshift) + 1)
            
            h, w = magnitude.shape
            cy, cx = h // 2, w // 2
            
            # Check multiple frequency bands
            bands = {
                'low_mid': (h//32, h//16),
                'mid': (h//16, h//8),
                'high_mid': (h//8, h//4),
            }
            
            band_scores = []
            for band_name, (r_inner, r_outer) in bands.items():
                y, x = np.ogrid[:h, :w]
                dist = np.sqrt((y - cy)**2 + (x - cx)**2)
                mask = (dist >= r_inner) & (dist <= r_outer)
                
                band_energy = magnitude[mask]
                if band_energy.size == 0:
                    band_scores.append(0.5)
                    continue
                
                mean_e = np.mean(band_energy)
                std_e = np.std(band_energy)
                max_e = np.max(band_energy)
                
                if std_e < 1e-6:
                    peakiness = 0.0
                else:
                    peakiness = (max_e - mean_e) / std_e
                
                # Convert peakiness to score
                if peakiness < 3.5:
                    score = 1.0
                elif peakiness < 5.0:
                    score = 0.8
                elif peakiness < 6.5:
                    score = 0.6
                elif peakiness < 8.0:
                    score = 0.4
                elif peakiness < 10.0:
                    score = 0.2
                else:
                    score = 0.1
                
                band_scores.append(score)
            
            final_score = np.mean(band_scores)
            
            # Additional directional check
            directional_score = self._check_directional_artifacts(gray)
            final_score = (final_score + directional_score) / 2
            
            return float(final_score)
            
        except Exception as e:
            logger.warning(f"Moiré detection error: {str(e)}")
            return 0.5

    def _detect_specular_glare_enhanced(self, image: np.ndarray) -> float:
        """
        Detect specular highlights from glossy screens.
        Glass surfaces produce sharp, bright reflection spots.
        """
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Detect very bright pixels (near-saturation)
            bright_mask = (gray > 235).astype(np.uint8) * 255
            
            # Find connected bright regions
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
                bright_mask, connectivity=8
            )
            
            total_pixels = gray.size
            hard_glare_pixels = 0
            hard_glare_blobs = 0
            
            for i in range(1, num_labels):  # skip background
                area = stats[i, cv2.CC_STAT_AREA]
                # Small, isolated bright blobs = glare spots
                if 3 <= area <= (total_pixels * 0.01):
                    hard_glare_blobs += 1
                    hard_glare_pixels += area
            
            glare_ratio = hard_glare_pixels / total_pixels
            
            # Score based on glare characteristics
            if hard_glare_blobs <= 2 and glare_ratio < 0.003:
                return 1.0
            elif hard_glare_blobs <= 4 and glare_ratio < 0.008:
                return 0.8
            elif hard_glare_blobs <= 7 and glare_ratio < 0.015:
                return 0.55
            elif hard_glare_blobs <= 12 and glare_ratio < 0.03:
                return 0.35
            else:
                return 0.15
                
        except Exception as e:
            logger.warning(f"Glare detection error: {str(e)}")
            return 0.5

    def _detect_screen_borders(self, image: np.ndarray) -> float:
        """
        Detect screen borders/edges in the image.
        """
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Edge detection
            edges = cv2.Canny(gray, 50, 150)
            
            # Look for straight lines (screen edges)
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, 
                                    threshold=100, 
                                    minLineLength=100, 
                                    maxLineGap=10)
            
            if lines is None:
                return 1.0
            
            horizontal_lines = 0
            vertical_lines = 0
            
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
                
                if angle < 10 or angle > 170:
                    horizontal_lines += 1
                elif 80 < angle < 100:
                    vertical_lines += 1
            
            # Score based on detected lines
            if horizontal_lines >= 2 and vertical_lines >= 2:
                return 0.2
            elif horizontal_lines >= 1 and vertical_lines >= 1:
                return 0.4
            elif horizontal_lines >= 2 or vertical_lines >= 2:
                return 0.6
            else:
                return 1.0
                
        except Exception as e:
            logger.warning(f"Border detection error: {str(e)}")
            return 0.5

    def _detect_halftone_patterns(self, image: np.ndarray) -> float:
        """
        Detect halftone dot patterns from printed photos.
        Prints use CMYK dots creating specific frequency signatures.
        """
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (512, 512))
            
            # FFT analysis for diagonal patterns (halftone rosette)
            f = np.fft.fft2(gray.astype(np.float64))
            fshift = np.fft.fftshift(f)
            magnitude = np.log(np.abs(fshift) + 1)
            
            h, w = magnitude.shape
            cy, cx = h // 2, w // 2
            
            # Check energy distribution in diagonal vs anti-diagonal
            y, x = np.ogrid[:h, :w]
            dist = np.sqrt((y - cy)**2 + (x - cx)**2)
            angle = np.degrees(np.arctan2(y - cy, x - cx)) % 180
            
            # Mask for mid-frequency band
            r_inner, r_outer = h//10, h//2.5
            band_mask = (dist >= r_inner) & (dist <= r_outer)
            
            # Diagonal masks
            diag_mask = ((angle >= 30) & (angle <= 60)) | ((angle >= 120) & (angle <= 150))
            anti_diag_mask = ((angle >= 60) & (angle <= 120)) | ((angle >= 150) & (angle <= 180))
            
            diag_energy = magnitude[band_mask & diag_mask]
            anti_diag_energy = magnitude[band_mask & anti_diag_mask]
            
            if diag_energy.size == 0 or anti_diag_energy.size == 0:
                return 0.5
            
            diag_mean = np.mean(diag_energy)
            anti_diag_mean = np.mean(anti_diag_energy)
            
            if anti_diag_mean > 0:
                ratio = diag_mean / anti_diag_mean
            else:
                ratio = 1.0
            
            # Halftone creates diagonal concentration
            if 0.85 < ratio < 1.15:
                return 1.0
            elif 0.7 < ratio < 1.3:
                return 0.7
            elif 0.5 < ratio < 1.5:
                return 0.4
            else:
                return 0.1
                
        except Exception as e:
            logger.warning(f"Halftone detection error: {str(e)}")
            return 0.5

    def _detect_paper_texture(self, image: np.ndarray) -> float:
        """
        Detect paper texture using Local Binary Patterns.
        Paper has uniform, fine texture unlike skin.
        """
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Compute simple LBP-like features
            h, w = gray.shape
            
            # Calculate local texture uniformity
            kernel_size = 5
            local_mean = cv2.blur(gray.astype(np.float32), (kernel_size, kernel_size))
            local_var = cv2.blur((gray.astype(np.float32) - local_mean)**2, (kernel_size, kernel_size))
            
            # Paper has uniform texture (low variance in local variance)
            var_of_var = np.std(local_var)
            mean_var = np.mean(local_var)
            
            # Calculate uniformity score
            if mean_var > 0:
                uniformity = var_of_var / mean_var
            else:
                uniformity = 0
            
            # Paper shows high uniformity (low var_of_var relative to mean_var)
            if uniformity < 0.5 and mean_var > 5:
                return 0.2  # Likely paper
            elif uniformity < 0.8 and mean_var > 3:
                return 0.4
            elif uniformity < 1.2:
                return 0.6
            else:
                return 0.9  # Likely real skin
                
        except Exception as e:
            logger.warning(f"Paper texture detection error: {str(e)}")
            return 0.5

    def _check_directional_artifacts(self, image: np.ndarray) -> float:
        """Check for directional artifacts common in screen captures"""
        try:
            f = np.fft.fft2(image.astype(np.float64))
            fshift = np.fft.fftshift(f)
            magnitude = np.abs(fshift)
            
            h, w = magnitude.shape
            cy, cx = h // 2, w // 2
            
            horizontal_energy = np.sum(magnitude[cy-5:cy+5, :])
            vertical_energy = np.sum(magnitude[:, cx-5:cx+5])
            
            if min(horizontal_energy, vertical_energy) > 0:
                ratio = max(horizontal_energy, vertical_energy) / min(horizontal_energy, vertical_energy)
                
                if ratio < 1.3:
                    return 1.0
                elif ratio < 1.6:
                    return 0.7
                elif ratio < 2.0:
                    return 0.4
                else:
                    return 0.2
            return 0.5
            
        except Exception:
            return 0.5

    # =====================================================================
    # SECONDARY CHECKS
    # =====================================================================

    def _analyze_micro_texture(self, image: np.ndarray) -> float:
        """Analyze micro-texture for skin characteristics"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Multi-scale analysis
            scores = []
            for sigma in [1, 2, 4]:
                blurred = cv2.GaussianBlur(gray, (0, 0), sigma)
                detail = cv2.absdiff(gray, blurred)
                
                # Calculate local variance
                kernel_size = 15
                local_mean = cv2.blur(detail, (kernel_size, kernel_size))
                local_sq_mean = cv2.blur(detail**2, (kernel_size, kernel_size))
                local_var = local_sq_mean - local_mean**2
                
                var_mean = np.mean(local_var)
                var_std = np.std(local_var)
                
                if 10 < var_mean < 100 and var_std > 5:
                    scores.append(1.0)
                elif 5 < var_mean < 150 and var_std > 3:
                    scores.append(0.7)
                elif var_mean > 0 and var_std > 1:
                    scores.append(0.4)
                else:
                    scores.append(0.2)
            
            return float(np.mean(scores))
            
        except Exception as e:
            logger.warning(f"Texture analysis error: {str(e)}")
            return 0.5

    def _analyze_color_distribution(self, image: np.ndarray) -> float:
        """Analyze color distribution for natural variation"""
        try:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # Check saturation range (screens often have limited range)
            saturation = hsv[:, :, 1]
            sat_range = np.max(saturation) - np.min(saturation)
            sat_std = np.std(saturation)
            
            # Check value range
            value = hsv[:, :, 2]
            val_std = np.std(value)
            
            # Combined score
            scores = []
            
            if sat_range > 150 and sat_std > 30:
                scores.append(1.0)
            elif sat_range > 100 and sat_std > 20:
                scores.append(0.7)
            elif sat_range > 50 and sat_std > 10:
                scores.append(0.4)
            else:
                scores.append(0.2)
            
            if val_std > 40:
                scores.append(1.0)
            elif val_std > 25:
                scores.append(0.7)
            elif val_std > 15:
                scores.append(0.4)
            else:
                scores.append(0.2)
            
            return float(np.mean(scores))
            
        except Exception as e:
            logger.warning(f"Color analysis error: {str(e)}")
            return 0.5

    def _analyze_noise_pattern(self, image: np.ndarray) -> float:
        """Analyze noise pattern for natural variation"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            scores = []
            for kernel_size in [3, 5, 7]:
                blurred = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
                noise = cv2.absdiff(gray, blurred)
                
                noise_mean = np.mean(noise)
                noise_std = np.std(noise)
                
                if 3 < noise_std < 30 and noise_mean < 20:
                    scores.append(1.0)
                elif 2 < noise_std < 40:
                    scores.append(0.7)
                elif 1 < noise_std < 50:
                    scores.append(0.4)
                else:
                    scores.append(0.2)
            
            return float(np.mean(scores))
            
        except Exception as e:
            logger.warning(f"Noise analysis error: {str(e)}")
            return 0.5

    def _get_spoof_indicators(self, results: List[Dict]) -> List[str]:
        """Get human-readable spoof indicators"""
        indicators = []
        
        for r in results:
            if r['critical'] and r['score'] < 0.3:
                if r['method'] == 'screen_moire':
                    indicators.append("Moiré pattern detected (screen replay)")
                elif r['method'] == 'specular_glare':
                    indicators.append("Abnormal specular reflections (glass surface)")
                elif r['method'] == 'screen_borders':
                    indicators.append("Screen borders/edges visible")
                elif r['method'] == 'halftone_pattern':
                    indicators.append("Halftone printing pattern detected")
                elif r['method'] == 'paper_texture':
                    indicators.append("Paper texture characteristics detected")
        
        return indicators