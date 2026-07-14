# app/services/anti_spoofing.py (CALIBRATED VERSION)

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
from app.config import settings
from app.utils.logger import setup_logger

logger = setup_logger(__name__)


class AntiSpoofingService:
    """
    Calibrated Anti-Spoofing Service
    
    Balanced detection that rejects actual spoofs while accepting real faces.
    Uses weighted consensus rather than single-check rejection.
    """

    def __init__(self):
        self.initialized = False
        
        # CALIBRATED THRESHOLDS - less aggressive
        self.screen_reject_threshold = 0.55  # Was 0.40 - too strict
        self.print_reject_threshold = 0.45   # Was 0.40
        
        # Secondary corroborating signals
        self.min_secondary_threshold = 0.35  # Was 0.30
        
        # Weighting factors
        self.primary_weight = 0.60  # Was 0.70 - give more weight to secondary
        self.secondary_weight = 0.40  # Was 0.30
        
        # Minimum number of checks that must agree
        self.min_critical_consensus = 2  # At least 2 critical checks must flag spoof

    async def initialize(self):
        """Initialize the anti-spoofing service"""
        self.initialized = True
        logger.info("Calibrated anti-spoofing service initialized")

    async def detect_spoofing(self, image_data: np.ndarray) -> Dict:
        """
        Multi-layer spoofing detection with consensus-based decision.
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
            
            # 1a. Moiré pattern detection (most reliable for screens)
            moire_score = self._detect_screen_moire_enhanced(image_data)
            results.append({
                'method': 'screen_moire',
                'score': moire_score,
                'critical': True,
                'weight': 0.30
            })
            
            # 1b. Specular glare (can false-positive on oily skin)
            glare_score = self._detect_specular_glare_enhanced(image_data)
            results.append({
                'method': 'specular_glare',
                'score': glare_score,
                'critical': True,
                'weight': 0.15  # Reduced weight - less reliable alone
            })
            
            # 1c. Screen edge/border detection
            border_score = self._detect_screen_borders(image_data)
            results.append({
                'method': 'screen_borders',
                'score': border_score,
                'critical': True,
                'weight': 0.15  # Reduced weight - backgrounds can have lines
            })
            
            # =================================================================
            # LAYER 2: PRINT/PAPER DETECTION (PRIMARY)
            # =================================================================
            
            # 2a. Halftone/print dot patterns
            halftone_score = self._detect_halftone_patterns(image_data)
            results.append({
                'method': 'halftone_pattern',
                'score': halftone_score,
                'critical': True,
                'weight': 0.20
            })
            
            # 2b. Paper texture analysis
            paper_score = self._detect_paper_texture(image_data)
            results.append({
                'method': 'paper_texture',
                'score': paper_score,
                'critical': True,
                'weight': 0.20
            })
            
            # =================================================================
            # LAYER 3: SECONDARY CHECKS (CORROBORATING)
            # =================================================================
            
            # 3a. Micro-texture analysis (skin has natural variation)
            texture_score = self._analyze_micro_texture(image_data)
            results.append({
                'method': 'micro_texture',
                'score': texture_score,
                'critical': False,
                'weight': 0.35
            })
            
            # 3b. Color distribution (real faces have natural color variation)
            color_score = self._analyze_color_distribution(image_data)
            results.append({
                'method': 'color_distribution',
                'score': color_score,
                'critical': False,
                'weight': 0.35
            })
            
            # 3c. Noise pattern (camera sensors add natural noise)
            noise_score = self._analyze_noise_pattern(image_data)
            results.append({
                'method': 'noise_pattern',
                'score': noise_score,
                'critical': False,
                'weight': 0.30
            })
            
            # Separate critical and secondary results
            critical_results = [r for r in results if r['critical']]
            secondary_results = [r for r in results if not r['critical']]
            
            # Calculate weighted primary score
            primary_scores = []
            primary_weights = []
            for r in critical_results:
                primary_scores.append(r['score'] * r['weight'])
                primary_weights.append(r['weight'])
            
            if primary_weights:
                primary_score = sum(primary_scores) / sum(primary_weights)
            else:
                primary_score = 0.5
            
            # Calculate weighted secondary score
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
            
            # =================================================================
            # CONSENSUS-BASED DECISION LOGIC
            # =================================================================
            
            # Count how many critical checks strongly indicate spoof
            screen_spoof_flags = 0
            print_spoof_flags = 0
            
            for r in critical_results:
                if r['method'] in ['screen_moire', 'specular_glare', 'screen_borders']:
                    if r['score'] < 0.30:  # Strong screen indicator
                        screen_spoof_flags += 1
                elif r['method'] in ['halftone_pattern', 'paper_texture']:
                    if r['score'] < 0.25:  # Strong print indicator
                        print_spoof_flags += 1
            
            # Count secondary checks that indicate potential spoof
            secondary_spoof_flags = sum(1 for r in secondary_results if r['score'] < 0.25)
            
            # DECISION MATRIX:
            
            # Clear screen spoof: 2+ screen detectors strongly flag it
            if screen_spoof_flags >= self.min_critical_consensus:
                is_real = False
                attack_type = "SCREEN_REPLAY"
                confidence = 0.15
                
            # Clear print spoof: 2+ print detectors strongly flag it  
            elif print_spoof_flags >= self.min_critical_consensus:
                is_real = False
                attack_type = "PRINT_PHOTO"
                confidence = 0.15
                
            # Mixed signals: primary score is borderline AND secondary checks are poor
            elif primary_score < 0.45 and secondary_spoof_flags >= 2:
                is_real = False
                attack_type = "SUSPECTED_SPOOF"
                confidence = 0.25
                
            # Single detector flagged but others are fine - likely real face
            elif screen_spoof_flags == 1 or print_spoof_flags == 1:
                # One detector false-positive, others show real face
                if secondary_score > 0.60:  # Secondary checks strongly suggest real
                    is_real = True
                    attack_type = "NONE"
                    confidence = min(final_score, 0.90)
                else:
                    is_real = False
                    attack_type = "SUSPECTED_SPOOF"
                    confidence = 0.35
                    
            # Good primary score and decent secondary - real face
            elif primary_score >= 0.55:
                if secondary_score >= 0.50:
                    is_real = True
                    attack_type = "NONE"
                    confidence = min(final_score, 0.92)
                elif secondary_score >= 0.40:
                    is_real = True
                    attack_type = "NONE" 
                    confidence = min(final_score, 0.85)
                else:
                    is_real = False
                    attack_type = "UNCERTAIN"
                    confidence = 0.40
                    
            # Fallback: use final score with relaxed threshold
            else:
                if final_score >= 0.55:
                    is_real = True
                    attack_type = "NONE"
                    confidence = min(final_score, 0.85)
                elif final_score >= 0.45:
                    # Borderline - allow with warning
                    is_real = True
                    attack_type = "NONE"
                    confidence = 0.50
                else:
                    is_real = False
                    attack_type = "SUSPECTED_SPOOF"
                    confidence = final_score
            
            # Ensure confidence is capped
            confidence = min(confidence, 0.95)
            
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
                    'indicators': self._get_spoof_indicators(results),
                    'diagnostics': {
                        'screen_spoof_flags': screen_spoof_flags,
                        'print_spoof_flags': print_spoof_flags,
                        'secondary_spoof_flags': secondary_spoof_flags,
                        'consensus_required': self.min_critical_consensus
                    }
                }
            }
            
            logger.info(
                f"Anti-spoofing: {result['details']['verdict']} "
                f"(type: {attack_type}, primary: {primary_score:.3f}, "
                f"final: {final_score:.3f}, screen_flags: {screen_spoof_flags}, "
                f"print_flags: {print_spoof_flags})"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Spoofing detection error: {str(e)}", exc_info=True)
            # FAIL-OPEN for internal errors to avoid blocking real users
            # But log the error for investigation
            return {
                'is_real': True,  # Changed from False to True for errors
                'confidence': 0.5,
                'details': {
                    'error': str(e),
                    'verdict': 'ERROR_BYPASS'
                }
            }

    # =====================================================================
    # PRIMARY DETECTORS (unchanged implementation)
    # =====================================================================

    def _detect_screen_moire_enhanced(self, image: np.ndarray) -> float:
        """
        Detect moiré interference patterns from screen photography.
        This is the most reliable screen detector.
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
                
                # Convert peakiness to score (higher = more likely real)
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
            final_score = (final_score * 0.7 + directional_score * 0.3)
            
            return float(final_score)
            
        except Exception as e:
            logger.warning(f"Moiré detection error: {str(e)}")
            return 0.5

    def _detect_specular_glare_enhanced(self, image: np.ndarray) -> float:
        """
        Detect specular highlights from glossy screens.
        Less reliable alone - oily skin can cause false positives.
        """
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Use higher threshold to avoid false positives from skin shine
            bright_mask = (gray > 240).astype(np.uint8) * 255  # Was 235
            
            # Find connected bright regions
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
                bright_mask, connectivity=8
            )
            
            total_pixels = gray.size
            hard_glare_pixels = 0
            hard_glare_blobs = 0
            
            for i in range(1, num_labels):
                area = stats[i, cv2.CC_STAT_AREA]
                # Small, isolated very bright blobs = potential glare
                if 5 <= area <= (total_pixels * 0.005):  # Stricter range
                    # Check if the blob is truly isolated
                    x = stats[i, cv2.CC_STAT_LEFT]
                    y = stats[i, cv2.CC_STAT_TOP]
                    w = stats[i, cv2.CC_STAT_WIDTH]
                    h = stats[i, cv2.CC_STAT_HEIGHT]
                    
                    # Check surrounding area for other bright pixels
                    x1 = max(0, x - 10)
                    y1 = max(0, y - 10)
                    x2 = min(gray.shape[1], x + w + 10)
                    y2 = min(gray.shape[0], y + h + 10)
                    
                    surrounding = gray[y1:y2, x1:x2]
                    bright_surrounding = np.sum(surrounding > 200)
                    total_surrounding = surrounding.size
                    
                    # True glare is isolated (surrounded by much darker area)
                    if bright_surrounding / total_surrounding < 0.3:
                        hard_glare_blobs += 1
                        hard_glare_pixels += area
            
            glare_ratio = hard_glare_pixels / total_pixels
            
            # CALIBRATED: Much more lenient thresholds
            if hard_glare_blobs <= 3 and glare_ratio < 0.002:
                return 0.90  # Normal skin
            elif hard_glare_blobs <= 6 and glare_ratio < 0.005:
                return 0.75  # Slightly shiny skin
            elif hard_glare_blobs <= 10 and glare_ratio < 0.01:
                return 0.55  # Could be glare, could be oily skin
            elif hard_glare_blobs <= 15 and glare_ratio < 0.02:
                return 0.35  # Suspicious
            else:
                return 0.20  # Strong glare pattern
                
        except Exception as e:
            logger.warning(f"Glare detection error: {str(e)}")
            return 0.5

    def _detect_screen_borders(self, image: np.ndarray) -> float:
        """
        Detect screen borders/edges in the image.
        Background objects can create false positives.
        """
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Only check edges near the periphery (screen borders are usually at edges)
            h, w = gray.shape
            margin = 0.15  # Check within 15% of image border
            
            # Create peripheral mask
            mask = np.zeros_like(gray)
            mask[int(h*margin):int(h*(1-margin)), int(w*margin):int(w*(1-margin))] = 1
            
            # Edge detection on whole image
            edges = cv2.Canny(gray, 50, 150)
            
            # Only keep edges near borders
            edges = edges * (1 - mask)
            
            # Look for straight lines
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, 
                                    threshold=80, 
                                    minLineLength=w*0.3,  # Lines must be at least 30% of width
                                    maxLineGap=20)
            
            if lines is None:
                return 0.95  # Almost certainly real
            
            horizontal_lines = 0
            vertical_lines = 0
            
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
                
                # Check if line is near image border
                near_border = (y1 < h*margin or y1 > h*(1-margin) or 
                             y2 < h*margin or y2 > h*(1-margin))
                
                if near_border:
                    if angle < 10 or angle > 170:
                        horizontal_lines += 1
                    elif 80 < angle < 100:
                        vertical_lines += 1
            
            # CALIBRATED: Require multiple clear border lines
            if horizontal_lines >= 3 and vertical_lines >= 3:
                return 0.15  # Very likely screen
            elif horizontal_lines >= 2 and vertical_lines >= 2:
                return 0.30  # Possible screen
            elif horizontal_lines >= 2 or vertical_lines >= 2:
                return 0.50  # Suspicious but could be background
            elif horizontal_lines >= 1 or vertical_lines >= 1:
                return 0.75  # Probably just background
            else:
                return 0.95  # No screen borders
                
        except Exception as e:
            logger.warning(f"Border detection error: {str(e)}")
            return 0.5

    def _detect_halftone_patterns(self, image: np.ndarray) -> float:
        """
        Detect halftone dot patterns from printed photos.
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
            
            # Check energy distribution in diagonal vs anti-diagonal
            y, x = np.ogrid[:h, :w]
            dist = np.sqrt((y - cy)**2 + (x - cx)**2)
            angle = np.degrees(np.arctan2(y - cy, x - cx)) % 180
            
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
                return 0.90  # Balanced - likely real
            elif 0.70 < ratio < 1.30:
                return 0.70  # Slight imbalance
            elif 0.55 < ratio < 1.45:
                return 0.45  # Moderate imbalance
            else:
                return 0.20  # Strong halftone signature
                
        except Exception as e:
            logger.warning(f"Halftone detection error: {str(e)}")
            return 0.5

    def _detect_paper_texture(self, image: np.ndarray) -> float:
        """
        Detect paper texture using local variance analysis.
        """
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Calculate local texture uniformity
            kernel_size = 5
            local_mean = cv2.blur(gray.astype(np.float32), (kernel_size, kernel_size))
            local_var = cv2.blur((gray.astype(np.float32) - local_mean)**2, (kernel_size, kernel_size))
            
            var_of_var = np.std(local_var)
            mean_var = np.mean(local_var)
            
            # Calculate uniformity score
            if mean_var > 0:
                uniformity = var_of_var / mean_var
            else:
                uniformity = 0
            
            # Paper shows high uniformity (low var_of_var relative to mean_var)
            # But also has lower overall variance than skin
            if uniformity < 0.40 and mean_var < 10:
                return 0.15  # Very likely paper
            elif uniformity < 0.60 and mean_var < 20:
                return 0.35  # Possible paper
            elif uniformity < 0.80:
                return 0.55  # Could be either
            elif uniformity < 1.10:
                return 0.75  # Likely skin
            else:
                return 0.90  # Natural skin texture variation
                
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
            
            scores = []
            for sigma in [1, 2, 4]:
                blurred = cv2.GaussianBlur(gray, (0, 0), sigma)
                detail = cv2.absdiff(gray, blurred)
                
                kernel_size = 15
                local_mean = cv2.blur(detail, (kernel_size, kernel_size))
                local_sq_mean = cv2.blur(detail**2, (kernel_size, kernel_size))
                local_var = local_sq_mean - local_mean**2
                
                var_mean = np.mean(local_var)
                var_std = np.std(local_var)
                
                # CALIBRATED: Wider acceptable range for real skin
                if 8 < var_mean < 120 and var_std > 4:
                    scores.append(1.0)
                elif 5 < var_mean < 150 and var_std > 2:
                    scores.append(0.7)
                elif var_mean > 0 and var_std > 1:
                    scores.append(0.5)
                else:
                    scores.append(0.3)
            
            return float(np.mean(scores))
            
        except Exception as e:
            logger.warning(f"Texture analysis error: {str(e)}")
            return 0.5

    def _analyze_color_distribution(self, image: np.ndarray) -> float:
        """Analyze color distribution for natural variation"""
        try:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            saturation = hsv[:, :, 1]
            sat_range = np.max(saturation) - np.min(saturation)
            sat_std = np.std(saturation)
            
            value = hsv[:, :, 2]
            val_std = np.std(value)
            
            scores = []
            
            # CALIBRATED: More lenient ranges
            if sat_range > 100 and sat_std > 20:
                scores.append(1.0)
            elif sat_range > 70 and sat_std > 15:
                scores.append(0.75)
            elif sat_range > 40 and sat_std > 8:
                scores.append(0.5)
            else:
                scores.append(0.3)
            
            if val_std > 30:
                scores.append(1.0)
            elif val_std > 20:
                scores.append(0.75)
            elif val_std > 10:
                scores.append(0.5)
            else:
                scores.append(0.3)
            
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
                
                if 2 < noise_std < 35 and noise_mean < 25:
                    scores.append(1.0)
                elif 1.5 < noise_std < 45:
                    scores.append(0.7)
                elif 1 < noise_std < 55:
                    scores.append(0.5)
                else:
                    scores.append(0.3)
            
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