# app/services/anti_spoofing.py (FINAL AGGRESSIVE VERSION)

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
from app.config import settings
from app.utils.logger import setup_logger

logger = setup_logger(__name__)


class AntiSpoofingService:
    """
    Aggressive Anti-Spoofing Service
    
    Multi-layer detection with strong phone/paper rejection:
    1. Screen/Phone Detection - Moiré patterns, pixel grid, glare, borders
    2. Print/Paper Detection - Halftone dots, paper texture
    3. Liveness Detection - Micro-texture, natural skin characteristics
    """

    def __init__(self):
        self.initialized = False
        
        # Primary detector thresholds (MORE AGGRESSIVE)
        self.screen_reject_threshold = 0.55
        self.print_reject_threshold = 0.45
        
        # Secondary corroborating signals
        self.min_secondary_threshold = 0.35
        
        # Weighting factors
        self.primary_weight = 0.60
        self.secondary_weight = 0.40
        
        # Minimum consensus reduced to catch more spoofs
        self.min_critical_consensus = 1  # Only 1 strong flag needed now

    async def initialize(self):
        """Initialize the anti-spoofing service"""
        self.initialized = True
        logger.info("Aggressive anti-spoofing service initialized")

    async def detect_spoofing(self, image_data: np.ndarray) -> Dict:
        """
        Multi-layer spoofing detection with aggressive screen/print rejection.
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
            
            # 1a. Moiré pattern detection (MOST RELIABLE for screens)
            moire_score = self._detect_screen_moire_enhanced(image_data)
            results.append({
                'method': 'screen_moire',
                'score': moire_score,
                'critical': True,
                'weight': 0.35  # INCREASED - best screen detector
            })
            
            # 1b. Specular glare (glass reflection)
            glare_score = self._detect_specular_glare_enhanced(image_data)
            results.append({
                'method': 'specular_glare',
                'score': glare_score,
                'critical': True,
                'weight': 0.15
            })
            
            # 1c. Screen edge/border detection
            border_score = self._detect_screen_borders(image_data)
            results.append({
                'method': 'screen_borders',
                'score': border_score,
                'critical': True,
                'weight': 0.10
            })
            
            # 1d. Pixel grid detection (screen door effect)
            grid_score = self._detect_pixel_grid(image_data)
            results.append({
                'method': 'pixel_grid',
                'score': grid_score,
                'critical': True,
                'weight': 0.10
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
                'weight': 0.15
            })
            
            # 2b. Paper texture analysis
            paper_score = self._detect_paper_texture(image_data)
            results.append({
                'method': 'paper_texture',
                'score': paper_score,
                'critical': True,
                'weight': 0.15
            })
            
            # =================================================================
            # LAYER 3: SECONDARY CHECKS (CORROBORATING)
            # =================================================================
            
            # 3a. Micro-texture analysis
            texture_score = self._analyze_micro_texture(image_data)
            results.append({
                'method': 'micro_texture',
                'score': texture_score,
                'critical': False,
                'weight': 0.35
            })
            
            # 3b. Color distribution
            color_score = self._analyze_color_distribution(image_data)
            results.append({
                'method': 'color_distribution',
                'score': color_score,
                'critical': False,
                'weight': 0.35
            })
            
            # 3c. Noise pattern
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
            # AGGRESSIVE CONSENSUS-BASED DECISION LOGIC
            # =================================================================
            
            # Separate screen and print checks
            screen_checks = [r for r in critical_results 
                           if r['method'] in ['screen_moire', 'specular_glare', 'screen_borders', 'pixel_grid']]
            print_checks = [r for r in critical_results 
                          if r['method'] in ['halftone_pattern', 'paper_texture']]
            
            # Calculate combined screen score
            if screen_checks:
                screen_scores = [r['score'] * r['weight'] for r in screen_checks]
                screen_weights = [r['weight'] for r in screen_checks]
                combined_screen_score = sum(screen_scores) / sum(screen_weights)
            else:
                combined_screen_score = 1.0
            
            # Calculate combined print score
            if print_checks:
                print_scores = [r['score'] * r['weight'] for r in print_checks]
                print_weights = [r['weight'] for r in print_checks]
                combined_print_score = sum(print_scores) / sum(print_weights)
            else:
                combined_print_score = 1.0
            
            # Count strong flags (very suspicious)
            screen_spoof_flags = sum(1 for r in screen_checks if r['score'] < 0.30)
            print_spoof_flags = sum(1 for r in print_checks if r['score'] < 0.25)
            secondary_spoof_flags = sum(1 for r in secondary_results if r['score'] < 0.25)
            
            # Count moderate flags (somewhat suspicious)
            screen_moderate_flags = sum(1 for r in screen_checks if 0.30 <= r['score'] < 0.50)
            print_moderate_flags = sum(1 for r in print_checks if 0.25 <= r['score'] < 0.45)
            secondary_moderate_flags = sum(1 for r in secondary_results if 0.25 <= r['score'] < 0.40)
            
            # DECISION MATRIX:
            
            # 1. STRONG SCREEN DETECTION: 1+ strong flags OR very low combined score
            if screen_spoof_flags >= 1 or combined_screen_score < 0.35:
                is_real = False
                attack_type = "SCREEN_REPLAY"
                confidence = 0.10
                
            # 2. MODERATE SCREEN + CONFIRMATION: moderate flags with secondary issues
            elif screen_moderate_flags >= 1 and (secondary_spoof_flags >= 1 or secondary_moderate_flags >= 2):
                is_real = False
                attack_type = "SCREEN_REPLAY"
                confidence = 0.15
                
            # 3. COMBINED SCREEN SCORE LOW with secondary confirmation
            elif combined_screen_score < 0.45 and secondary_spoof_flags >= 1:
                is_real = False
                attack_type = "SCREEN_REPLAY"
                confidence = 0.20
                
            # 4. COMBINED SCREEN SCORE BORDERLINE with poor secondary
            elif combined_screen_score < 0.50 and secondary_score < 0.45:
                is_real = False
                attack_type = "SUSPECTED_SPOOF"
                confidence = 0.25
                
            # 5. STRONG PRINT DETECTION
            elif print_spoof_flags >= 1 or combined_print_score < 0.30:
                is_real = False
                attack_type = "PRINT_PHOTO"
                confidence = 0.10
                
            # 6. MODERATE PRINT + CONFIRMATION
            elif print_moderate_flags >= 1 and (secondary_spoof_flags >= 1 or secondary_moderate_flags >= 2):
                is_real = False
                attack_type = "PRINT_PHOTO"
                confidence = 0.15
                
            # 7. COMBINED PRINT SCORE LOW with secondary confirmation
            elif combined_print_score < 0.40 and secondary_spoof_flags >= 1:
                is_real = False
                attack_type = "PRINT_PHOTO"
                confidence = 0.20
                
            # 8. MULTIPLE MODERATE FLAGS ACROSS ALL CHECKS
            elif (screen_moderate_flags + print_moderate_flags + secondary_moderate_flags) >= 4:
                is_real = False
                attack_type = "SUSPECTED_SPOOF"
                confidence = 0.30
                
            # 9. GOOD SCORES: Both screen and print scores are good
            elif combined_screen_score >= 0.60 and combined_print_score >= 0.55:
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
                    
            # 10. GOOD SCREEN BUT BORDERLINE PRINT
            elif combined_screen_score >= 0.55 and combined_print_score >= 0.45:
                if secondary_score >= 0.55:
                    is_real = True
                    attack_type = "NONE"
                    confidence = min(final_score, 0.88)
                elif secondary_score >= 0.45:
                    is_real = True
                    attack_type = "NONE"
                    confidence = min(final_score, 0.80)
                else:
                    is_real = False
                    attack_type = "SUSPECTED_SPOOF"
                    confidence = 0.35
                    
            # 11. FALLBACK: Use final score
            else:
                if final_score >= 0.60:
                    is_real = True
                    attack_type = "NONE"
                    confidence = min(final_score, 0.85)
                elif final_score >= 0.52 and secondary_score >= 0.55:
                    is_real = True
                    attack_type = "NONE"
                    confidence = 0.65
                elif final_score >= 0.48 and secondary_score >= 0.60:
                    is_real = True
                    attack_type = "NONE"
                    confidence = 0.55
                else:
                    is_real = False
                    attack_type = "SUSPECTED_SPOOF"
                    confidence = 0.30
            
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
                    'combined_screen_score': round(combined_screen_score, 4),
                    'combined_print_score': round(combined_print_score, 4),
                    'attack_type': attack_type,
                    'verdict': 'REAL' if is_real else 'SPOOF',
                    'indicators': self._get_spoof_indicators(results),
                    'diagnostics': {
                        'screen_spoof_flags': screen_spoof_flags,
                        'screen_moderate_flags': screen_moderate_flags,
                        'print_spoof_flags': print_spoof_flags,
                        'print_moderate_flags': print_moderate_flags,
                        'secondary_spoof_flags': secondary_spoof_flags,
                        'secondary_moderate_flags': secondary_moderate_flags
                    }
                }
            }
            
            logger.info(
                f"Anti-spoofing: {result['details']['verdict']} "
                f"(type: {attack_type}, screen: {combined_screen_score:.3f}, "
                f"print: {combined_print_score:.3f}, final: {final_score:.3f}, "
                f"screen_flags: {screen_spoof_flags}/{screen_moderate_flags}, "
                f"print_flags: {print_spoof_flags}/{print_moderate_flags})"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Spoofing detection error: {str(e)}", exc_info=True)
            # FAIL-CLOSED for errors to maintain security
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
        MORE SENSITIVE thresholds to catch subtle screen patterns.
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
                
                # MORE SENSITIVE thresholds
                if peakiness < 2.5:       # Was 3.5
                    score = 1.0
                elif peakiness < 3.5:     # Was 5.0
                    score = 0.8
                elif peakiness < 4.5:     # Was 6.5
                    score = 0.6
                elif peakiness < 5.5:     # Was 8.0
                    score = 0.4
                elif peakiness < 7.0:     # Was 10.0
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

    def _detect_pixel_grid(self, image: np.ndarray) -> float:
        """
        Detect screen pixel grid (screen door effect).
        When photographing a screen, the camera captures the actual pixel structure.
        """
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Use high-pass filter to enhance fine details
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            high_pass = cv2.absdiff(gray, blurred)
            
            # Look for regular grid patterns using autocorrelation
            h, w = high_pass.shape
            crop = high_pass[h//4:3*h//4, w//4:3*w//4]
            crop = crop.astype(np.float32) / 255.0
            
            # Compute 2D autocorrelation
            f = np.fft.fft2(crop)
            power_spectrum = np.abs(f) ** 2
            autocorr = np.fft.ifft2(power_spectrum).real
            autocorr = np.fft.fftshift(autocorr)
            
            h_ac, w_ac = autocorr.shape
            cy_ac, cx_ac = h_ac // 2, w_ac // 2
            
            # Check for peaks at regular intervals
            peak_distances = []
            threshold = np.max(autocorr) * 0.3
            
            for angle in range(0, 180, 45):
                angle_rad = np.radians(angle)
                for dist in range(5, min(h_ac, w_ac)//4, 2):
                    y = int(cy_ac + dist * np.sin(angle_rad))
                    x = int(cx_ac + dist * np.cos(angle_rad))
                    if 0 <= y < h_ac and 0 <= x < w_ac:
                        if autocorr[y, x] > threshold:
                            peak_distances.append(dist)
            
            if len(peak_distances) > 5:
                diffs = np.diff(sorted(peak_distances))
                if len(diffs) > 0:
                    regularity = 1.0 - (np.std(diffs) / (np.mean(diffs) + 1e-6))
                    regularity = max(0, min(1, regularity))
                else:
                    regularity = 0
                score = 1.0 - regularity
            else:
                score = 1.0
            
            return float(score)
            
        except Exception as e:
            logger.warning(f"Pixel grid detection error: {str(e)}")
            return 0.5

    def _detect_specular_glare_enhanced(self, image: np.ndarray) -> float:
        """
        Detect specular highlights from glossy screens.
        """
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            bright_mask = (gray > 240).astype(np.uint8) * 255
            
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
                bright_mask, connectivity=8
            )
            
            total_pixels = gray.size
            hard_glare_pixels = 0
            hard_glare_blobs = 0
            
            for i in range(1, num_labels):
                area = stats[i, cv2.CC_STAT_AREA]
                if 5 <= area <= (total_pixels * 0.005):
                    x = stats[i, cv2.CC_STAT_LEFT]
                    y = stats[i, cv2.CC_STAT_TOP]
                    w = stats[i, cv2.CC_STAT_WIDTH]
                    h = stats[i, cv2.CC_STAT_HEIGHT]
                    
                    x1 = max(0, x - 10)
                    y1 = max(0, y - 10)
                    x2 = min(gray.shape[1], x + w + 10)
                    y2 = min(gray.shape[0], y + h + 10)
                    
                    surrounding = gray[y1:y2, x1:x2]
                    bright_surrounding = np.sum(surrounding > 200)
                    total_surrounding = surrounding.size
                    
                    if bright_surrounding / total_surrounding < 0.3:
                        hard_glare_blobs += 1
                        hard_glare_pixels += area
            
            glare_ratio = hard_glare_pixels / total_pixels
            
            if hard_glare_blobs <= 3 and glare_ratio < 0.002:
                return 0.90
            elif hard_glare_blobs <= 6 and glare_ratio < 0.005:
                return 0.75
            elif hard_glare_blobs <= 10 and glare_ratio < 0.01:
                return 0.55
            elif hard_glare_blobs <= 15 and glare_ratio < 0.02:
                return 0.35
            else:
                return 0.20
                
        except Exception as e:
            logger.warning(f"Glare detection error: {str(e)}")
            return 0.5

    def _detect_screen_borders(self, image: np.ndarray) -> float:
        """
        Detect screen borders/edges in the image.
        """
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            h, w = gray.shape
            margin = 0.15
            
            mask = np.ones_like(gray)
            mask[int(h*margin):int(h*(1-margin)), int(w*margin):int(w*(1-margin))] = 0
            
            edges = cv2.Canny(gray, 50, 150)
            edges = edges * mask
            
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, 
                                    threshold=80, 
                                    minLineLength=int(w*0.3), 
                                    maxLineGap=20)
            
            if lines is None:
                return 0.95
            
            horizontal_lines = 0
            vertical_lines = 0
            
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
                
                near_border = (y1 < h*margin or y1 > h*(1-margin) or 
                             y2 < h*margin or y2 > h*(1-margin))
                
                if near_border:
                    if angle < 10 or angle > 170:
                        horizontal_lines += 1
                    elif 80 < angle < 100:
                        vertical_lines += 1
            
            if horizontal_lines >= 3 and vertical_lines >= 3:
                return 0.15
            elif horizontal_lines >= 2 and vertical_lines >= 2:
                return 0.30
            elif horizontal_lines >= 2 or vertical_lines >= 2:
                return 0.50
            elif horizontal_lines >= 1 or vertical_lines >= 1:
                return 0.75
            else:
                return 0.95
                
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
            
            f = np.fft.fft2(gray.astype(np.float64))
            fshift = np.fft.fftshift(f)
            magnitude = np.log(np.abs(fshift) + 1)
            
            h, w = magnitude.shape
            cy, cx = h // 2, w // 2
            
            y, x = np.ogrid[:h, :w]
            dist = np.sqrt((y - cy)**2 + (x - cx)**2)
            angle = np.degrees(np.arctan2(y - cy, x - cx)) % 180
            
            r_inner, r_outer = h//10, h//2.5
            band_mask = (dist >= r_inner) & (dist <= r_outer)
            
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
            
            if 0.85 < ratio < 1.15:
                return 0.90
            elif 0.70 < ratio < 1.30:
                return 0.70
            elif 0.55 < ratio < 1.45:
                return 0.45
            else:
                return 0.20
                
        except Exception as e:
            logger.warning(f"Halftone detection error: {str(e)}")
            return 0.5

    def _detect_paper_texture(self, image: np.ndarray) -> float:
        """
        Detect paper texture using local variance analysis.
        """
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            kernel_size = 5
            local_mean = cv2.blur(gray.astype(np.float32), (kernel_size, kernel_size))
            local_var = cv2.blur((gray.astype(np.float32) - local_mean)**2, (kernel_size, kernel_size))
            
            var_of_var = np.std(local_var)
            mean_var = np.mean(local_var)
            
            if mean_var > 0:
                uniformity = var_of_var / mean_var
            else:
                uniformity = 0
            
            if uniformity < 0.40 and mean_var < 10:
                return 0.15
            elif uniformity < 0.60 and mean_var < 20:
                return 0.35
            elif uniformity < 0.80:
                return 0.55
            elif uniformity < 1.10:
                return 0.75
            else:
                return 0.90
                
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
            if r['critical'] and r['score'] < 0.30:
                if r['method'] == 'screen_moire':
                    indicators.append("Moiré pattern detected (screen replay)")
                elif r['method'] == 'pixel_grid':
                    indicators.append("Screen pixel grid detected")
                elif r['method'] == 'specular_glare':
                    indicators.append("Abnormal specular reflections (glass surface)")
                elif r['method'] == 'screen_borders':
                    indicators.append("Screen borders/edges visible")
                elif r['method'] == 'halftone_pattern':
                    indicators.append("Halftone printing pattern detected")
                elif r['method'] == 'paper_texture':
                    indicators.append("Paper texture characteristics detected")
            elif r['critical'] and r['score'] < 0.50:
                if r['method'] == 'screen_moire':
                    indicators.append("Possible moiré pattern (screen replay)")
                elif r['method'] == 'pixel_grid':
                    indicators.append("Possible screen pixel grid")
                elif r['method'] == 'specular_glare':
                    indicators.append("Possible specular reflections")
                elif r['method'] == 'screen_borders':
                    indicators.append("Possible screen borders")
                elif r['method'] == 'halftone_pattern':
                    indicators.append("Possible halftone pattern")
                elif r['method'] == 'paper_texture':
                    indicators.append("Possible paper texture")
        
        return indicators