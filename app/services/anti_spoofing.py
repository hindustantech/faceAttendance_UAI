# app/services/anti_spoofing.py (ENHANCED VERSION)

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional  # Added all needed types
from app.config import settings
from app.utils.logger import setup_logger

logger = setup_logger(__name__)


class AntiSpoofingService:
    """
    Enhanced Anti-Spoofing Service
    
    Multi-layer detection for presentation attacks:
    1. Screen/Phone Detection - Moiré patterns, pixel grid, refresh artifacts
    2. Print/Paper Detection - Halftone dots, paper texture, lack of depth
    3. Digital Manipulation - Compression artifacts, edge inconsistencies
    4. Liveness Detection - Micro-texture, blood flow simulation impossible on replays
    """

    def __init__(self):
        self.initialized = False
        
        # Primary detectors (attack-specific)
        self.screen_reject_threshold = 0.40    # Below = screen detected
        self.print_reject_threshold = 0.40     # Below = print detected
        self.digital_reject_threshold = 0.45   # Below = digital manipulation
        
        # Secondary corroborating signals
        self.min_secondary_threshold = 0.30
        
        # Weighting factors for final score
        self.primary_weight = 0.70   # 70% weight to attack-specific detectors
        self.secondary_weight = 0.30  # 30% weight to general quality checks

    async def initialize(self):
        self.initialized = True
        logger.info("Enhanced anti-spoofing service initialized")

    async def detect_spoofing(self, image_data: np.ndarray) -> Dict:
        """
        Multi-layer spoofing detection.
        
        Returns detailed analysis with per-check scores and final verdict.
        """
        try:
            # Ensure image is valid
            if image_data is None or image_data.size == 0:
                raise ValueError("Invalid image data")
            
            # Resize for consistent processing
            h, w = image_data.shape[:2]
            if h > 1000 or w > 1000:
                scale = 1000 / max(h, w)
                new_h, new_w = int(h * scale), int(w * scale)
                image_data = cv2.resize(image_data, (new_w, new_h))
            
            results = []
            
            # =================================================================
            # LAYER 1: SCREEN/PHONE DETECTION (PRIMARY)
            # =================================================================
            
            # 1a. Moiré pattern detection (sub-pixel grid interference)
            moire_score, moire_details = self._detect_screen_moire_enhanced(image_data)
            results.append({
                'method': 'screen_moire',
                'score': moire_score,
                'passed': moire_score >= self.screen_reject_threshold,
                'critical': True,
                'weight': 0.25,
                'details': moire_details
            })
            
            # 1b. Pixel grid detection (screen door effect)
            grid_score, grid_details = self._detect_pixel_grid(image_data)
            results.append({
                'method': 'pixel_grid',
                'score': grid_score,
                'passed': grid_score >= self.screen_reject_threshold,
                'critical': True,
                'weight': 0.20,
                'details': grid_details
            })
            
            # 1c. Specular glare (glass reflection)
            glare_score, glare_details = self._detect_specular_glare_enhanced(image_data)
            results.append({
                'method': 'specular_glare',
                'score': glare_score,
                'passed': glare_score >= self.screen_reject_threshold,
                'critical': True,
                'weight': 0.15,
                'details': glare_details
            })
            
            # 1d. Screen edge/border detection
            border_score, border_details = self._detect_screen_borders(image_data)
            results.append({
                'method': 'screen_borders',
                'score': border_score,
                'passed': border_score >= 0.35,
                'critical': True,
                'weight': 0.10,
                'details': border_details
            })
            
            # =================================================================
            # LAYER 2: PRINT/PAPER DETECTION (PRIMARY)
            # =================================================================
            
            # 2a. Halftone/print dot patterns
            halftone_score, halftone_details = self._detect_halftone_patterns(image_data)
            results.append({
                'method': 'halftone_pattern',
                'score': halftone_score,
                'passed': halftone_score >= self.print_reject_threshold,
                'critical': True,
                'weight': 0.15,
                'details': halftone_details
            })
            
            # 2b. Paper texture analysis
            paper_score, paper_details = self._detect_paper_texture(image_data)
            results.append({
                'method': 'paper_texture',
                'score': paper_score,
                'passed': paper_score >= self.print_reject_threshold,
                'critical': True,
                'weight': 0.15,
                'details': paper_details
            })
            
            # =================================================================
            # LAYER 3: LIVENESS/BIO-METRICS (SECONDARY)
            # =================================================================
            
            # 3a. Micro-texture analysis (skin vs screen/paper)
            texture_score = self._analyze_micro_texture(image_data)
            results.append({
                'method': 'micro_texture',
                'score': texture_score,
                'passed': texture_score >= self.min_secondary_threshold,
                'critical': False,
                'weight': 0.50  # Within secondary weight group
            })
            
            # 3b. Color distribution analysis
            color_score = self._analyze_color_enhanced(image_data)
            results.append({
                'method': 'color_distribution',
                'score': color_score,
                'passed': color_score >= self.min_secondary_threshold,
                'critical': False,
                'weight': 0.30
            })
            
            # 3c. Noise pattern analysis
            noise_score = self._analyze_noise_enhanced(image_data)
            results.append({
                'method': 'noise_pattern',
                'score': noise_score,
                'passed': noise_score >= self.min_secondary_threshold,
                'critical': False,
                'weight': 0.20
            })
            
            # Calculate weighted scores
            critical_results = [r for r in results if r['critical']]
            secondary_results = [r for r in results if not r['critical']]
            
            # Primary score (weighted average of critical checks)
            primary_scores = []
            primary_weights = []
            for r in critical_results:
                primary_scores.append(r['score'] * r['weight'])
                primary_weights.append(r['weight'])
            
            primary_score = sum(primary_scores) / sum(primary_weights) if primary_weights else 0.5
            
            # Secondary score
            secondary_scores = []
            secondary_weights = []
            for r in secondary_results:
                secondary_scores.append(r['score'] * r['weight'])
                secondary_weights.append(r['weight'])
            
            secondary_score = sum(secondary_scores) / sum(secondary_weights) if secondary_weights else 0.5
            
            # Final combined score
            final_score = (primary_score * self.primary_weight + 
                         secondary_score * self.secondary_weight)
            
            # Decision logic:
            # 1. If ANY screen-specific detector is very confident of spoof -> REJECT
            # 2. If ANY print-specific detector is very confident of spoof -> REJECT
            # 3. If overall primary score is below threshold -> REJECT
            # 4. Otherwise, use combined score
            
            screen_checks = [r for r in critical_results 
                           if r['method'] in ['screen_moire', 'pixel_grid', 'specular_glare', 'screen_borders']]
            print_checks = [r for r in critical_results 
                          if r['method'] in ['halftone_pattern', 'paper_texture']]
            
            screen_spoof_confirmed = any(r['score'] < 0.25 for r in screen_checks)
            print_spoof_confirmed = any(r['score'] < 0.25 for r in print_checks)
            
            if screen_spoof_confirmed:
                is_real = False
                attack_type = "SCREEN_REPLAY"
                confidence = 0.1
            elif print_spoof_confirmed:
                is_real = False
                attack_type = "PRINT_PHOTO"
                confidence = 0.1
            elif primary_score < self.screen_reject_threshold:
                is_real = False
                attack_type = "SUSPECTED_SPOOF"
                confidence = primary_score
            else:
                # Check if enough secondary checks pass
                secondary_passed = sum(1 for r in secondary_results if r['passed'])
                if secondary_passed >= 2:  # At least 2 of 3 secondary checks must pass
                    is_real = True
                    attack_type = "NONE"
                    confidence = final_score
                else:
                    is_real = False
                    attack_type = "SUSPECTED_SPOOF"
                    confidence = final_score * 0.7  # Penalize low secondary scores
            
            # Cap confidence at 0.95 max (even for real faces)
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
                    'indicators': self._get_spoof_indicators(results)
                }
            }
            
            logger.info(
                f"Anti-spoofing result: {result['details']['verdict']} "
                f"(type: {attack_type}, primary: {primary_score:.3f}, "
                f"final: {final_score:.3f})"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Spoofing detection error: {str(e)}", exc_info=True)
            # FAIL-CLOSED: Always reject on error rather than risking bypass
            return {
                'is_real': False,
                'confidence': 0.0,
                'details': {
                    'error': str(e),
                    'verdict': 'ERROR'
                }
            }

    def _detect_screen_moire_enhanced(self, image: np.ndarray) -> Tuple[float, Dict]:
        """
        Enhanced moiré pattern detection with multiple frequency bands.
        Screens create interference patterns at specific frequencies.
        """
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (512, 512))
            
            # Multi-scale FFT analysis
            scores = []
            details = {}
            
            for scale_name, scale_factor in [('fine', 1.0), ('medium', 0.5), ('coarse', 0.25)]:
                if scale_factor < 1.0:
                    scaled = cv2.resize(gray, None, fx=scale_factor, fy=scale_factor)
                else:
                    scaled = gray
                
                f = np.fft.fft2(scaled.astype(np.float64))
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
                    
                    # Convert peakiness to score (lower = more likely spoof)
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
                
                scale_score = np.mean(band_scores)
                scores.append(scale_score)
                details[scale_name] = {
                    'band_scores': band_scores,
                    'average': scale_score
                }
            
            final_score = np.mean(scores)
            
            # Additional check: asymmetry in frequency distribution
            # Screens often show directional artifacts
            directional_score = self._check_directional_artifacts(gray)
            final_score = (final_score + directional_score) / 2
            
            return final_score, details
            
        except Exception as e:
            logger.warning(f"Moiré detection error: {str(e)}")
            return 0.5, {'error': str(e)}

    def _detect_pixel_grid(self, image: np.ndarray) -> Tuple[float, Dict]:
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
            # Take a central crop
            crop = high_pass[h//4:3*h//4, w//4:3*w//4]
            
            # Normalize
            crop = crop.astype(np.float32) / 255.0
            
            # Compute 2D autocorrelation
            f = np.fft.fft2(crop)
            power_spectrum = np.abs(f) ** 2
            autocorr = np.fft.ifft2(power_spectrum).real
            autocorr = np.fft.fftshift(autocorr)
            
            # Look for periodic peaks
            h_ac, w_ac = autocorr.shape
            cy_ac, cx_ac = h_ac // 2, w_ac // 2
            
            # Check for peaks at regular intervals (indicating grid)
            peak_distances = []
            threshold = np.max(autocorr) * 0.3
            
            for angle in range(0, 180, 45):  # Check multiple angles
                angle_rad = np.radians(angle)
                for dist in range(5, min(h_ac, w_ac)//4, 2):
                    y = int(cy_ac + dist * np.sin(angle_rad))
                    x = int(cx_ac + dist * np.cos(angle_rad))
                    if 0 <= y < h_ac and 0 <= x < w_ac:
                        if autocorr[y, x] > threshold:
                            peak_distances.append(dist)
            
            # Screen grids produce regular peaks
            if len(peak_distances) > 5:
                # Check for regularity
                diffs = np.diff(sorted(peak_distances))
                if len(diffs) > 0:
                    regularity = 1.0 - (np.std(diffs) / (np.mean(diffs) + 1e-6))
                    regularity = max(0, min(1, regularity))
                else:
                    regularity = 0
                
                # High regularity = likely screen grid
                score = 1.0 - regularity
            else:
                score = 1.0  # No regular pattern = likely real
            
            return score, {'peaks_found': len(peak_distances)}
            
        except Exception as e:
            logger.warning(f"Pixel grid detection error: {str(e)}")
            return 0.5, {'error': str(e)}

    def _detect_screen_borders(self, image: np.ndarray) -> Tuple[float, Dict]:
        """
        Detect screen borders/bezels in the image.
        When someone holds a phone, the screen edges might be visible.
        """
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Edge detection
            edges = cv2.Canny(gray, 50, 150)
            
            # Look for straight horizontal/vertical lines (screen edges)
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, 
                                    threshold=100, 
                                    minLineLength=100, 
                                    maxLineGap=10)
            
            if lines is None:
                return 1.0, {'lines_detected': 0}
            
            horizontal_lines = []
            vertical_lines = []
            
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
                
                if angle < 10 or angle > 170:  # Horizontal
                    horizontal_lines.append(line[0])
                elif 80 < angle < 100:  # Vertical
                    vertical_lines.append(line[0])
            
            # Screen borders would create perpendicular lines
            if len(horizontal_lines) >= 2 and len(vertical_lines) >= 2:
                # Check if they form a rectangular boundary
                score = 0.2  # Strong indication of screen
            elif len(horizontal_lines) >= 1 and len(vertical_lines) >= 1:
                score = 0.4  # Possible screen
            elif len(horizontal_lines) >= 2 or len(vertical_lines) >= 2:
                score = 0.6  # Suspicious
            else:
                score = 1.0  # No screen borders detected
            
            return score, {
                'h_lines': len(horizontal_lines),
                'v_lines': len(vertical_lines)
            }
            
        except Exception as e:
            logger.warning(f"Screen border detection error: {str(e)}")
            return 0.5, {'error': str(e)}

    def _detect_halftone_patterns(self, image: np.ndarray) -> Tuple[float, Dict]:
        """
        Enhanced halftone detection for printed photos.
        Prints use CMYK dot patterns that create specific frequency signatures.
        """
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (512, 512))
            
            # Multi-angle analysis for rosette patterns
            scores = []
            
            for angle_offset in [0, 15, 30, 45]:
                # Rotate image to check different angles
                matrix = cv2.getRotationMatrix2D((256, 256), angle_offset, 1.0)
                rotated = cv2.warpAffine(gray, matrix, (512, 512))
                
                f = np.fft.fft2(rotated.astype(np.float64))
                fshift = np.fft.fftshift(f)
                magnitude = np.log(np.abs(fshift) + 1)
                
                h, w = magnitude.shape
                cy, cx = h // 2, w // 2
                
                # Check diagonal quadrants (halftone dots cluster diagonally)
                diagonal_energy = []
                anti_diagonal_energy = []
                
                for y in range(h):
                    for x in range(w):
                        if y < cy and x < cx:  # Top-left
                            diagonal_energy.append(magnitude[y, x])
                        elif y > cy and x > cx:  # Bottom-right
                            diagonal_energy.append(magnitude[y, x])
                        elif y < cy and x > cx:  # Top-right
                            anti_diagonal_energy.append(magnitude[y, x])
                        elif y > cy and x < cx:  # Bottom-left
                            anti_diagonal_energy.append(magnitude[y, x])
                
                diag_mean = np.mean(diagonal_energy)
                anti_diag_mean = np.mean(anti_diagonal_energy)
                
                # Strong asymmetry indicates halftone pattern
                if anti_diag_mean > 0:
                    ratio = diag_mean / anti_diag_mean
                else:
                    ratio = 1.0
                
                if 0.85 < ratio < 1.15:
                    scores.append(1.0)  # Balanced = real
                elif 0.7 < ratio < 1.3:
                    scores.append(0.7)
                elif 0.5 < ratio < 1.5:
                    scores.append(0.4)
                else:
                    scores.append(0.1)  # Highly asymmetric = halftone
            
            final_score = np.mean(scores)
            return final_score, {'angle_scores': scores}
            
        except Exception as e:
            logger.warning(f"Halftone detection error: {str(e)}")
            return 0.5, {'error': str(e)}

    def _detect_paper_texture(self, image: np.ndarray) -> Tuple[float, Dict]:
        """
        Detect paper texture characteristics.
        Paper has unique fiber patterns and surface roughness.
        """
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Use Local Binary Patterns for texture analysis
            lbp = self._compute_lbp(gray)
            
            # Paper has uniform texture with specific LBP histogram
            hist = cv2.calcHist([lbp], [0], None, [256], [0, 256])
            hist = hist.flatten() / hist.sum()
            
            # Calculate texture uniformity (paper is more uniform than skin)
            uniformity = np.sum(hist ** 2)
            
            # Calculate entropy
            entropy = -np.sum(hist * np.log2(hist + 1e-7))
            
            # Paper typically has higher uniformity and lower entropy
            if uniformity > 0.1 and entropy < 5.0:
                score = 0.2  # Likely paper
            elif uniformity > 0.08 and entropy < 5.5:
                score = 0.4
            elif uniformity > 0.06 and entropy < 6.0:
                score = 0.6
            else:
                score = 0.9  # Likely real skin
            
            return score, {
                'uniformity': float(uniformity),
                'entropy': float(entropy)
            }
            
        except Exception as e:
            logger.warning(f"Paper texture detection error: {str(e)}")
            return 0.5, {'error': str(e)}

    def _compute_lbp(self, image: np.ndarray) -> np.ndarray:
        """Compute Local Binary Patterns"""
        h, w = image.shape
        lbp = np.zeros((h-2, w-2), dtype=np.uint8)
        
        for i in range(1, h-1):
            for j in range(1, w-1):
                center = image[i, j]
                code = 0
                
                # Compare with 8 neighbors
                neighbors = [
                    (i-1, j-1), (i-1, j), (i-1, j+1),
                    (i, j+1), (i+1, j+1), (i+1, j),
                    (i+1, j-1), (i, j-1)
                ]
                
                for k, (ni, nj) in enumerate(neighbors):
                    if image[ni, nj] >= center:
                        code |= (1 << k)
                
                lbp[i-1, j-1] = code
        
        return lbp

    def _check_directional_artifacts(self, image: np.ndarray) -> float:
        """Check for directional artifacts common in screen captures"""
        try:
            f = np.fft.fft2(image.astype(np.float64))
            fshift = np.fft.fftshift(f)
            magnitude = np.abs(fshift)
            
            h, w = magnitude.shape
            cy, cx = h // 2, w // 2
            
            # Check energy distribution in horizontal vs vertical
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

    def _analyze_micro_texture(self, image: np.ndarray) -> float:
        """Analyze micro-texture for skin characteristics"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Multi-scale texture analysis
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
                
                # Real skin has moderate, natural variance
                if 10 < var_mean < 100 and var_std > 5:
                    scores.append(1.0)
                elif 5 < var_mean < 150 and var_std > 3:
                    scores.append(0.7)
                elif var_mean > 0 and var_std > 1:
                    scores.append(0.4)
                else:
                    scores.append(0.2)
            
            return np.mean(scores)
            
        except Exception as e:
            logger.warning(f"Micro-texture analysis error: {str(e)}")
            return 0.5

    def _analyze_color_enhanced(self, image: np.ndarray) -> float:
        """Enhanced color analysis for screen vs real"""
        try:
            # Convert to multiple color spaces for comprehensive analysis
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            
            scores = []
            
            # HSV Saturation - screens often have reduced saturation range
            sat = hsv[:, :, 1]
            sat_range = np.max(sat) - np.min(sat)
            sat_std = np.std(sat)
            
            if sat_range > 150 and sat_std > 30:
                scores.append(1.0)
            elif sat_range > 100 and sat_std > 20:
                scores.append(0.7)
            elif sat_range > 50 and sat_std > 10:
                scores.append(0.4)
            else:
                scores.append(0.2)
            
            # LAB color space - screens have limited gamut
            a_channel = lab[:, :, 1]
            b_channel = lab[:, :, 2]
            
            ab_range = np.max(a_channel) - np.min(a_channel) + np.max(b_channel) - np.min(b_channel)
            
            if ab_range > 200:
                scores.append(1.0)
            elif ab_range > 150:
                scores.append(0.7)
            elif ab_range > 100:
                scores.append(0.4)
            else:
                scores.append(0.2)
            
            return np.mean(scores)
            
        except Exception as e:
            logger.warning(f"Color analysis error: {str(e)}")
            return 0.5

    def _analyze_noise_enhanced(self, image: np.ndarray) -> float:
        """Enhanced noise analysis"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Multi-level noise extraction
            scores = []
            for kernel_size in [3, 5, 7]:
                blurred = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
                noise = cv2.absdiff(gray, blurred)
                
                noise_mean = np.mean(noise)
                noise_std = np.std(noise)
                
                # Real images have natural noise distribution
                if 3 < noise_std < 30 and noise_mean < 20:
                    scores.append(1.0)
                elif 2 < noise_std < 40:
                    scores.append(0.7)
                elif 1 < noise_std < 50:
                    scores.append(0.4)
                else:
                    scores.append(0.2)
            
            return np.mean(scores)
            
        except Exception as e:
            logger.warning(f"Noise analysis error: {str(e)}")
            return 0.5

    def _get_spoof_indicators(self, results: List[Dict]) -> List[str]:
        """Get human-readable indicators of why something was flagged as spoof"""
        indicators = []
        
        for r in results:
            if r['critical'] and r['score'] < 0.3:
                if 'screen_moire' in r['method']:
                    indicators.append("Strong moiré pattern detected (screen replay)")
                elif 'pixel_grid' in r['method']:
                    indicators.append("Screen pixel grid detected")
                elif 'specular_glare' in r['method']:
                    indicators.append("Abnormal specular reflections (glass surface)")
                elif 'screen_borders' in r['method']:
                    indicators.append("Screen borders/edges visible")
                elif 'halftone' in r['method']:
                    indicators.append("Halftone printing pattern detected")
                elif 'paper' in r['method']:
                    indicators.append("Paper texture characteristics detected")
        
        return indicators


# Keep the old method signatures but deprecate them
class AntiSpoofingServiceLegacy(AntiSpoofingService):
    """Legacy compatibility class"""
    
    def _detect_screen_moire(self, image: np.ndarray) -> float:
        score, _ = self._detect_screen_moire_enhanced(image)
        return score
    
    def _detect_specular_glare(self, image: np.ndarray) -> float:
        score, _ = self._detect_specular_glare_enhanced(image)
        return score
    
    def _detect_print_artifacts(self, image: np.ndarray) -> float:
        score, _ = self._detect_halftone_patterns(image)
        return score
    
    def _analyze_texture(self, image: np.ndarray) -> float:
        return self._analyze_micro_texture(image)
    
    def _analyze_color_distribution(self, image: np.ndarray) -> float:
        return self._analyze_color_enhanced(image)
    
    def _analyze_noise_pattern(self, image: np.ndarray) -> float:
        return self._analyze_noise_enhanced(image)