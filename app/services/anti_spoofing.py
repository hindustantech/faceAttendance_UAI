import cv2
import numpy as np
from typing import Dict, List, Optional
from app.config import settings
from app.utils.logger import setup_logger

logger = setup_logger(__name__)


class AntiSpoofingService:
    """
    Anti-Spoofing Service - Multi-Attack Detection
    
    Detects three types of attacks:
    1. Screen/Display attacks (digital screens)
    2. Printed/Paper attacks (photos printed on paper)
    3. Real faces (should pass all checks)
    
    Strategy: Three independent checks with majority voting.
    - Screen detection: moire + glare + color artifacts
    - Print detection: texture + edge + paper reflectance
    - Real face: passes all checks naturally
    """

    def __init__(self):
        logger.info(f"[AntiSpoofingService.__init__] Initializing AntiSpoofingService")
        self.initialized = False
        
        # General thresholds
        self.spoofing_threshold = 0.60
        self.min_checks_required = 2
        self.min_individual_threshold = 0.40
        
        # Screen attack detection thresholds (CALIBRATED for production overlap)
        self.screen_moire_threshold = 0.35      # Raised from 0.40 to accommodate real faces
        self.screen_glare_threshold = 0.35      # Raised from 0.40
        self.screen_color_threshold = 0.40      # Keep at 0.40
        
        # Print attack detection thresholds
        self.print_texture_threshold = 0.50     # Low texture = possibly printed
        self.print_edge_threshold = 0.50        # Unnatural edges = possibly printed
        self.print_reflectance_threshold = 0.50 # Paper reflectance characteristics
        
        # Real face validation
        self.real_face_min_score = 0.60
        self.low_confidence_threshold = 0.60
        
        logger.info(f"[AntiSpoofingService.__init__] Thresholds calibrated for production:")
        logger.info(f"  Screen moire: {self.screen_moire_threshold}")
        logger.info(f"  Screen glare: {self.screen_glare_threshold}")
        logger.info(f"  Screen color: {self.screen_color_threshold}")
        logger.info(f"  Print texture: {self.print_texture_threshold}")
        logger.info(f"  Print edge: {self.print_edge_threshold}")

    async def initialize(self):
        """Initialize anti-spoofing service"""
        logger.info("[AntiSpoofingService.initialize] Starting initialization")
        self.initialized = True
        logger.info("Anti-spoofing service initialized - MULTI-ATTACK DETECTION MODE")

    async def detect_spoofing(
        self,
        image_data: np.ndarray,
        motion_frames: Optional[List[np.ndarray]] = None
    ) -> Dict:
        """
        Detect if face is real or spoofed.
        
        Returns verdict: REAL, SCREEN_ATTACK, PRINT_ATTACK, or UNKNOWN_ATTACK
        """
        logger.info("[AntiSpoofingService.detect_spoofing] ========== STARTING DETECTION ==========")
        logger.info(f"[AntiSpoofingService.detect_spoofing] Image shape: {image_data.shape}, dtype: {image_data.dtype}")

        try:
            # Extract face region if possible for more accurate analysis
            face_roi = self._extract_face_region(image_data)
            analysis_image = face_roi if face_roi is not None else image_data
            
            if face_roi is not None:
                logger.info(f"[AntiSpoofingService.detect_spoofing] Face ROI extracted: {face_roi.shape}")
            else:
                logger.warning("[AntiSpoofingService.detect_spoofing] No face detected, analyzing full image")

            # ================================================================
            # THREE-PRONGED ATTACK DETECTION
            # ================================================================
            
            # ---- SCREEN ATTACK DETECTION ----
            screen_checks = []
            
            logger.info("[AntiSpoofingService.detect_spoofing] --- Screen Attack Checks ---")
            
            moire_score = self._detect_moire_pattern(analysis_image)
            moire_result = self._make_result('moire_pattern', moire_score, self.screen_moire_threshold)
            screen_checks.append(moire_result)
            
            glare_score = self._analyze_specular_highlights(analysis_image)
            glare_result = self._make_result('specular_glare', glare_score, self.screen_glare_threshold)
            screen_checks.append(glare_result)
            
            color_score = self._analyze_screen_color_artifacts(analysis_image)
            color_result = self._make_result('screen_color_artifacts', color_score, self.screen_color_threshold)
            screen_checks.append(color_result)
            
            # ---- PRINT ATTACK DETECTION ----
            print_checks = []
            
            logger.info("[AntiSpoofingService.detect_spoofing] --- Print Attack Checks ---")
            
            texture_score = self._analyze_texture_quality(analysis_image)
            texture_result = self._make_result('texture_quality', texture_score, self.print_texture_threshold)
            print_checks.append(texture_result)
            
            edge_score = self._analyze_edge_naturalness(analysis_image)
            edge_result = self._make_result('edge_naturalness', edge_score, self.print_edge_threshold)
            print_checks.append(edge_result)
            
            reflectance_score = self._analyze_paper_reflectance(image_data)
            reflectance_result = self._make_result('paper_reflectance', reflectance_score, self.print_reflectance_threshold)
            print_checks.append(reflectance_result)
            
            # ---- COMPUTE VERDICTS ----
            screen_passed = sum(1 for r in screen_checks if r['passed'])
            print_passed = sum(1 for r in print_checks if r['passed'])
            
            screen_score = sum(r['score'] for r in screen_checks) / len(screen_checks)
            print_score = sum(r['score'] for r in print_checks) / len(print_checks)
            
            # Determine attack type and verdict
            is_screen_attack = screen_passed < 2  # Majority says it's a screen
            is_print_attack = print_passed < 2    # Majority says it's printed
            
            if is_screen_attack:
                attack_type = "SCREEN_ATTACK"
                is_real = False
                confidence = 1.0 - screen_score
            elif is_print_attack:
                attack_type = "PRINT_ATTACK"
                is_real = False
                confidence = 1.0 - print_score
            else:
                attack_type = "NONE"
                is_real = True
                confidence = (screen_score + print_score) / 2
            
            # Log detailed results
            logger.info("[AntiSpoofingService.detect_spoofing] ========== RESULTS SUMMARY ==========")
            logger.info(f"  Screen checks passed: {screen_passed}/3 (Score: {screen_score:.4f})")
            for r in screen_checks:
                status = "✓ PASS" if r['passed'] else "✗ FAIL"
                logger.info(f"    {r['method']}: {r['score']:.4f} {status}")
            
            logger.info(f"  Print checks passed: {print_passed}/3 (Score: {print_score:.4f})")
            for r in print_checks:
                status = "✓ PASS" if r['passed'] else "✗ FAIL"
                logger.info(f"    {r['method']}: {r['score']:.4f} {status}")
            
            logger.info(f"  Attack type: {attack_type}")
            logger.info(f"  Verdict: {'REAL' if is_real else 'SPOOF'}")
            logger.info(f"  Confidence: {confidence:.4f}")
            
            result = {
                'is_real': is_real,
                'confidence': round(confidence, 4),
                'threshold': self.spoofing_threshold,
                'attack_type': attack_type,
                'details': {
                    'screen_checks': screen_checks,
                    'print_checks': print_checks,
                    'screen_passed': screen_passed,
                    'print_passed': print_passed,
                    'verdict': 'REAL' if is_real else 'SPOOF'
                }
            }
            
            logger.info(f"[AntiSpoofingService.detect_spoofing] ========== DETECTION COMPLETE ==========")
            return result
            
        except Exception as e:
            logger.error(f"[AntiSpoofingService.detect_spoofing] ❌ Detection FAILED: {str(e)}", exc_info=True)
            logger.critical("[AntiSpoofingService.detect_spoofing] FAILING SAFE - allowing real on error")
            return {
                'is_real': True,  # Fail safe - don't block real users on error
                'confidence': 0.5,
                'threshold': self.spoofing_threshold,
                'attack_type': 'UNKNOWN',
                'details': {
                    'error': str(e),
                    'verdict': 'PASSED_ON_ERROR'
                }
            }

    def _extract_face_region(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Extract face region for more accurate analysis."""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(100, 100))
            
            if len(faces) > 0:
                # Get the largest face
                largest_face = max(faces, key=lambda f: f[2] * f[3])
                x, y, w, h = largest_face
                
                # Add padding around face (20%)
                pad_x = int(w * 0.2)
                pad_y = int(h * 0.2)
                
                x1 = max(0, x - pad_x)
                y1 = max(0, y - pad_y)
                x2 = min(image.shape[1], x + w + pad_x)
                y2 = min(image.shape[0], y + h + pad_y)
                
                return image[y1:y2, x1:x2]
            return None
        except Exception as e:
            logger.warning(f"Face extraction failed: {str(e)}")
            return None

    def _make_result(self, method: str, score: float, threshold: Optional[float] = None) -> Dict:
        """Create standardized result dictionary."""
        t = threshold if threshold is not None else self.min_individual_threshold
        passed = score >= t
        result = {'method': method, 'score': score, 'passed': passed}
        logger.info(f"[AntiSpoofingService._make_result] {method}: {score:.4f} (threshold: {t:.4f}) - {'PASS' if passed else 'FAIL'}")
        return result

    # ------------------------------------------------------------------
    # SCREEN ATTACK DETECTION METHODS
    # ------------------------------------------------------------------

    def _detect_moire_pattern(self, image: np.ndarray) -> float:
        """
        Detect moire patterns indicative of screen displays.
        
        FIXED: Calibrated for production overlap between real faces and screens.
        Real faces: peak_ratio ~4.5-5.0, outlier_frac ~0.017-0.018
        Screens: peak_ratio ~5.0-8.0, outlier_frac ~0.017-0.025
        """
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            all_peak_ratios = []
            all_outlier_fractions = []
            
            scales = [(256, 256), (512, 512)]
            
            for scale_size in scales:
                gray_scaled = cv2.resize(gray, scale_size)
                hann_window = np.outer(np.hanning(scale_size[0]), np.hanning(scale_size[1]))
                gray_windowed = gray_scaled * hann_window
                
                f = np.fft.fft2(gray_windowed.astype(np.float32))
                fshift = np.fft.fftshift(f)
                magnitude = np.log1p(np.abs(fshift))
                
                h, w = magnitude.shape
                cy, cx = h // 2, w // 2
                
                # Focus on mid-frequencies where screen moire appears
                for r_min, r_max in [(10, 30), (15, 40)]:
                    y, x = np.ogrid[:h, :w]
                    dist_from_center = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
                    band_mask = (dist_from_center > r_min) & (dist_from_center <= r_max)
                    
                    band_energy = magnitude[band_mask]
                    if len(band_energy) > 0:
                        mean_energy = np.mean(band_energy)
                        std_energy = np.std(band_energy)
                        
                        if std_energy > 1e-6:
                            peak_ratio = (np.max(band_energy) - mean_energy) / std_energy
                            all_peak_ratios.append(peak_ratio)
                            
                            # Detect periodic outliers
                            for threshold_mult in [2.5, 3.0]:
                                outlier_threshold = mean_energy + threshold_mult * std_energy
                                outlier_fraction = np.sum(band_energy > outlier_threshold) / len(band_energy)
                                all_outlier_fractions.append(outlier_fraction)
            
            if not all_peak_ratios:
                return 0.5
            
            max_peak_ratio = max(all_peak_ratios)
            max_outlier_fraction = max(all_outlier_fractions)
            
            logger.info(
                f"[AntiSpoofingService._detect_moire_pattern] "
                f"peak_ratio={max_peak_ratio:.3f} outlier_frac={max_outlier_fraction:.5f}"
            )
            
            # CALIBRATED SCORING:
            # Production data shows real faces have peak_ratio up to 5.0 and outlier_frac up to 0.018
            # Screens typically show peak_ratio > 6.0 and outlier_frac > 0.020
            # We use a graduated scoring system that accommodates this overlap
            
            # Peak ratio scoring (higher = more likely screen)
            if max_peak_ratio < 4.0:
                peak_score = 1.0  # Definitely real
            elif max_peak_ratio < 5.5:
                peak_score = 0.7  # Likely real (real faces can reach 5.0)
            elif max_peak_ratio < 7.0:
                peak_score = 0.4  # Suspicious
            elif max_peak_ratio < 10.0:
                peak_score = 0.2  # Likely screen
            else:
                peak_score = 0.05  # Definitely screen
            
            # Outlier fraction scoring (higher = more likely screen)
            if max_outlier_fraction < 0.015:
                outlier_score = 1.0  # Real face range
            elif max_outlier_fraction < 0.020:
                outlier_score = 0.6  # Overlap zone (both real and screen)
            elif max_outlier_fraction < 0.025:
                outlier_score = 0.3  # Likely screen
            else:
                outlier_score = 0.1  # Definitely screen
            
            # Weighted combination (peak ratio is more reliable)
            final_score = 0.6 * peak_score + 0.4 * outlier_score
            
            return float(max(0.05, min(1.0, final_score)))
            
        except Exception as e:
            logger.warning(f"Moire detection failed: {str(e)}")
            return 0.5

    def _analyze_specular_highlights(self, image: np.ndarray) -> float:
        """
        Detect specular highlights characteristic of screens.
        
        FIXED: Less aggressive thresholds. Real faces in normal lighting
        shouldn't be penalized heavily for having some bright pixels.
        """
        try:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            v_channel = hsv[:, :, 2]
            
            # Check multiple brightness thresholds
            thresholds = [250, 240, 230]
            scores = []
            
            for thresh in thresholds:
                bright_mask = (v_channel > thresh).astype(np.uint8) * 255
                total_pixels = bright_mask.size
                bright_fraction = np.sum(bright_mask > 0) / total_pixels
                
                # Find connected bright regions
                num_labels, _, stats, _ = cv2.connectedComponentsWithStats(bright_mask, connectivity=8)
                
                large_blob_fraction = 0.0
                if num_labels > 1:
                    areas = stats[1:, cv2.CC_STAT_AREA]
                    if len(areas) > 0:
                        large_blob_fraction = areas.max() / total_pixels
                
                # CALIBRATED SCORING (less aggressive):
                if bright_fraction > 0.15 or large_blob_fraction > 0.08:
                    scores.append(0.15)  # Very bright - screen likely
                elif bright_fraction > 0.10 or large_blob_fraction > 0.05:
                    scores.append(0.35)  # Moderately bright - suspicious
                elif bright_fraction > 0.05 or large_blob_fraction > 0.02:
                    scores.append(0.55)  # Some bright spots - could be real
                elif bright_fraction > 0.02 or large_blob_fraction > 0.01:
                    scores.append(0.75)  # Minor reflections - likely real
                else:
                    scores.append(0.90)  # Very little glare - real
            
            return min(scores) if scores else 0.5
            
        except Exception as e:
            logger.warning(f"Glare analysis failed: {str(e)}")
            return 0.5

    def _analyze_screen_color_artifacts(self, image: np.ndarray) -> float:
        """
        Detect color artifacts specific to screens (color banding, clipping).
        """
        try:
            b, g, r = cv2.split(image)
            
            # Check for color channel clipping (common in screens)
            r_high_clip = np.sum(r > 240) / r.size
            g_high_clip = np.sum(g > 240) / g.size
            b_high_clip = np.sum(b > 240) / b.size
            high_clip_score = max(r_high_clip, g_high_clip, b_high_clip)
            
            # Check for histogram banding (screen characteristic)
            def count_banding_gaps(channel):
                hist = cv2.calcHist([channel], [0], None, [256], [0, 256])
                gaps = 0
                consecutive_zeros = 0
                for i in range(len(hist)):
                    if hist[i][0] == 0:
                        consecutive_zeros += 1
                    else:
                        if consecutive_zeros >= 3:
                            gaps += 1
                        consecutive_zeros = 0
                if consecutive_zeros >= 3:
                    gaps += 1
                return gaps
            
            total_gaps = count_banding_gaps(image[:, :, 2]) + \
                        count_banding_gaps(image[:, :, 1]) + \
                        count_banding_gaps(image[:, :, 0])
            
            # Check saturation (screens often have lower saturation)
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            saturation = hsv[:, :, 1]
            sat_std = np.std(saturation)
            
            screen_score = 0.0
            
            # High clip scoring
            if high_clip_score > 0.08:
                screen_score += 0.4
            elif high_clip_score > 0.05:
                screen_score += 0.25
            elif high_clip_score > 0.02:
                screen_score += 0.15
            
            # Banding scoring
            if total_gaps > 12:
                screen_score += 0.4
            elif total_gaps > 8:
                screen_score += 0.25
            elif total_gaps > 4:
                screen_score += 0.15
            
            # Saturation scoring
            if sat_std < 12:
                screen_score += 0.2
            elif sat_std < 18:
                screen_score += 0.1
            
            return 1.0 - min(screen_score, 1.0)
            
        except Exception as e:
            logger.warning(f"Screen color analysis failed: {str(e)}")
            return 0.5

    # ------------------------------------------------------------------
    # PRINT ATTACK DETECTION METHODS
    # ------------------------------------------------------------------

    def _analyze_texture_quality(self, image: np.ndarray) -> float:
        """
        Analyze texture quality for print detection.
        Printed photos lack natural skin texture micro-details.
        """
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Laplacian variance (measures texture detail)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # Local Binary Pattern for micro-texture
            gray_resized = cv2.resize(gray, (200, 200)).astype(np.int32)
            center = gray_resized[1:-1, 1:-1]
            offsets = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]
            
            lbp = np.zeros_like(center, dtype=np.uint8)
            for i, (dy, dx) in enumerate(offsets):
                neighbor = gray_resized[1+dy:1+dy+center.shape[0], 1+dx:1+dx+center.shape[1]]
                lbp |= ((neighbor >= center).astype(np.uint8) << i)
            
            hist, _ = np.histogram(lbp, bins=256, range=(0, 256))
            hist = hist.astype(np.float64)
            hist_sum = hist.sum()
            
            if hist_sum == 0:
                return 0.5
            
            prob = hist / hist_sum
            prob = prob[prob > 0]
            entropy = -np.sum(prob * np.log2(prob))
            normalized_entropy = entropy / 8.0
            
            # Combine Laplacian and entropy for texture score
            if laplacian_var > 300 and normalized_entropy > 0.75:
                return 0.95  # Rich texture - real skin
            elif laplacian_var > 150 and normalized_entropy > 0.65:
                return 0.80  # Good texture - likely real
            elif laplacian_var > 80 and normalized_entropy > 0.55:
                return 0.60  # Moderate texture - suspicious
            elif laplacian_var > 40 and normalized_entropy > 0.45:
                return 0.40  # Poor texture - likely printed
            elif laplacian_var > 20:
                return 0.25  # Very poor texture
            else:
                return 0.10  # No texture - definitely printed/artificial
                
        except Exception as e:
            logger.warning(f"Texture analysis failed: {str(e)}")
            return 0.5

    def _analyze_edge_naturalness(self, image: np.ndarray) -> float:
        """
        Analyze edge characteristics for print detection.
        Printed photos often have uniform, unnatural edge distributions.
        """
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Multi-scale edge detection
            scores = []
            for low_thresh, high_thresh in [(30, 90), (50, 150), (80, 200)]:
                edges = cv2.Canny(gray, low_thresh, high_thresh)
                edge_density = np.sum(edges > 0) / edges.size
                
                # Printed photos often have very uniform edge density
                # or very low edge density
                if 0.08 < edge_density < 0.20:
                    scores.append(0.90)  # Natural edge distribution
                elif 0.05 < edge_density < 0.25:
                    scores.append(0.70)  # Acceptable
                elif 0.02 < edge_density < 0.30:
                    scores.append(0.45)  # Suspicious
                else:
                    scores.append(0.20)  # Unnatural
            
            # Edge direction uniformity (printed photos often have aligned edges)
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            magnitude = np.sqrt(sobelx**2 + sobely**2)
            direction = np.arctan2(sobely, sobelx)
            
            # Check if edge directions are too uniform
            mag_threshold = np.percentile(magnitude, 80)
            strong_edges = magnitude > mag_threshold
            if np.any(strong_edges):
                strong_dirs = direction[strong_edges]
                dir_hist, _ = np.histogram(strong_dirs, bins=36, range=(-np.pi, np.pi))
                dir_uniformity = np.std(dir_hist) / (np.mean(dir_hist) + 1e-6)
                
                if dir_uniformity > 0.8:
                    edge_score = 0.20  # Too uniform - printed
                elif dir_uniformity > 0.5:
                    edge_score = 0.50  # Somewhat uniform
                else:
                    edge_score = 0.85  # Natural variation
            else:
                edge_score = 0.60
            
            return (np.mean(scores) * 0.6 + edge_score * 0.4)
            
        except Exception as e:
            logger.warning(f"Edge analysis failed: {str(e)}")
            return 0.5

    def _analyze_paper_reflectance(self, image: np.ndarray) -> float:
        """
        Detect paper-specific reflectance characteristics.
        Paper has different light scattering properties than skin.
        """
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Paper tends to have more uniform reflectance
            # Calculate local variance in reflectance
            kernel_size = 15
            local_mean = cv2.blur(gray, (kernel_size, kernel_size))
            local_var = cv2.blur(gray.astype(np.float32)**2, (kernel_size, kernel_size)) - local_mean.astype(np.float32)**2
            
            # Skin has natural variation, paper is more uniform
            var_mean = np.mean(local_var)
            var_std = np.std(local_var)
            
            # Check for specular reflection patterns (paper reflects differently)
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            v_channel = hsv[:, :, 2]
            s_channel = hsv[:, :, 1]
            
            # Paper often has low saturation but high value uniformity
            sat_uniformity = 1.0 - (np.std(s_channel) / 128.0)
            val_uniformity = 1.0 - (np.std(v_channel) / 128.0)
            
            # Combined score
            if var_mean > 500 and sat_uniformity < 0.5:
                return 0.90  # Natural skin reflectance
            elif var_mean > 200 and sat_uniformity < 0.7:
                return 0.65  # Likely real
            elif var_mean > 100:
                return 0.40  # Suspicious - too uniform
            elif sat_uniformity > 0.8:
                return 0.20  # Very uniform - likely paper
            else:
                return 0.30  # Paper characteristics detected
                
        except Exception as e:
            logger.warning(f"Paper reflectance analysis failed: {str(e)}")
            return 0.5

    # ------------------------------------------------------------------
    # DEPRECATED METHODS (kept for compatibility)
    # ------------------------------------------------------------------

    def _analyze_texture(self, image: np.ndarray) -> float:
        """Deprecated: Use _analyze_texture_quality instead."""
        return self._analyze_texture_quality(image)

    def _analyze_color_distribution(self, image: np.ndarray) -> float:
        """Deprecated: Use _analyze_screen_color_artifacts instead."""
        return self._analyze_screen_color_artifacts(image)

    def _analyze_edges(self, image: np.ndarray) -> float:
        """Deprecated: Use _analyze_edge_naturalness instead."""
        return self._analyze_edge_naturalness(image)

    def _analyze_noise_pattern(self, image: np.ndarray) -> float:
        """Analyze noise pattern for additional validation."""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            noise = cv2.absdiff(gray, blurred)
            noise_std = np.std(noise)
            
            if 5 < noise_std < 40:
                return 1.0
            elif 3 < noise_std < 50:
                return 0.85
            elif 2 < noise_std < 60:
                return 0.70
            elif 1 < noise_std < 70:
                return 0.55
            else:
                return 0.40
        except Exception as e:
            logger.warning(f"Noise analysis failed: {str(e)}")
            return 0.5

    def _analyze_texture_lbp(self, image: np.ndarray) -> float:
        """LBP texture analysis (kept for compatibility)."""
        return self._analyze_texture_quality(image)

    def _analyze_motion_liveness(self, frames: List[np.ndarray]) -> float:
        """Motion liveness check."""
        try:
            grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
            
            flow_means = []
            flow_stds = []
            for i in range(len(grays) - 1):
                flow = cv2.calcOpticalFlowFarneback(
                    grays[i], grays[i + 1], None,
                    pyr_scale=0.5, levels=3, winsize=15,
                    iterations=3, poly_n=5, poly_sigma=1.2, flags=0
                )
                magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
                flow_means.append(np.mean(magnitude))
                flow_stds.append(np.std(magnitude))
            
            avg_motion = float(np.mean(flow_means))
            avg_variation = float(np.mean(flow_stds))
            
            if avg_motion < 0.05:
                return 0.15
            else:
                rigidity_ratio = avg_variation / max(avg_motion, 1e-6)
                if rigidity_ratio < 0.15:
                    return 0.30
                elif rigidity_ratio < 0.30:
                    return 0.55
                elif rigidity_ratio < 0.50:
                    return 0.75
                else:
                    return 0.90
        except Exception as e:
            logger.warning(f"Motion analysis failed: {str(e)}")
            return 0.5