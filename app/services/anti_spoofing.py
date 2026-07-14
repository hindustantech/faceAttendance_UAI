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

    VERDICT STRATEGY (fixed):
    Each category (screen, print) is judged two ways, and EITHER one
    firing is enough to flag that category as an attack:

      (a) Weighted average of that category's checks falls below the
          category threshold. This is the "overall impression" signal.

      (b) ANY single check in that category falls below a hard floor
          (`hard_fail_floor`). This exists because a strongly
          spoof-like signal (e.g. a very low moire score) should not
          be able to be "outvoted" or averaged away by two so-so
          passing checks. This is the piece the previous majority-vote
          logic (screen_passed < 2) was missing -- it only flagged an
          attack if a MAJORITY of checks failed, so one severe red flag
          plus two lukewarm passes was scored as REAL.

    Both category checks are independent and either one can veto a
    "real" verdict.
    """

    def __init__(self):
        logger.info(f"[AntiSpoofingService.__init__] Initializing AntiSpoofingService")
        self.initialized = False

        # Overall verdict threshold (weighted across all 6 checks)
        self.spoofing_threshold = 0.60

        # Per-category weighted-average thresholds
        self.screen_category_threshold = 0.55
        self.print_category_threshold = 0.55

        # Hard floor: ANY single check below this value is treated as a
        # strong, unambiguous spoof signal and vetoes the verdict, no
        # matter how well the other checks in that category score.
        self.hard_fail_floor = 0.20

        # Individual check thresholds (used for per-check pass/fail reporting
        # and for the "passed" flag shown in details/logs; the ACTUAL verdict
        # is decided by the weighted + hard-floor logic above, not by a
        # simple pass count).
        self.screen_moire_threshold = 0.35
        self.screen_glare_threshold = 0.35
        self.screen_color_threshold = 0.40
        self.print_texture_threshold = 0.50
        self.print_edge_threshold = 0.50
        self.print_reflectance_threshold = 0.50

        # Within-category weights (must each sum to 1.0)
        self.screen_weights = {
            'moire_pattern': 0.45,
            'specular_glare': 0.30,
            'screen_color_artifacts': 0.25,
        }
        self.print_weights = {
            'texture_quality': 0.45,
            'edge_naturalness': 0.30,
            'paper_reflectance': 0.25,
        }

        logger.info("[AntiSpoofingService.__init__] Thresholds:")
        logger.info(f"  Overall spoofing threshold: {self.spoofing_threshold}")
        logger.info(f"  Screen category threshold: {self.screen_category_threshold}")
        logger.info(f"  Print category threshold: {self.print_category_threshold}")
        logger.info(f"  Hard-fail floor (any single check): {self.hard_fail_floor}")

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
        (UNKNOWN_ATTACK covers the rare case where both categories fire
        simultaneously -- treated as screen since that's checked first).
        """
        logger.info("[AntiSpoofingService.detect_spoofing] ========== STARTING DETECTION ==========")
        logger.info(f"[AntiSpoofingService.detect_spoofing] Image shape: {image_data.shape}, dtype: {image_data.dtype}")

        try:
            face_roi = self._extract_face_region(image_data)
            analysis_image = face_roi if face_roi is not None else image_data

            if face_roi is not None:
                logger.info(f"[AntiSpoofingService.detect_spoofing] Face ROI extracted: {face_roi.shape}")
            else:
                logger.warning("[AntiSpoofingService.detect_spoofing] No face detected, analyzing full image")

            # ================================================================
            # RUN ALL SIX CHECKS
            # ================================================================
            logger.info("[AntiSpoofingService.detect_spoofing] --- Screen Attack Checks ---")
            moire_score = self._detect_moire_pattern(analysis_image)
            glare_score = self._analyze_specular_highlights(analysis_image)
            color_score = self._analyze_screen_color_artifacts(analysis_image)

            screen_checks = [
                self._make_result('moire_pattern', moire_score, self.screen_moire_threshold),
                self._make_result('specular_glare', glare_score, self.screen_glare_threshold),
                self._make_result('screen_color_artifacts', color_score, self.screen_color_threshold),
            ]

            logger.info("[AntiSpoofingService.detect_spoofing] --- Print Attack Checks ---")
            texture_score = self._analyze_texture_quality(analysis_image)
            edge_score = self._analyze_edge_naturalness(analysis_image)
            reflectance_score = self._analyze_paper_reflectance(image_data)

            print_checks = [
                self._make_result('texture_quality', texture_score, self.print_texture_threshold),
                self._make_result('edge_naturalness', edge_score, self.print_edge_threshold),
                self._make_result('paper_reflectance', reflectance_score, self.print_reflectance_threshold),
            ]

            # ================================================================
            # WEIGHTED CATEGORY SCORES
            # ================================================================
            screen_weighted = (
                moire_score * self.screen_weights['moire_pattern']
                + glare_score * self.screen_weights['specular_glare']
                + color_score * self.screen_weights['screen_color_artifacts']
            )
            print_weighted = (
                texture_score * self.print_weights['texture_quality']
                + edge_score * self.print_weights['edge_naturalness']
                + reflectance_score * self.print_weights['paper_reflectance']
            )

            screen_min_check = min(moire_score, glare_score, color_score)
            print_min_check = min(texture_score, edge_score, reflectance_score)

            # ================================================================
            # VERDICT LOGIC: weighted-average failure OR hard-floor veto
            # ================================================================
            screen_avg_fail = screen_weighted < self.screen_category_threshold
            screen_hard_fail = screen_min_check < self.hard_fail_floor
            is_screen_attack = screen_avg_fail or screen_hard_fail

            print_avg_fail = print_weighted < self.print_category_threshold
            print_hard_fail = print_min_check < self.hard_fail_floor
            is_print_attack = print_avg_fail or print_hard_fail

            overall_score = (screen_weighted + print_weighted) / 2.0

            if is_screen_attack and is_print_attack:
                attack_type = "UNKNOWN_ATTACK"
                is_real = False
                confidence = 1.0 - min(screen_weighted, print_weighted)
            elif is_screen_attack:
                attack_type = "SCREEN_ATTACK"
                is_real = False
                confidence = 1.0 - screen_weighted
            elif is_print_attack:
                attack_type = "PRINT_ATTACK"
                is_real = False
                confidence = 1.0 - print_weighted
            else:
                attack_type = "NONE"
                is_real = overall_score >= self.spoofing_threshold
                confidence = overall_score
                if not is_real:
                    # Weighted average alone didn't clear the bar even though
                    # no category or hard-floor veto fired individually.
                    attack_type = "UNKNOWN_ATTACK"

            # ---- Logging ----
            logger.info("[AntiSpoofingService.detect_spoofing] ========== RESULTS SUMMARY ==========")
            logger.info(f"  Screen weighted score: {screen_weighted:.4f} (threshold {self.screen_category_threshold}) "
                        f"| min check: {screen_min_check:.4f} (floor {self.hard_fail_floor}) "
                        f"| avg_fail={screen_avg_fail} hard_fail={screen_hard_fail}")
            for r in screen_checks:
                status = "✓ PASS" if r['passed'] else "✗ FAIL"
                logger.info(f"    {r['method']}: {r['score']:.4f} {status}")

            logger.info(f"  Print weighted score: {print_weighted:.4f} (threshold {self.print_category_threshold}) "
                        f"| min check: {print_min_check:.4f} (floor {self.hard_fail_floor}) "
                        f"| avg_fail={print_avg_fail} hard_fail={print_hard_fail}")
            for r in print_checks:
                status = "✓ PASS" if r['passed'] else "✗ FAIL"
                logger.info(f"    {r['method']}: {r['score']:.4f} {status}")

            logger.info(f"  Attack type: {attack_type}")
            logger.info(f"  Verdict: {'REAL' if is_real else 'SPOOF'}")
            logger.info(f"  Confidence: {confidence:.4f}")

            result = {
                'is_real': is_real,
                'confidence': round(float(confidence), 4),
                'threshold': self.spoofing_threshold,
                'attack_type': attack_type,
                'details': {
                    'screen_checks': screen_checks,
                    'print_checks': print_checks,
                    'screen_weighted_score': round(float(screen_weighted), 4),
                    'print_weighted_score': round(float(print_weighted), 4),
                    'screen_hard_fail': screen_hard_fail,
                    'print_hard_fail': print_hard_fail,
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
                largest_face = max(faces, key=lambda f: f[2] * f[3])
                x, y, w, h = largest_face

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

    def _make_result(self, method: str, score: float, threshold: float) -> Dict:
        """Create standardized result dictionary (used for logging/reporting only;
        the real verdict comes from the weighted + hard-floor logic above)."""
        passed = score >= threshold
        result = {'method': method, 'score': score, 'passed': passed}
        logger.info(f"[AntiSpoofingService._make_result] {method}: {score:.4f} (threshold: {threshold:.4f}) - {'PASS' if passed else 'FAIL'}")
        return result

    # ------------------------------------------------------------------
    # SCREEN ATTACK DETECTION METHODS
    # ------------------------------------------------------------------

    def _detect_moire_pattern(self, image: np.ndarray) -> float:
        """Detect moire patterns indicative of screen displays."""
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

            if max_peak_ratio < 4.0:
                peak_score = 1.0
            elif max_peak_ratio < 5.5:
                peak_score = 0.7
            elif max_peak_ratio < 7.0:
                peak_score = 0.4
            elif max_peak_ratio < 10.0:
                peak_score = 0.2
            else:
                peak_score = 0.05

            if max_outlier_fraction < 0.015:
                outlier_score = 1.0
            elif max_outlier_fraction < 0.020:
                outlier_score = 0.6
            elif max_outlier_fraction < 0.025:
                outlier_score = 0.3
            else:
                outlier_score = 0.1

            final_score = 0.6 * peak_score + 0.4 * outlier_score

            return float(max(0.05, min(1.0, final_score)))

        except Exception as e:
            logger.warning(f"Moire detection failed: {str(e)}")
            return 0.5

    def _analyze_specular_highlights(self, image: np.ndarray) -> float:
        """Detect specular highlights characteristic of screens."""
        try:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            v_channel = hsv[:, :, 2]

            thresholds = [250, 240, 230]
            scores = []

            for thresh in thresholds:
                bright_mask = (v_channel > thresh).astype(np.uint8) * 255
                total_pixels = bright_mask.size
                bright_fraction = np.sum(bright_mask > 0) / total_pixels

                num_labels, _, stats, _ = cv2.connectedComponentsWithStats(bright_mask, connectivity=8)

                large_blob_fraction = 0.0
                if num_labels > 1:
                    areas = stats[1:, cv2.CC_STAT_AREA]
                    if len(areas) > 0:
                        large_blob_fraction = areas.max() / total_pixels

                if bright_fraction > 0.15 or large_blob_fraction > 0.08:
                    scores.append(0.15)
                elif bright_fraction > 0.10 or large_blob_fraction > 0.05:
                    scores.append(0.35)
                elif bright_fraction > 0.05 or large_blob_fraction > 0.02:
                    scores.append(0.55)
                elif bright_fraction > 0.02 or large_blob_fraction > 0.01:
                    scores.append(0.75)
                else:
                    scores.append(0.90)

            return min(scores) if scores else 0.5

        except Exception as e:
            logger.warning(f"Glare analysis failed: {str(e)}")
            return 0.5

    def _analyze_screen_color_artifacts(self, image: np.ndarray) -> float:
        """Detect color artifacts specific to screens (color banding, clipping)."""
        try:
            b, g, r = cv2.split(image)

            r_high_clip = np.sum(r > 240) / r.size
            g_high_clip = np.sum(g > 240) / g.size
            b_high_clip = np.sum(b > 240) / b.size
            high_clip_score = max(r_high_clip, g_high_clip, b_high_clip)

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

            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            saturation = hsv[:, :, 1]
            sat_std = np.std(saturation)

            screen_score = 0.0

            if high_clip_score > 0.08:
                screen_score += 0.4
            elif high_clip_score > 0.05:
                screen_score += 0.25
            elif high_clip_score > 0.02:
                screen_score += 0.15

            if total_gaps > 12:
                screen_score += 0.4
            elif total_gaps > 8:
                screen_score += 0.25
            elif total_gaps > 4:
                screen_score += 0.15

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
        """Analyze texture quality for print detection."""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

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

            if laplacian_var > 300 and normalized_entropy > 0.75:
                return 0.95
            elif laplacian_var > 150 and normalized_entropy > 0.65:
                return 0.80
            elif laplacian_var > 80 and normalized_entropy > 0.55:
                return 0.60
            elif laplacian_var > 40 and normalized_entropy > 0.45:
                return 0.40
            elif laplacian_var > 20:
                return 0.25
            else:
                return 0.10

        except Exception as e:
            logger.warning(f"Texture analysis failed: {str(e)}")
            return 0.5

    def _analyze_edge_naturalness(self, image: np.ndarray) -> float:
        """Analyze edge characteristics for print detection."""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            scores = []
            for low_thresh, high_thresh in [(30, 90), (50, 150), (80, 200)]:
                edges = cv2.Canny(gray, low_thresh, high_thresh)
                edge_density = np.sum(edges > 0) / edges.size

                if 0.08 < edge_density < 0.20:
                    scores.append(0.90)
                elif 0.05 < edge_density < 0.25:
                    scores.append(0.70)
                elif 0.02 < edge_density < 0.30:
                    scores.append(0.45)
                else:
                    scores.append(0.20)

            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            magnitude = np.sqrt(sobelx**2 + sobely**2)
            direction = np.arctan2(sobely, sobelx)

            mag_threshold = np.percentile(magnitude, 80)
            strong_edges = magnitude > mag_threshold
            if np.any(strong_edges):
                strong_dirs = direction[strong_edges]
                dir_hist, _ = np.histogram(strong_dirs, bins=36, range=(-np.pi, np.pi))
                dir_uniformity = np.std(dir_hist) / (np.mean(dir_hist) + 1e-6)

                if dir_uniformity > 0.8:
                    edge_score = 0.20
                elif dir_uniformity > 0.5:
                    edge_score = 0.50
                else:
                    edge_score = 0.85
            else:
                edge_score = 0.60

            return (np.mean(scores) * 0.6 + edge_score * 0.4)

        except Exception as e:
            logger.warning(f"Edge analysis failed: {str(e)}")
            return 0.5

    def _analyze_paper_reflectance(self, image: np.ndarray) -> float:
        """Detect paper-specific reflectance characteristics."""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            kernel_size = 15
            local_mean = cv2.blur(gray, (kernel_size, kernel_size))
            local_var = cv2.blur(gray.astype(np.float32)**2, (kernel_size, kernel_size)) - local_mean.astype(np.float32)**2

            var_mean = np.mean(local_var)

            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            v_channel = hsv[:, :, 2]
            s_channel = hsv[:, :, 1]

            sat_uniformity = 1.0 - (np.std(s_channel) / 128.0)
            val_uniformity = 1.0 - (np.std(v_channel) / 128.0)

            if var_mean > 500 and sat_uniformity < 0.5:
                return 0.90
            elif var_mean > 200 and sat_uniformity < 0.7:
                return 0.65
            elif var_mean > 100:
                return 0.40
            elif sat_uniformity > 0.8:
                return 0.20
            else:
                return 0.30

        except Exception as e:
            logger.warning(f"Paper reflectance analysis failed: {str(e)}")
            return 0.5

    # ------------------------------------------------------------------
    # OPTIONAL: motion-based liveness (still useful as an extra signal if
    # you have burst/video frames available; not part of the core verdict
    # above since detect_spoofing works on a single frame)
    # ------------------------------------------------------------------

    def _analyze_motion_liveness(self, frames: List[np.ndarray]) -> float:
        """Motion liveness check across a short burst of frames."""
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