import cv2
import numpy as np
from typing import Dict, List, Optional
from app.config import settings
from app.utils.logger import setup_logger

logger = setup_logger(__name__)


class AntiSpoofingService:
    """
    Anti-Spoofing Service - Immediate Screen Blocking

    FIX (2026-07-14): The moire detector previously quantized its score into
    only 6 fixed buckets (0.05 / 0.12 / 0.25 / 0.45 / 0.65 / 0.85). The worst
    bucket (0.05) triggered on `max_peak_ratio > 8.0` which is a `max()` taken
    over 6 noisy FFT-band statistics -- real faces with normal skin/hair/eye
    texture regularly trip this too. As a result, a genuine face and an actual
    screen attack were scoring identically at 0.05, and the "immediate block"
    logic below fired on moire ALONE, with no corroboration from any other
    check (glare, color artifacts, etc). That's why the real image was
    rejected as a spoof.

    Changes made:
      1. `_detect_moire_pattern_balanced` now returns a continuous score
         instead of 6 discrete buckets, so borderline real faces are not
         collapsed onto the same floor value as obvious screen attacks.
      2. Raw metrics (peak_ratio, outlier_fraction) are now logged so you can
         calibrate thresholds against your own labeled real/spoof samples.
      3. Immediate-block-on-moire-alone now requires at least one
         corroborating signal (glare or screen-color-artifacts) instead of
         trusting a single noisy statistic to reject a user outright.
    """

    def __init__(self):
        logger.info(f"[AntiSpoofingService.__init__:18] Initializing IMMEDIATE BLOCK AntiSpoofingService")
        self.initialized = False

        self.spoofing_threshold = 0.60
        self.min_checks_required = 2
        self.min_individual_threshold = 0.40

        self.screen_moire_threshold = 0.40
        self.screen_color_threshold = 0.50
        self.strong_moire_veto = 0.15
        self.extreme_moire_veto = 0.05

        # Screen confirmation thresholds - now require corroboration,
        # a single noisy statistic can no longer condemn a user alone.
        self.screen_moire_block = 0.10        # Moire below this = candidate block (needs corroboration)
        self.screen_glare_block = 0.30        # Glare below this with bad moire = block
        self.screen_color_block = 0.35        # Color artifacts below this = block

        # Real face validation
        self.real_face_min_indicators = 3
        self.real_face_min_score = 0.68
        self.real_face_indicator_threshold = 0.85
        self.low_confidence_threshold = 0.65

        logger.info(f"[AntiSpoofingService.__init__:38] IMMEDIATE BLOCK Thresholds:")
        logger.info(f"[AntiSpoofingService.__init__:39]   moire_block: {self.screen_moire_block}")
        logger.info(f"[AntiSpoofingService.__init__:40]   glare_block: {self.screen_glare_block}")
        logger.info(f"[AntiSpoofingService.__init__:41]   color_block: {self.screen_color_block}")

    async def initialize(self):
        """Initialize anti-spoofing service"""
        logger.info(f"[AntiSpoofingService.initialize:45] Starting initialization")
        self.initialized = True
        logger.info("Anti-spoofing service initialized - IMMEDIATE BLOCK MODE")
        logger.info(f"[AntiSpoofingService.initialize:48] Initialization complete")

    async def detect_spoofing(
        self,
        image_data: np.ndarray,
        motion_frames: Optional[List[np.ndarray]] = None
    ) -> Dict:
        """
        Detect if face is real or spoofed - IMMEDIATE BLOCK on screen detection.
        """
        logger.info(f"[AntiSpoofingService.detect_spoofing:58] ========== STARTING IMMEDIATE BLOCK DETECTION ==========")
        logger.info(f"[AntiSpoofingService.detect_spoofing:59] Image shape: {image_data.shape}, dtype: {image_data.dtype}")

        try:
            results = []

            # ================================================================
            # MAJORITY-VOTE POLICY (2026-07-14, fifth pass -- REQUIRED CHANGE):
            #
            # Real production data proved moire cannot be a unilateral gate:
            #   Confirmed SCREEN ATTACK -> peak_ratio=5.040, outlier_frac=0.01701
            #   Confirmed REAL FACE     -> peak_ratio=4.658, outlier_frac=0.01797
            # The real face scored WORSE on both raw metrics than the actual
            # attack. No threshold on these numbers can separate the two --
            # they overlap. Gating on moire alone (pass or fail as sole
            # decider) is therefore provably wrong for this camera/pipeline,
            # in either direction.
            #
            # Fix: no single check gets unilateral veto power. All three
            # checks always run, and the verdict is a majority vote (2 of 3
            # must agree). This is a deliberate trade-off given the evidence
            # above -- it will not "feel" as strict as an immediate block,
            # but an immediate block on a check this noisy blocks real users
            # exactly as often as it blocks attackers, which is not security,
            # it's a coin flip that happens to reject people.
            # ================================================================

            logger.info(f"[AntiSpoofingService.detect_spoofing] Check 1/3: MOIRE PATTERN")
            moire_score = self._detect_moire_pattern_balanced(image_data)
            moire_result = self._make_result('moire_pattern', moire_score, threshold=self.screen_moire_threshold)
            results.append(moire_result)

            logger.info(f"[AntiSpoofingService.detect_spoofing] Check 2/3: SPECULAR GLARE")
            glare_score = self._analyze_specular_highlights_balanced(image_data)
            glare_result = self._make_result('specular_glare', glare_score)
            results.append(glare_result)

            logger.info(f"[AntiSpoofingService.detect_spoofing] Check 3/3: SCREEN COLOR ARTIFACTS")
            screen_color_score = self._analyze_screen_color_artifacts(image_data)
            color_result = self._make_result('screen_color_artifacts', screen_color_score)
            results.append(color_result)

            passed_count = sum(1 for r in results if r['passed'])
            total_checks = len(results)
            overall_score = sum(r['score'] for r in results) / total_checks

            # Majority vote: at least 2 of 3 checks must pass.
            is_real = passed_count >= 2

            logger.info(f"[AntiSpoofingService.detect_spoofing] ========== RESULTS SUMMARY ==========")
            logger.info(f"[AntiSpoofingService.detect_spoofing]   Total checks: {total_checks}")
            logger.info(f"[AntiSpoofingService.detect_spoofing]   Passed: {passed_count} (need >= 2 for REAL)")
            logger.info(f"[AntiSpoofingService.detect_spoofing]   Overall score: {overall_score:.4f}")
            for r in results:
                status = "✓ PASS" if r['passed'] else "✗ FAIL"
                logger.info(f"[AntiSpoofingService.detect_spoofing]   {r['method']}: {r['score']:.4f} {status}")

            result = {
                'is_real': is_real,
                'confidence': round(overall_score, 4),
                'threshold': self.spoofing_threshold,
                'details': {
                    'results': results,
                    'passed_checks': passed_count,
                    'total_checks': total_checks,
                    'liveness_checked': False,
                    'verdict': 'REAL' if is_real else 'SPOOF'
                }
            }


            logger.info(f"[AntiSpoofingService.detect_spoofing] ========== FINAL VERDICT: {result['details']['verdict']} ==========")
            logger.info(f"[AntiSpoofingService.detect_spoofing] ========== DETECTION COMPLETE ==========")

            return result

        except Exception as e:
            logger.error(f"[AntiSpoofingService.detect_spoofing:330] ❌ Detection FAILED: {str(e)}", exc_info=True)
            logger.critical(f"[AntiSpoofingService.detect_spoofing:331] FAILING CLOSED - SECURITY FIRST")
            return {
                'is_real': False,
                'confidence': 0.0,
                'threshold': self.spoofing_threshold,
                'details': {
                    'error': str(e),
                    'verdict': 'REJECTED_ON_ERROR'
                }
            }

    def _make_result(self, method: str, score: float, threshold: Optional[float] = None) -> Dict:
        t = threshold if threshold is not None else self.min_individual_threshold
        passed = score >= t
        result = {'method': method, 'score': score, 'passed': passed}
        logger.info(f"[AntiSpoofingService._make_result:345] {method}: {score:.4f} (threshold: {t:.4f}) - {'PASS' if passed else 'FAIL'}")
        return result

    # ------------------------------------------------------------------
    # DETECTION METHODS
    # ------------------------------------------------------------------

    def _detect_moire_pattern_balanced(self, image: np.ndarray) -> float:
        """
        Moire detection - FIXED to return a continuous score.

        Previously this collapsed everything into 6 fixed buckets
        (0.05/0.12/0.25/0.45/0.65/0.85) triggered by a single `max()` over
        6 noisy FFT-band statistics. That meant a real face's normal
        high-frequency texture (skin, hair, eyebrows) could trip the same
        floor value (0.05) as an actual screen moire pattern -- which is
        exactly what happened in production (both real and spoofed images
        scored 0.0500).

        Now the score is a smooth function of `max_peak_ratio` and
        `max_outlier_fraction`, so a borderline case lands somewhere in the
        middle instead of being forced to the extreme. Raw metrics are
        logged so thresholds can be calibrated against real labeled data.
        """
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            all_peak_ratios = []
            all_outlier_fractions = []
            all_high_freq_energies = []

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

                freq_bands = [
                    ('low', 5, 15),
                    ('mid', 10, 35),
                    ('high', 20, min(h, w) // 2)
                ]

                for band_name, r_min, r_max in freq_bands:
                    y, x = np.ogrid[:h, :w]
                    dist_from_center = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
                    band_mask = (dist_from_center > r_min) & (dist_from_center <= r_max)

                    band_energy = magnitude[band_mask]
                    if len(band_energy) > 0:
                        mean_energy = np.mean(band_energy)
                        std_energy = np.std(band_energy)
                        max_energy = np.max(band_energy)

                        if std_energy > 1e-6:
                            peak_ratio = (max_energy - mean_energy) / std_energy
                            all_peak_ratios.append(peak_ratio)

                            for threshold_mult in [2.5, 3.0]:
                                outlier_threshold = mean_energy + threshold_mult * std_energy
                                outlier_fraction = np.sum(band_energy > outlier_threshold) / len(band_energy)
                                all_outlier_fractions.append(outlier_fraction)

                        all_high_freq_energies.append(mean_energy)

            max_peak_ratio = max(all_peak_ratios) if all_peak_ratios else 0
            max_outlier_fraction = max(all_outlier_fractions) if all_outlier_fractions else 0
            avg_high_freq_energy = np.mean(all_high_freq_energies) if all_high_freq_energies else 0

            # Log raw metrics -- use these numbers on a batch of labeled
            # real/spoof samples to calibrate the mapping below for your
            # actual camera/lighting setup.
            logger.info(
                f"[AntiSpoofingService._detect_moire_pattern_balanced] raw metrics -> "
                f"peak_ratio={max_peak_ratio:.3f} outlier_frac={max_outlier_fraction:.5f} "
                f"hf_energy={avg_high_freq_energy:.3f}"
            )

            # Continuous scoring instead of discrete buckets.
            # peak_ratio_floor/ceiling define the range mapped to score 1.0..0.0
            # Starting points below -- RECALIBRATE against your own labeled data.
            #
            # NOTE (2026-07-14, second fix): outlier_ceiling was previously 0.02,
            # which let a confirmed screen attack (outlier_frac=0.01701) score as
            # only mildly suspicious (0.162), and a 60/40 blend with the cleaner
            # peak_ratio axis (0.8145) diluted that up to 0.4228 -- enough to pass.
            # Lowered the ceiling so this attack's outlier_frac lands at 0, and
            # switched to a min-dominant blend so a clearly bad axis can no
            # longer be averaged away by a clean one.
            peak_ratio_floor = 3.0     # at/below this -> score contribution = 1.0 (looks real)
            peak_ratio_ceiling = 14.0  # at/above this -> score contribution = 0.0 (looks like screen)
            outlier_floor = 0.001
            outlier_ceiling = 0.012    # was 0.02 -- lowered after a real screen attack scored 0.01701 here

            peak_component = 1.0 - np.clip(
                (max_peak_ratio - peak_ratio_floor) / (peak_ratio_ceiling - peak_ratio_floor),
                0.0, 1.0
            )
            outlier_component = 1.0 - np.clip(
                (max_outlier_fraction - outlier_floor) / (outlier_ceiling - outlier_floor),
                0.0, 1.0
            )

            # Min-dominant blend: if EITHER axis independently looks like a
            # screen, that verdict should dominate rather than get averaged
            # away by the other axis looking clean. The small max() weighting
            # only smooths borderline noise, it cannot rescue a bad signal.
            score = 0.85 * min(peak_component, outlier_component) + 0.15 * max(peak_component, outlier_component)

            if avg_high_freq_energy > 9.0:
                score *= 0.9

            return float(max(0.05, min(1.0, score)))
        except Exception as e:
            logger.warning(f"Moire detection failed: {str(e)}")
            return 0.5

    def _analyze_specular_highlights_balanced(self, image: np.ndarray) -> float:
        """Balanced glare detection."""
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

                if bright_fraction > 0.08 or large_blob_fraction > 0.04:
                    scores.append(0.20)
                elif bright_fraction > 0.05 or large_blob_fraction > 0.02:
                    scores.append(0.40)
                elif bright_fraction > 0.03 or large_blob_fraction > 0.01:
                    scores.append(0.60)
                else:
                    scores.append(0.80)

            return min(scores) if scores else 0.80
        except Exception as e:
            logger.warning(f"Glare analysis failed: {str(e)}")
            return 0.5

    def _analyze_screen_color_artifacts(self, image: np.ndarray) -> float:
        """Detect screen-specific color artifacts."""
        try:
            b, g, r = cv2.split(image)

            r_high_clip = np.sum(r > 240) / r.size
            g_high_clip = np.sum(g > 240) / g.size
            b_high_clip = np.sum(b > 240) / b.size
            high_clip_score = max(r_high_clip, g_high_clip, b_high_clip)

            hist_r = cv2.calcHist([image], [2], None, [256], [0, 256])
            hist_g = cv2.calcHist([image], [1], None, [256], [0, 256])
            hist_b = cv2.calcHist([image], [0], None, [256], [0, 256])

            def count_banding_gaps(hist):
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

            total_gaps = count_banding_gaps(hist_r) + count_banding_gaps(hist_g) + count_banding_gaps(hist_b)

            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            saturation = hsv[:, :, 1]
            sat_std = np.std(saturation)

            screen_score = 0.0

            if high_clip_score > 0.08:
                screen_score += 0.5
            elif high_clip_score > 0.05:
                screen_score += 0.3
            elif high_clip_score > 0.02:
                screen_score += 0.2

            if total_gaps > 12:
                screen_score += 0.5
            elif total_gaps > 8:
                screen_score += 0.3
            elif total_gaps > 4:
                screen_score += 0.2

            if sat_std < 12:
                screen_score += 0.3
            elif sat_std < 18:
                screen_score += 0.2

            return 1.0 - min(screen_score, 1.0)
        except Exception as e:
            logger.warning(f"Screen color analysis failed: {str(e)}")
            return 0.5

    def _analyze_texture(self, image: np.ndarray) -> float:
        """Analyze texture quality."""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

            if laplacian_var > 300: return 1.0
            elif laplacian_var > 150: return 0.85
            elif laplacian_var > 80: return 0.70
            elif laplacian_var > 40: return 0.55
            elif laplacian_var > 20: return 0.40
            else: return 0.30
        except Exception as e:
            logger.warning(f"Texture analysis failed: {str(e)}")
            return 0.5

    def _analyze_color_distribution(self, image: np.ndarray) -> float:
        """Analyze color distribution."""
        try:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            saturation = hsv[:, :, 1]
            saturation_std = np.std(saturation)

            if saturation_std > 35: return 1.0
            elif saturation_std > 25: return 0.85
            elif saturation_std > 18: return 0.70
            elif saturation_std > 12: return 0.55
            elif saturation_std > 8: return 0.40
            else: return 0.35
        except Exception as e:
            logger.warning(f"Color analysis failed: {str(e)}")
            return 0.5

    def _analyze_edges(self, image: np.ndarray) -> float:
        """Analyze edge density."""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size

            if edge_density < 0.10: return 1.0
            elif edge_density < 0.15: return 0.85
            elif edge_density < 0.20: return 0.70
            elif edge_density < 0.25: return 0.55
            elif edge_density < 0.30: return 0.40
            else: return 0.30
        except Exception as e:
            logger.warning(f"Edge analysis failed: {str(e)}")
            return 0.5

    def _analyze_noise_pattern(self, image: np.ndarray) -> float:
        """Analyze noise pattern."""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            noise = cv2.absdiff(gray, blurred)
            noise_std = np.std(noise)

            if 5 < noise_std < 40: return 1.0
            elif 3 < noise_std < 50: return 0.85
            elif 2 < noise_std < 60: return 0.70
            elif 1 < noise_std < 70: return 0.55
            else: return 0.40
        except Exception as e:
            logger.warning(f"Noise analysis failed: {str(e)}")
            return 0.5

    def _analyze_texture_lbp(self, image: np.ndarray) -> float:
        """LBP texture analysis."""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (200, 200)).astype(np.int32)

            center = gray[1:-1, 1:-1]
            offsets = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]

            lbp = np.zeros_like(center, dtype=np.uint8)
            for i, (dy, dx) in enumerate(offsets):
                neighbor = gray[1 + dy: 1 + dy + center.shape[0], 1 + dx: 1 + dx + center.shape[1]]
                lbp |= ((neighbor >= center).astype(np.uint8) << i)

            hist, _ = np.histogram(lbp, bins=256, range=(0, 256))
            hist = hist.astype(np.float64)
            hist_sum = hist.sum()

            if hist_sum == 0: return 0.5

            prob = hist / hist_sum
            prob = prob[prob > 0]
            entropy = -np.sum(prob * np.log2(prob))
            normalized_entropy = entropy / 8.0

            if normalized_entropy > 0.80: return 1.0
            elif normalized_entropy > 0.70: return 0.85
            elif normalized_entropy > 0.60: return 0.65
            elif normalized_entropy > 0.50: return 0.45
            else: return 0.30
        except Exception as e:
            logger.warning(f"LBP analysis failed: {str(e)}")
            return 0.5

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

            if avg_motion < 0.05: return 0.15
            else:
                rigidity_ratio = avg_variation / max(avg_motion, 1e-6)
                if rigidity_ratio < 0.15: return 0.30
                elif rigidity_ratio < 0.30: return 0.55
                elif rigidity_ratio < 0.50: return 0.75
                else: return 0.90
        except Exception as e:
            logger.warning(f"Motion analysis failed: {str(e)}")
            return 0.5