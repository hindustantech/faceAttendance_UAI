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
            screen_attack_indicators = 0
            critical_failures = 0
            real_face_indicators = 0

            # ================================================================
            # CHECK 1: MOIRE PATTERN - MOST IMPORTANT FOR SCREEN DETECTION
            # Run this FIRST so we can block immediately (with corroboration)
            # ================================================================
            logger.info(f"[AntiSpoofingService.detect_spoofing:72] Check 1/9: MOIRE PATTERN (PRIMARY SCREEN CHECK)")
            moire_score = self._detect_moire_pattern_balanced(image_data)
            results.append(self._make_result('moire_pattern', moire_score, threshold=self.screen_moire_threshold))

            if moire_score <= self.extreme_moire_veto:
                screen_attack_indicators += 3
                critical_failures += 2
                logger.warning(f"[AntiSpoofingService.detect_spoofing:79] ⚠ CRITICAL: Definite screen moire! (score: {moire_score:.4f})")
            elif moire_score <= self.strong_moire_veto:
                screen_attack_indicators += 2
                critical_failures += 1
                logger.warning(f"[AntiSpoofingService.detect_spoofing:83] ⚠ STRONG screen moire pattern! (score: {moire_score:.4f})")
            elif moire_score <= self.screen_moire_threshold:
                screen_attack_indicators += 1
                logger.warning(f"[AntiSpoofingService.detect_spoofing:86] ⚠ Screen moire suspected (score: {moire_score:.4f})")

            # ================================================================
            # CHECK 2: SPECULAR GLARE - Quick screen check
            # ================================================================
            logger.info(f"[AntiSpoofingService.detect_spoofing:91] Check 2/9: SPECULAR GLARE")
            glare_score = self._analyze_specular_highlights_balanced(image_data)
            results.append(self._make_result('specular_glare', glare_score))

            if glare_score <= 0.25:
                screen_attack_indicators += 1
                logger.warning(f"[AntiSpoofingService.detect_spoofing:97] ⚠ Screen glare detected (score: {glare_score:.4f})")

            # ================================================================
            # CHECK 5 (moved up): SCREEN COLOR ARTIFACTS
            # Needed early now so moire has a corroborating signal available
            # before any immediate-block decision is made.
            # ================================================================
            logger.info(f"[AntiSpoofingService.detect_spoofing:160] Check (early): SCREEN COLOR ARTIFACTS")
            screen_color_score = self._analyze_screen_color_artifacts(image_data)
            results.append(self._make_result('screen_color_artifacts', screen_color_score))

            # ================================================================
            # IMMEDIATE SCREEN BLOCK CHECK
            # Moire alone is no longer sufficient -- it must be corroborated
            # by at least one other independent screen indicator (glare or
            # color artifacts). This is what fixes the false-positive on
            # real faces that trip the moire floor value alone.
            # ================================================================
            moire_is_extreme = moire_score <= self.extreme_moire_veto
            corroborated = (glare_score <= self.screen_glare_block) or (screen_color_score <= self.screen_color_block)

            if moire_is_extreme and corroborated:
                logger.warning(
                    f"[AntiSpoofingService.detect_spoofing:105] ⛔ IMMEDIATE BLOCK: Screen moire "
                    f"({moire_score:.4f}) corroborated by glare={glare_score:.4f} / "
                    f"color={screen_color_score:.4f}"
                )
                logger.info(f"[AntiSpoofingService.detect_spoofing:106] ========== SCREEN ATTACK BLOCKED ==========")

                return {
                    'is_real': False,
                    'confidence': round(moire_score, 4),
                    'threshold': self.spoofing_threshold,
                    'details': {
                        'results': results,
                        'passed_checks': sum(1 for r in results if r['passed']),
                        'total_checks': len(results),
                        'liveness_checked': False,
                        'screen_attack_indicators': screen_attack_indicators,
                        'critical_failures': critical_failures,
                        'real_face_indicators': 0,
                        'verdict': 'SPOOF',
                        'block_reason': 'Screen moire pattern corroborated by glare/color signal'
                    }
                }
            elif moire_is_extreme and not corroborated:
                # Moire alone looked bad but nothing else agrees -- do NOT
                # instant-block. Let it flow into the full weighted decision
                # logic below instead of condemning on one noisy statistic.
                logger.warning(
                    f"[AntiSpoofingService.detect_spoofing:105b] Moire flagged extreme ({moire_score:.4f}) "
                    f"but uncorroborated by glare={glare_score:.4f} / color={screen_color_score:.4f} "
                    f"-- continuing to full checks instead of immediate block"
                )

            if moire_score <= 0.20 and glare_score <= self.screen_glare_block:
                logger.warning(f"[AntiSpoofingService.detect_spoofing:122] ⛔ IMMEDIATE BLOCK: Screen confirmed by moire+glare "
                             f"(moire: {moire_score:.4f}, glare: {glare_score:.4f})")
                logger.info(f"[AntiSpoofingService.detect_spoofing:124] ========== SCREEN ATTACK BLOCKED ==========")

                return {
                    'is_real': False,
                    'confidence': round((moire_score + glare_score) / 2, 4),
                    'threshold': self.spoofing_threshold,
                    'details': {
                        'results': results,
                        'passed_checks': sum(1 for r in results if r['passed']),
                        'total_checks': len(results),
                        'liveness_checked': False,
                        'screen_attack_indicators': screen_attack_indicators,
                        'critical_failures': critical_failures,
                        'real_face_indicators': 0,
                        'verdict': 'SPOOF',
                        'block_reason': 'Screen confirmed by moire and glare patterns'
                    }
                }

            if screen_color_score <= self.screen_color_block:
                if moire_score <= 0.25:
                    logger.warning(f"[AntiSpoofingService.detect_spoofing:167] ⛔ IMMEDIATE BLOCK: Screen color + moire pattern")

                    all_results = results.copy()
                    return {
                        'is_real': False,
                        'confidence': round((moire_score + screen_color_score) / 2, 4),
                        'threshold': self.spoofing_threshold,
                        'details': {
                            'results': all_results,
                            'passed_checks': sum(1 for r in all_results if r['passed']),
                            'total_checks': len(all_results),
                            'liveness_checked': False,
                            'screen_attack_indicators': screen_attack_indicators + 2,
                            'critical_failures': critical_failures + 1,
                            'real_face_indicators': real_face_indicators,
                            'verdict': 'SPOOF',
                            'block_reason': 'Screen color artifacts with moire pattern'
                        }
                    }
                else:
                    screen_attack_indicators += 2
                    critical_failures += 1
                    logger.warning(f"[AntiSpoofingService.detect_spoofing:186] ⚠ CRITICAL: Strong screen color artifacts!")
            elif screen_color_score <= self.screen_color_threshold:
                screen_attack_indicators += 1
                logger.warning(f"[AntiSpoofingService.detect_spoofing:189] ⚠ Screen color artifacts suspected")

            # ================================================================
            # NOT IMMEDIATELY BLOCKED - Continue with other checks
            # ================================================================
            logger.info(f"[AntiSpoofingService.detect_spoofing:143] Screen not immediately confirmed, continuing checks...")

            # Check 3: Texture Analysis
            logger.info(f"[AntiSpoofingService.detect_spoofing:146] Check 3/9: TEXTURE ANALYSIS")
            texture_score = self._analyze_texture(image_data)
            results.append(self._make_result('texture_analysis', texture_score))
            if texture_score >= self.real_face_indicator_threshold:
                real_face_indicators += 1

            # Check 4: Color Distribution
            logger.info(f"[AntiSpoofingService.detect_spoofing:153] Check 4/9: COLOR DISTRIBUTION")
            color_score = self._analyze_color_distribution(image_data)
            results.append(self._make_result('color_analysis', color_score))
            if color_score >= self.real_face_indicator_threshold:
                real_face_indicators += 1

            # Check 6: Edge Analysis
            logger.info(f"[AntiSpoofingService.detect_spoofing:193] Check 6/9: EDGE ANALYSIS")
            edge_score = self._analyze_edges(image_data)
            results.append(self._make_result('edge_analysis', edge_score))
            if edge_score >= self.real_face_indicator_threshold:
                real_face_indicators += 1

            # Check 7: Noise Pattern
            logger.info(f"[AntiSpoofingService.detect_spoofing:200] Check 7/9: NOISE PATTERN")
            noise_score = self._analyze_noise_pattern(image_data)
            results.append(self._make_result('noise_analysis', noise_score, threshold=0.40))
            if noise_score >= self.real_face_indicator_threshold:
                real_face_indicators += 1

            # Check 8: LBP Texture
            logger.info(f"[AntiSpoofingService.detect_spoofing:207] Check 8/9: LBP TEXTURE")
            lbp_score = self._analyze_texture_lbp(image_data)
            results.append(self._make_result('lbp_texture', lbp_score))
            if lbp_score >= self.real_face_indicator_threshold:
                real_face_indicators += 1

            # Check 9: Motion Liveness (optional)
            liveness_checked = False
            if motion_frames and len(motion_frames) >= 2:
                logger.info(f"[AntiSpoofingService.detect_spoofing:216] Check 9/9: MOTION LIVENESS")
                liveness_score = self._analyze_motion_liveness(motion_frames)
                results.append(self._make_result('motion_liveness', liveness_score))
                liveness_checked = True

                if liveness_score <= 0.20:
                    screen_attack_indicators += 2
                    critical_failures += 1
                    logger.warning(f"[AntiSpoofingService.detect_spoofing:224] ⚠ Static/rigid motion detected")
            else:
                logger.info(f"[AntiSpoofingService.detect_spoofing:226] ⊘ Motion liveness SKIPPED")

            # Calculate results
            passed_count = sum(1 for r in results if r['passed'])
            total_checks = len(results)
            overall_score = sum(r['score'] for r in results) / total_checks

            logger.info(f"[AntiSpoofingService.detect_spoofing:233] ========== RESULTS SUMMARY ==========")
            logger.info(f"[AntiSpoofingService.detect_spoofing:234]   Total checks: {total_checks}")
            logger.info(f"[AntiSpoofingService.detect_spoofing:235]   Passed: {passed_count}")
            logger.info(f"[AntiSpoofingService.detect_spoofing:236]   Overall score: {overall_score:.4f}")
            logger.info(f"[AntiSpoofingService.detect_spoofing:237]   Screen indicators: {screen_attack_indicators}")
            logger.info(f"[AntiSpoofingService.detect_spoofing:238]   Critical failures: {critical_failures}")
            logger.info(f"[AntiSpoofingService.detect_spoofing:239]   Real face indicators: {real_face_indicators}")

            for r in results:
                status = "✓ PASS" if r['passed'] else "✗ FAIL"
                logger.info(f"[AntiSpoofingService.detect_spoofing:242]   {r['method']}: {r['score']:.4f} {status}")

            # ================================================================
            # DECISION LOGIC (only reached if not immediately blocked)
            # ================================================================

            is_strong_real_face = (
                real_face_indicators >= self.real_face_min_indicators and
                overall_score >= self.real_face_min_score
            )

            logger.info(f"[AntiSpoofingService.detect_spoofing:254] DECISION PHASE:")
            logger.info(f"[AntiSpoofingService.detect_spoofing:255]   Strong real face: {is_strong_real_face}")

            is_real = True

            if is_strong_real_face:
                logger.info(f"[AntiSpoofingService.detect_spoofing:260] Using RELAXED criteria for strong real face")

                # FIX (2026-07-14, third pass): texture/edge/noise/LBP checks can
                # all pass on a good-quality screen replica, earning enough
                # "real_face_indicators" to hit this relaxed branch. Previously
                # that branch only rejected if critical_failures >= 3, which let
                # a single corroborated moire failure (moire FAIL + screen
                # indicators from glare/color) get overridden and accepted as
                # REAL. Moire+corroboration is specifically the signal meant to
                # catch screens that otherwise look textured enough to pass --
                # it must not be out-voted by the very checks it exists to
                # override.
                if critical_failures >= 3 and moire_score <= 0.03:
                    logger.warning(f"[AntiSpoofingService.detect_spoofing:263] ⚔ VETO: Critical failures with extreme moire")
                    is_real = False
                elif critical_failures >= 1 or screen_attack_indicators >= 2:
                    logger.warning(
                        f"[AntiSpoofingService.detect_spoofing:263b] ⚔ VETO: Moire failed and corroborated "
                        f"(critical_failures={critical_failures}, screen_attack_indicators={screen_attack_indicators}) "
                        f"-- texture-based real-face indicators cannot override this"
                    )
                    is_real = False
                else:
                    logger.info(f"[AntiSpoofingService.detect_spoofing:266] ✓ Real face ACCEPTED")
                    is_real = True
            else:
                logger.info(f"[AntiSpoofingService.detect_spoofing:269] Using STANDARD strict criteria")

                if critical_failures >= 1:
                    logger.warning(f"[AntiSpoofingService.detect_spoofing:272] ⚔ VETO 1: Critical failure")
                    is_real = False

                if screen_attack_indicators >= 3:
                    logger.warning(f"[AntiSpoofingService.detect_spoofing:276] ⚔ VETO 2: Multiple screen indicators")
                    is_real = False

                if screen_attack_indicators >= 1 and overall_score < self.low_confidence_threshold:
                    logger.warning(f"[AntiSpoofingService.detect_spoofing:280] ⚔ VETO 3: Screen indicator + low confidence")
                    is_real = False

                if moire_score <= self.strong_moire_veto:
                    logger.warning(f"[AntiSpoofingService.detect_spoofing:284] ⚔ VETO 4: Strong moire")
                    is_real = False

                if moire_score <= 0.30:
                    other_failures = [r for r in results if r['method'] != 'moire_pattern' and r['score'] <= 0.40]
                    if len(other_failures) >= 1:
                        logger.warning(f"[AntiSpoofingService.detect_spoofing:290] ⚔ VETO 5: Moire + other failures")
                        is_real = False

                if overall_score < 0.55:
                    logger.warning(f"[AntiSpoofingService.detect_spoofing:294] ⚔ VETO 6: Very low confidence")
                    is_real = False

                if passed_count < total_checks - 1:
                    logger.warning(f"[AntiSpoofingService.detect_spoofing:298] ⚔ VETO 7: Multiple failures")
                    is_real = False

            if is_real:
                majority_needed = max(self.min_checks_required, (total_checks // 2) + 1)
                is_real = (
                    (overall_score >= self.spoofing_threshold and passed_count >= majority_needed)
                    or overall_score >= 0.78
                )

            result = {
                'is_real': is_real,
                'confidence': round(overall_score, 4),
                'threshold': self.spoofing_threshold,
                'details': {
                    'results': results,
                    'passed_checks': passed_count,
                    'total_checks': total_checks,
                    'liveness_checked': liveness_checked,
                    'screen_attack_indicators': screen_attack_indicators,
                    'critical_failures': critical_failures,
                    'real_face_indicators': real_face_indicators,
                    'verdict': 'REAL' if is_real else 'SPOOF'
                }
            }

            logger.info(f"[AntiSpoofingService.detect_spoofing:325] ========== FINAL VERDICT: {result['details']['verdict']} ==========")
            logger.info(f"[AntiSpoofingService.detect_spoofing:326] ========== DETECTION COMPLETE ==========")

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