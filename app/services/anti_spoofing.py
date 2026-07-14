import cv2
import numpy as np
from typing import Dict, List, Optional
from app.config import settings
from app.utils.logger import setup_logger

logger = setup_logger(__name__)


class AntiSpoofingService:
    """
    Anti-Spoofing Service - Fixed Screen Attack Detection
    Detects presentation attacks: printed photos, screen replays
    (phone/tablet/monitor/laptop), and static images held up to the camera.

    KEY FIX: Screen confirmation uses moire+glare combo before allowing real face bypass.
    """

    def __init__(self):
        logger.info(f"[AntiSpoofingService.__init__:21] Initializing FIXED AntiSpoofingService")
        self.initialized = False
        
        # BALANCED THRESHOLDS
        self.spoofing_threshold = 0.60
        self.min_checks_required = 2
        self.min_individual_threshold = 0.40
        
        # Screen-specific thresholds
        self.screen_moire_threshold = 0.40
        self.screen_color_threshold = 0.50
        self.strong_moire_veto = 0.15
        self.extreme_moire_veto = 0.05
        
        # Screen confirmation thresholds (KEY FIX)
        self.screen_moire_confirm = 0.10  # Moire below this is suspicious
        self.screen_glare_confirm = 0.30  # Glare below this confirms screen
        
        # Real face validation
        self.real_face_min_indicators = 3
        self.real_face_min_score = 0.68
        self.real_face_indicator_threshold = 0.85
        self.low_confidence_threshold = 0.65
        
        logger.info(f"[AntiSpoofingService.__init__:40] FIXED Thresholds:")
        logger.info(f"[AntiSpoofingService.__init__:41]   spoofing: {self.spoofing_threshold}")
        logger.info(f"[AntiSpoofingService.__init__:42]   screen_confirm: moire<{self.screen_moire_confirm} & glare<{self.screen_glare_confirm}")

    async def initialize(self):
        """Initialize anti-spoofing service"""
        logger.info(f"[AntiSpoofingService.initialize:46] Starting initialization")
        self.initialized = True
        logger.info("Anti-spoofing service initialized - FIXED MODE")
        logger.info(f"[AntiSpoofingService.initialize:49] Initialization complete")

    async def detect_spoofing(
        self,
        image_data: np.ndarray,
        motion_frames: Optional[List[np.ndarray]] = None
    ) -> Dict:
        """
        Detect if face is real or spoofed - FIXED detection.
        """
        logger.info(f"[AntiSpoofingService.detect_spoofing:59] ========== STARTING FIXED SPOOFING DETECTION ==========")
        logger.info(f"[AntiSpoofingService.detect_spoofing:60] Image shape: {image_data.shape}, dtype: {image_data.dtype}")
        logger.info(f"[AntiSpoofingService.detect_spoofing:61] Motion frames: {motion_frames is not None}, "
                    f"count: {len(motion_frames) if motion_frames else 0}")
        
        try:
            results = []
            screen_attack_indicators = 0
            critical_failures = 0
            real_face_indicators = 0
            
            # Check 1: Texture Analysis
            logger.info(f"[AntiSpoofingService.detect_spoofing:71] Check 1/9: TEXTURE ANALYSIS")
            texture_score = self._analyze_texture(image_data)
            results.append(self._make_result('texture_analysis', texture_score))
            if texture_score >= self.real_face_indicator_threshold:
                real_face_indicators += 1
                logger.info(f"[AntiSpoofingService.detect_spoofing:76]   ✓ Real face indicator: texture={texture_score:.4f}")
            
            # Check 2: Color Distribution
            logger.info(f"[AntiSpoofingService.detect_spoofing:80] Check 2/9: COLOR DISTRIBUTION")
            color_score = self._analyze_color_distribution(image_data)
            results.append(self._make_result('color_analysis', color_score))
            if color_score >= self.real_face_indicator_threshold:
                real_face_indicators += 1
                logger.info(f"[AntiSpoofingService.detect_spoofing:85]   ✓ Real face indicator: color={color_score:.4f}")
            
            # Check 3: Screen Color Artifacts
            logger.info(f"[AntiSpoofingService.detect_spoofing:89] Check 3/9: SCREEN COLOR ARTIFACTS")
            screen_color_score = self._analyze_screen_color_artifacts(image_data)
            results.append(self._make_result('screen_color_artifacts', screen_color_score))
            
            if screen_color_score <= 0.35:
                screen_attack_indicators += 2
                critical_failures += 1
                logger.warning(f"[AntiSpoofingService.detect_spoofing:96] ⚠ CRITICAL: Strong screen color artifacts! (score: {screen_color_score:.4f})")
            elif screen_color_score <= self.screen_color_threshold:
                screen_attack_indicators += 1
                logger.warning(f"[AntiSpoofingService.detect_spoofing:99] ⚠ Screen color artifacts suspected (score: {screen_color_score:.4f})")
            
            # Check 4: Edge Analysis
            logger.info(f"[AntiSpoofingService.detect_spoofing:103] Check 4/9: EDGE ANALYSIS")
            edge_score = self._analyze_edges(image_data)
            results.append(self._make_result('edge_analysis', edge_score))
            if edge_score >= self.real_face_indicator_threshold:
                real_face_indicators += 1
                logger.info(f"[AntiSpoofingService.detect_spoofing:108]   ✓ Real face indicator: edge={edge_score:.4f}")
            
            # Check 5: Noise Pattern
            logger.info(f"[AntiSpoofingService.detect_spoofing:112] Check 5/9: NOISE PATTERN")
            noise_score = self._analyze_noise_pattern(image_data)
            results.append(self._make_result('noise_analysis', noise_score, threshold=0.40))
            if noise_score >= self.real_face_indicator_threshold:
                real_face_indicators += 1
                logger.info(f"[AntiSpoofingService.detect_spoofing:117]   ✓ Real face indicator: noise={noise_score:.4f}")
            
            # Check 6: Moire Pattern (CRITICAL FOR SCREENS)
            logger.info(f"[AntiSpoofingService.detect_spoofing:121] Check 6/9: MOIRE PATTERN")
            moire_score = self._detect_moire_pattern_balanced(image_data)
            results.append(self._make_result('moire_pattern', moire_score, threshold=self.screen_moire_threshold))
            
            if moire_score <= self.extreme_moire_veto:
                screen_attack_indicators += 3
                critical_failures += 2
                logger.warning(f"[AntiSpoofingService.detect_spoofing:128] ⚠ CRITICAL: Definite screen moire! (score: {moire_score:.4f})")
            elif moire_score <= self.strong_moire_veto:
                screen_attack_indicators += 2
                critical_failures += 1
                logger.warning(f"[AntiSpoofingService.detect_spoofing:132] ⚠ STRONG screen moire pattern! (score: {moire_score:.4f})")
            elif moire_score <= self.screen_moire_threshold:
                screen_attack_indicators += 1
                logger.warning(f"[AntiSpoofingService.detect_spoofing:135] ⚠ Screen moire suspected (score: {moire_score:.4f})")
            
            # Check 7: LBP Texture
            logger.info(f"[AntiSpoofingService.detect_spoofing:139] Check 7/9: LBP TEXTURE")
            lbp_score = self._analyze_texture_lbp(image_data)
            results.append(self._make_result('lbp_texture', lbp_score))
            if lbp_score >= self.real_face_indicator_threshold:
                real_face_indicators += 1
                logger.info(f"[AntiSpoofingService.detect_spoofing:144]   ✓ Real face indicator: lbp={lbp_score:.4f}")
            
            # Check 8: Specular Glare
            logger.info(f"[AntiSpoofingService.detect_spoofing:148] Check 8/9: SPECULAR GLARE")
            glare_score = self._analyze_specular_highlights_balanced(image_data)
            results.append(self._make_result('specular_glare', glare_score))
            
            if glare_score <= 0.25:
                screen_attack_indicators += 1
                logger.warning(f"[AntiSpoofingService.detect_spoofing:154] ⚠ Screen glare detected (score: {glare_score:.4f})")
            
            # Check 9: Motion Liveness (optional)
            liveness_checked = False
            if motion_frames and len(motion_frames) >= 2:
                logger.info(f"[AntiSpoofingService.detect_spoofing:160] Check 9/9: MOTION LIVENESS")
                liveness_score = self._analyze_motion_liveness(motion_frames)
                results.append(self._make_result('motion_liveness', liveness_score))
                liveness_checked = True
                
                if liveness_score <= 0.20:
                    screen_attack_indicators += 2
                    critical_failures += 1
                    logger.warning(f"[AntiSpoofingService.detect_spoofing:168] ⚠ Static/rigid motion detected")
            else:
                logger.info(f"[AntiSpoofingService.detect_spoofing:170] ⊘ Motion liveness SKIPPED")

            # Calculate results
            passed_count = sum(1 for r in results if r['passed'])
            total_checks = len(results)
            overall_score = sum(r['score'] for r in results) / total_checks

            logger.info(f"[AntiSpoofingService.detect_spoofing:177] ========== RESULTS SUMMARY ==========")
            logger.info(f"[AntiSpoofingService.detect_spoofing:178]   Total checks: {total_checks}")
            logger.info(f"[AntiSpoofingService.detect_spoofing:179]   Passed: {passed_count}")
            logger.info(f"[AntiSpoofingService.detect_spoofing:180]   Overall score: {overall_score:.4f}")
            logger.info(f"[AntiSpoofingService.detect_spoofing:181]   Screen indicators: {screen_attack_indicators}")
            logger.info(f"[AntiSpoofingService.detect_spoofing:182]   Critical failures: {critical_failures}")
            logger.info(f"[AntiSpoofingService.detect_spoofing:183]   Real face indicators: {real_face_indicators}")
            
            for r in results:
                status = "✓ PASS" if r['passed'] else "✗ FAIL"
                logger.info(f"[AntiSpoofingService.detect_spoofing:186]   {r['method']}: {r['score']:.4f} {status}")

            # ================================================================
            # FIXED DECISION LOGIC
            # ================================================================
            
            # Check if this has strong real face characteristics
            is_strong_real_face = (
                real_face_indicators >= self.real_face_min_indicators and 
                overall_score >= self.real_face_min_score
            )
            
            # Get critical scores
            moire_result = next(r for r in results if r['method'] == 'moire_pattern')
            glare_result = next(r for r in results if r['method'] == 'specular_glare')
            
            # KEY FIX: Screen confirmation using moire+glare combo
            # Bad moire + bad glare = DEFINITELY a screen, regardless of other checks
            screen_confirmed = (
                moire_result['score'] <= self.screen_moire_confirm and 
                glare_result['score'] <= self.screen_glare_confirm
            )
            
            logger.info(f"[AntiSpoofingService.detect_spoofing:206] DECISION PHASE:")
            logger.info(f"[AntiSpoofingService.detect_spoofing:207]   Moire: {moire_result['score']:.4f}, Glare: {glare_result['score']:.4f}")
            logger.info(f"[AntiSpoofingService.detect_spoofing:208]   Screen confirmed: {screen_confirmed}")
            logger.info(f"[AntiSpoofingService.detect_spoofing:209]   Strong real face: {is_strong_real_face}")
            
            is_real = True
            
            # SCREEN CONFIRMED - Reject immediately, no exceptions
            if screen_confirmed:
                logger.warning(f"[AntiSpoofingService.detect_spoofing:214] ⚔ SCREEN CONFIRMED: "
                             f"Bad moire ({moire_result['score']:.4f}) + Bad glare ({glare_result['score']:.4f}) = DEFINITE SCREEN")
                is_real = False
            
            # STRONG REAL FACE with no screen confirmation
            elif is_strong_real_face:
                logger.info(f"[AntiSpoofingService.detect_spoofing:219] Using RELAXED criteria for strong real face "
                           f"({real_face_indicators} indicators, score: {overall_score:.4f})")
                
                # Only reject if EXTREME conditions
                if critical_failures >= 3 and moire_result['score'] <= 0.03:
                    logger.warning(f"[AntiSpoofingService.detect_spoofing:224] ⚔ VETO REAL: {critical_failures} critical failures "
                                 f"with extreme moire ({moire_result['score']:.4f})")
                    is_real = False
                elif moire_result['score'] <= 0.03:
                    logger.warning(f"[AntiSpoofingService.detect_spoofing:227] ⚔ VETO REAL: Extreme moire ({moire_result['score']:.4f})")
                    is_real = False
                else:
                    logger.info(f"[AntiSpoofingService.detect_spoofing:230] ✓ Real face ACCEPTED")
                    is_real = True
            
            # STANDARD PATH - Full strict criteria
            else:
                logger.info(f"[AntiSpoofingService.detect_spoofing:235] Using STANDARD strict criteria")
                
                # VETO 1: Critical failures
                if critical_failures >= 1:
                    logger.warning(f"[AntiSpoofingService.detect_spoofing:239] ⚔ VETO 1: Critical failure ({critical_failures})")
                    is_real = False
                
                # VETO 2: Multiple screen indicators
                if screen_attack_indicators >= 3:
                    logger.warning(f"[AntiSpoofingService.detect_spoofing:244] ⚔ VETO 2: Multiple screen indicators ({screen_attack_indicators})")
                    is_real = False
                
                # VETO 3: Screen indicator + low confidence
                if screen_attack_indicators >= 1 and overall_score < self.low_confidence_threshold:
                    logger.warning(f"[AntiSpoofingService.detect_spoofing:249] ⚔ VETO 3: Screen indicator with low confidence "
                                 f"(score: {overall_score:.4f}, indicators: {screen_attack_indicators})")
                    is_real = False
                
                # VETO 4: Strong moire
                if moire_result['score'] <= self.strong_moire_veto:
                    logger.warning(f"[AntiSpoofingService.detect_spoofing:254] ⚔ VETO 4: Strong moire ({moire_result['score']:.4f})")
                    is_real = False
                
                # VETO 5: Moire + other failures
                if moire_result['score'] <= 0.30:
                    other_failures = [r for r in results if r['method'] != 'moire_pattern' and r['score'] <= 0.40]
                    if len(other_failures) >= 1:
                        logger.warning(f"[AntiSpoofingService.detect_spoofing:261] ⚔ VETO 5: Moire + other failures "
                                     f"(moire: {moire_result['score']:.4f}, other: {len(other_failures)})")
                        is_real = False
                
                # VETO 6: Very low overall score
                if overall_score < 0.55:
                    logger.warning(f"[AntiSpoofingService.detect_spoofing:267] ⚔ VETO 6: Very low confidence ({overall_score:.4f})")
                    is_real = False
                
                # VETO 7: Multiple check failures
                if passed_count < total_checks - 1:
                    logger.warning(f"[AntiSpoofingService.detect_spoofing:272] ⚔ VETO 7: Multiple failures ({passed_count}/{total_checks})")
                    is_real = False

            # STANDARD CHECK (if all vetos passed)
            if is_real:
                majority_needed = max(self.min_checks_required, (total_checks // 2) + 1)
                
                is_real = (
                    (overall_score >= self.spoofing_threshold and passed_count >= majority_needed)
                    or overall_score >= 0.78
                )
                
                logger.info(f"[AntiSpoofingService.detect_spoofing:284] STANDARD CHECK:")
                logger.info(f"[AntiSpoofingService.detect_spoofing:285]   Score >= {self.spoofing_threshold}: {overall_score >= self.spoofing_threshold}")
                logger.info(f"[AntiSpoofingService.detect_spoofing:286]   Passed >= {majority_needed}: {passed_count >= majority_needed}")
                logger.info(f"[AntiSpoofingService.detect_spoofing:287]   Score >= 0.78: {overall_score >= 0.78}")

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

            logger.info(f"[AntiSpoofingService.detect_spoofing:304] ========== FINAL VERDICT: {result['details']['verdict']} ==========")
            logger.info(f"[AntiSpoofingService.detect_spoofing:305] Score: {result['confidence']:.4f}, "
                       f"Passed: {passed_count}/{total_checks}")
            logger.info(f"[AntiSpoofingService.detect_spoofing:307] Screen: {screen_attack_indicators}, "
                       f"Critical: {critical_failures}, Real: {real_face_indicators}")
            logger.info(f"[AntiSpoofingService.detect_spoofing:309] ========== DETECTION COMPLETE ==========")

            return result

        except Exception as e:
            logger.error(f"[AntiSpoofingService.detect_spoofing:313] ❌ Detection FAILED: {str(e)}", exc_info=True)
            logger.critical(f"[AntiSpoofingService.detect_spoofing:314] FAILING CLOSED - SECURITY FIRST")
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
        logger.info(f"[AntiSpoofingService._make_result:328] {method}: {score:.4f} (threshold: {t:.4f}) - {'PASS' if passed else 'FAIL'}")
        return result

    # ------------------------------------------------------------------
    # DETECTION METHODS
    # ------------------------------------------------------------------

    def _analyze_screen_color_artifacts(self, image: np.ndarray) -> float:
        """Detect screen-specific color artifacts."""
        logger.info(f"[AntiSpoofingService._analyze_screen_color_artifacts:338] Screen color analysis...")
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
            
            logger.info(f"[AntiSpoofingService._analyze_screen_color_artifacts:369] "
                        f"High clip: {high_clip_score:.4f}, Gaps: {total_gaps}, Sat std: {sat_std:.2f}")
            
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
            
            final_score = 1.0 - min(screen_score, 1.0)
            
            logger.info(f"[AntiSpoofingService._analyze_screen_color_artifacts:391] "
                        f"Screen color score: {final_score:.4f}")
            
            return final_score

        except Exception as e:
            logger.warning(f"[AntiSpoofingService._analyze_screen_color_artifacts:396] ❌ Failed: {str(e)}")
            return 0.5

    def _detect_moire_pattern_balanced(self, image: np.ndarray) -> float:
        """Balanced moire detection."""
        logger.info(f"[AntiSpoofingService._detect_moire_pattern_balanced:402] Balanced moire detection...")
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
                    ('high', 20, min(h, w)//2)
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
            
            logger.info(f"[AntiSpoofingService._detect_moire_pattern_balanced:461] "
                        f"Peak ratio: {max_peak_ratio:.4f}, Outlier: {max_outlier_fraction:.6f}, "
                        f"Avg energy: {avg_high_freq_energy:.4f}")
            
            if max_peak_ratio > 8.0 or max_outlier_fraction > 0.008:
                score = 0.05
            elif max_peak_ratio > 6.0 or max_outlier_fraction > 0.005:
                score = 0.12
            elif max_peak_ratio > 4.5 or max_outlier_fraction > 0.002:
                score = 0.25
            elif max_peak_ratio > 3.5:
                score = 0.45
            elif max_peak_ratio > 2.5:
                score = 0.65
            else:
                score = 0.85
            
            if avg_high_freq_energy > 9.0:
                score *= 0.8
            
            score = max(0.05, min(1.0, score))
            
            logger.info(f"[AntiSpoofingService._detect_moire_pattern_balanced:482] "
                        f"Final moire score: {score:.4f}")
            
            return score

        except Exception as e:
            logger.warning(f"[AntiSpoofingService._detect_moire_pattern_balanced:487] ❌ Failed: {str(e)}")
            return 0.5

    def _analyze_specular_highlights_balanced(self, image: np.ndarray) -> float:
        """Balanced glare detection."""
        logger.info(f"[AntiSpoofingService._analyze_specular_highlights_balanced:493] Glare analysis...")
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
            
            final_score = min(scores) if scores else 0.80
            
            logger.info(f"[AntiSpoofingService._analyze_specular_highlights_balanced:526] "
                        f"Glare score: {final_score:.4f}")
            
            return final_score

        except Exception as e:
            logger.warning(f"[AntiSpoofingService._analyze_specular_highlights_balanced:531] ❌ Failed: {str(e)}")
            return 0.5

    # ------------------------------------------------------------------
    # STANDARD CHECKS
    # ------------------------------------------------------------------

    def _analyze_texture(self, image: np.ndarray) -> float:
        """Analyze texture quality."""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            if laplacian_var > 300:
                return 1.0
            elif laplacian_var > 150:
                return 0.85
            elif laplacian_var > 80:
                return 0.70
            elif laplacian_var > 40:
                return 0.55
            elif laplacian_var > 20:
                return 0.40
            else:
                return 0.30
        except Exception as e:
            logger.warning(f"Texture analysis failed: {str(e)}")
            return 0.5

    def _analyze_color_distribution(self, image: np.ndarray) -> float:
        """Analyze color distribution."""
        try:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            saturation = hsv[:, :, 1]
            saturation_std = np.std(saturation)
            
            if saturation_std > 35:
                return 1.0
            elif saturation_std > 25:
                return 0.85
            elif saturation_std > 18:
                return 0.70
            elif saturation_std > 12:
                return 0.55
            elif saturation_std > 8:
                return 0.40
            else:
                return 0.35
        except Exception as e:
            logger.warning(f"Color analysis failed: {str(e)}")
            return 0.5

    def _analyze_edges(self, image: np.ndarray) -> float:
        """Analyze edge density."""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            
            if edge_density < 0.10:
                return 1.0
            elif edge_density < 0.15:
                return 0.85
            elif edge_density < 0.20:
                return 0.70
            elif edge_density < 0.25:
                return 0.55
            elif edge_density < 0.30:
                return 0.40
            else:
                return 0.30
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
            
            if hist_sum == 0:
                return 0.5
            
            prob = hist / hist_sum
            prob = prob[prob > 0]
            entropy = -np.sum(prob * np.log2(prob))
            normalized_entropy = entropy / 8.0
            
            if normalized_entropy > 0.80:
                return 1.0
            elif normalized_entropy > 0.70:
                return 0.85
            elif normalized_entropy > 0.60:
                return 0.65
            elif normalized_entropy > 0.50:
                return 0.45
            else:
                return 0.30
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