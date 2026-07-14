import cv2
import numpy as np
from typing import Dict, List, Optional
from app.config import settings
from app.utils.logger import setup_logger

logger = setup_logger(__name__)


class AntiSpoofingService:
    """
    Anti-Spoofing Service - Maximum Security Screen Attack Detection
    Detects presentation attacks: printed photos, screen replays
    (phone/tablet/monitor/laptop), and static images held up to the camera.

    SECURITY PHILOSOPHY: When in doubt, REJECT. Better to have false negatives
    than let a single screen attack through.

    Checks:
      1. Texture analysis (Laplacian variance)      -> blur/detail cues
      2. Color distribution analysis                -> screen saturation shift
      3. Screen-specific color artifacts             -> RGB clipping, color banding
      4. Edge analysis (Canny density)               -> print edge artifacts
      5. Noise pattern analysis                      -> sensor noise plausibility
      6. Moire pattern detection (FFT)               -> screen replay attacks (AGGRESSIVE)
      7. LBP texture analysis                        -> printed photo attacks
      8. Specular highlight / glare analysis         -> screen glare
      9. Motion liveness (optional, needs >=2 frames)-> static photo/video attacks
    """

    def __init__(self):
        logger.info(f"[AntiSpoofingService.__init__:32] Initializing MAXIMUM SECURITY AntiSpoofingService")
        self.initialized = False
        
        # AGGRESSIVE THRESHOLDS - No screen attack should pass
        self.spoofing_threshold = 0.65  # Increased from 0.50
        self.min_checks_required = 1    # Reduced from 2 - even 1 failure is suspicious
        self.min_individual_threshold = 0.40  # Increased from 0.35
        
        # Screen-specific thresholds - VERY AGGRESSIVE
        self.screen_moire_threshold = 0.40  # Anything below this is suspicious
        self.screen_color_threshold = 0.50  # More sensitive to screen colors
        self.strong_moire_veto = 0.20  # If moire is this bad, instant reject
        self.low_confidence_threshold = 0.75  # Higher bar for acceptance
        
        logger.info(f"[AntiSpoofingService.__init__:44] AGGRESSIVE Thresholds set:")
        logger.info(f"[AntiSpoofingService.__init__:45]   spoofing: {self.spoofing_threshold}")
        logger.info(f"[AntiSpoofingService.__init__:46]   min_checks: {self.min_checks_required}")
        logger.info(f"[AntiSpoofingService.__init__:47]   min_individual: {self.min_individual_threshold}")
        logger.info(f"[AntiSpoofingService.__init__:48]   screen_moire: {self.screen_moire_threshold}")
        logger.info(f"[AntiSpoofingService.__init__:49]   screen_color: {self.screen_color_threshold}")
        logger.info(f"[AntiSpoofingService.__init__:50]   strong_moire_veto: {self.strong_moire_veto}")

    async def initialize(self):
        """Initialize anti-spoofing service"""
        logger.info(f"[AntiSpoofingService.initialize:54] Starting initialization")
        self.initialized = True
        logger.info("Anti-spoofing service initialized - MAXIMUM SECURITY MODE")
        logger.info(f"[AntiSpoofingService.initialize:57] Initialization complete, initialized={self.initialized}")

    async def detect_spoofing(
        self,
        image_data: np.ndarray,
        motion_frames: Optional[List[np.ndarray]] = None
    ) -> Dict:
        """
        Detect if face is real or spoofed - AGGRESSIVE SCREEN DETECTION.
        """
        logger.info(f"[AntiSpoofingService.detect_spoofing:67] ========== STARTING MAXIMUM SECURITY SPOOFING DETECTION ==========")
        logger.info(f"[AntiSpoofingService.detect_spoofing:68] Image shape: {image_data.shape}, dtype: {image_data.dtype}")
        logger.info(f"[AntiSpoofingService.detect_spoofing:69] Motion frames provided: {motion_frames is not None}, "
                    f"count: {len(motion_frames) if motion_frames else 0}")
        
        try:
            results = []
            screen_attack_indicators = 0
            critical_failures = 0
            
            # Check 1: Texture Analysis
            logger.info(f"[AntiSpoofingService.detect_spoofing:78] Running TEXTURE ANALYSIS (Check 1/9)...")
            texture_score = self._analyze_texture(image_data)
            results.append(self._make_result('texture_analysis', texture_score))
            logger.info(f"[AntiSpoofingService.detect_spoofing:81] ✓ Texture analysis - score: {texture_score:.4f}")

            # Check 2: Color Distribution Analysis
            logger.info(f"[AntiSpoofingService.detect_spoofing:84] Running COLOR DISTRIBUTION ANALYSIS (Check 2/9)...")
            color_score = self._analyze_color_distribution(image_data)
            results.append(self._make_result('color_analysis', color_score))
            logger.info(f"[AntiSpoofingService.detect_spoofing:87] ✓ Color analysis - score: {color_score:.4f}")

            # Check 3: Screen-Specific Color Artifacts (CRITICAL)
            logger.info(f"[AntiSpoofingService.detect_spoofing:90] Running SCREEN COLOR ARTIFACTS ANALYSIS (Check 3/9)...")
            screen_color_score = self._analyze_screen_color_artifacts(image_data)
            results.append(self._make_result('screen_color_artifacts', screen_color_score))
            logger.info(f"[AntiSpoofingService.detect_spoofing:93] ✓ Screen color analysis - score: {screen_color_score:.4f}")
            
            if screen_color_score <= 0.40:  # Very suspicious
                screen_attack_indicators += 2  # Weight this more heavily
                critical_failures += 1
                logger.warning(f"[AntiSpoofingService.detect_spoofing:98] ⚠ CRITICAL: Screen color artifacts detected!")
            elif screen_color_score <= self.screen_color_threshold:
                screen_attack_indicators += 1
                logger.warning(f"[AntiSpoofingService.detect_spoofing:101] ⚠ Screen color artifacts suspected!")

            # Check 4: Edge Analysis
            logger.info(f"[AntiSpoofingService.detect_spoofing:104] Running EDGE ANALYSIS (Check 4/9)...")
            edge_score = self._analyze_edges(image_data)
            results.append(self._make_result('edge_analysis', edge_score))
            logger.info(f"[AntiSpoofingService.detect_spoofing:107] ✓ Edge analysis - score: {edge_score:.4f}")

            # Check 5: Noise Pattern Analysis
            logger.info(f"[AntiSpoofingService.detect_spoofing:110] Running NOISE PATTERN ANALYSIS (Check 5/9)...")
            noise_score = self._analyze_noise_pattern(image_data)
            results.append(self._make_result('noise_analysis', noise_score, threshold=0.40))
            logger.info(f"[AntiSpoofingService.detect_spoofing:113] ✓ Noise analysis - score: {noise_score:.4f}")

            # Check 6: Moire Pattern Detection (MOST CRITICAL FOR SCREENS)
            logger.info(f"[AntiSpoofingService.detect_spoofing:116] Running AGGRESSIVE MOIRE PATTERN DETECTION (Check 6/9)...")
            moire_score = self._detect_moire_pattern_aggressive(image_data)
            results.append(self._make_result('moire_pattern', moire_score, threshold=self.screen_moire_threshold))
            logger.info(f"[AntiSpoofingService.detect_spoofing:119] ✓ Moire pattern - score: {moire_score:.4f}")
            
            # MULTI-LEVEL MOIRE DETECTION
            if moire_score <= 0.10:  # Definitely a screen
                screen_attack_indicators += 3
                critical_failures += 2
                logger.warning(f"[AntiSpoofingService.detect_spoofing:125] ⚠ CRITICAL: Definite screen moire pattern!")
            elif moire_score <= 0.20:  # Very likely a screen
                screen_attack_indicators += 2
                critical_failures += 1
                logger.warning(f"[AntiSpoofingService.detect_spoofing:129] ⚠ STRONG screen moire pattern detected!")
            elif moire_score <= self.screen_moire_threshold:  # Suspicious
                screen_attack_indicators += 1
                logger.warning(f"[AntiSpoofingService.detect_spoofing:132] ⚠ Screen moire pattern suspected!")

            # Check 7: LBP Texture Analysis
            logger.info(f"[AntiSpoofingService.detect_spoofing:135] Running LBP TEXTURE ANALYSIS (Check 7/9)...")
            lbp_score = self._analyze_texture_lbp(image_data)
            results.append(self._make_result('lbp_texture', lbp_score))
            logger.info(f"[AntiSpoofingService.detect_spoofing:138] ✓ LBP texture - score: {lbp_score:.4f}")

            # Check 8: Specular Glare Analysis
            logger.info(f"[AntiSpoofingService.detect_spoofing:141] Running SPECULAR GLARE ANALYSIS (Check 8/9)...")
            glare_score = self._analyze_specular_highlights_aggressive(image_data)
            results.append(self._make_result('specular_glare', glare_score))
            logger.info(f"[AntiSpoofingService.detect_spoofing:144] ✓ Specular glare - score: {glare_score:.4f}")
            
            if glare_score <= 0.30:  # Significant glare - screen indicator
                screen_attack_indicators += 1
                logger.warning(f"[AntiSpoofingService.detect_spoofing:148] ⚠ Screen glare detected!")

            # Check 9: Motion Liveness (optional)
            liveness_checked = False
            if motion_frames and len(motion_frames) >= 2:
                logger.info(f"[AntiSpoofingService.detect_spoofing:153] Running MOTION LIVENESS CHECK (Check 9/9)...")
                liveness_score = self._analyze_motion_liveness(motion_frames)
                results.append(self._make_result('motion_liveness', liveness_score))
                liveness_checked = True
                logger.info(f"[AntiSpoofingService.detect_spoofing:157] ✓ Motion liveness - score: {liveness_score:.4f}")
                
                if liveness_score <= 0.20:  # Very little motion
                    screen_attack_indicators += 2
                    critical_failures += 1
                    logger.warning(f"[AntiSpoofingService.detect_spoofing:161] ⚠ Static/rigid motion - likely screen!")
            else:
                logger.info(f"[AntiSpoofingService.detect_spoofing:163] ⊘ Motion liveness SKIPPED")

            # Calculate results
            passed_count = sum(1 for r in results if r['passed'])
            total_checks = len(results)
            overall_score = sum(r['score'] for r in results) / total_checks

            logger.info(f"[AntiSpoofingService.detect_spoofing:170] ========== RESULTS SUMMARY ==========")
            logger.info(f"[AntiSpoofingService.detect_spoofing:171]   Total checks: {total_checks}")
            logger.info(f"[AntiSpoofingService.detect_spoofing:172]   Passed: {passed_count}")
            logger.info(f"[AntiSpoofingService.detect_spoofing:173]   Overall score: {overall_score:.4f}")
            logger.info(f"[AntiSpoofingService.detect_spoofing:174]   Screen attack indicators: {screen_attack_indicators}")
            logger.info(f"[AntiSpoofingService.detect_spoofing:175]   Critical failures: {critical_failures}")
            
            # Log individual scores
            for r in results:
                status = "✓ PASS" if r['passed'] else "✗ FAIL"
                logger.info(f"[AntiSpoofingService.detect_spoofing:179]   {r['method']}: {r['score']:.4f} {status}")

            # ===================================================================
            # AGGRESSIVE VETO LOGIC - MULTIPLE LAYERS OF PROTECTION
            # ===================================================================
            
            is_real = True  # Start assuming real, then prove otherwise
            
            # VETO LAYER 1: Any critical failure = immediate rejection
            if critical_failures >= 1:
                logger.warning(f"[AntiSpoofingService.detect_spoofing:189] ⚔ VETO LAYER 1: CRITICAL FAILURE DETECTED "
                             f"({critical_failures} critical failures) - FORCING SPOOF")
                is_real = False
            
            # VETO LAYER 2: Strong screen indicators
            if screen_attack_indicators >= 3:
                logger.warning(f"[AntiSpoofingService.detect_spoofing:194] ⚔ VETO LAYER 2: MULTIPLE SCREEN INDICATORS "
                             f"({screen_attack_indicators} indicators) - FORCING SPOOF")
                is_real = False
            
            # VETO LAYER 3: Any screen indicator + low confidence
            if screen_attack_indicators >= 1 and overall_score < self.low_confidence_threshold:
                logger.warning(f"[AntiSpoofingService.detect_spoofing:199] ⚔ VETO LAYER 3: SCREEN INDICATOR WITH LOW CONFIDENCE "
                             f"(score: {overall_score:.4f}, indicators: {screen_attack_indicators}) - FORCING SPOOF")
                is_real = False
            
            # VETO LAYER 4: Strong moire alone (most common screen signature)
            moire_result = next(r for r in results if r['method'] == 'moire_pattern')
            if moire_result['score'] <= self.strong_moire_veto:
                logger.warning(f"[AntiSpoofingService.detect_spoofing:205] ⚔ VETO LAYER 4: STRONG MOIRE PATTERN "
                             f"(score: {moire_result['score']:.4f}) - FORCING SPOOF")
                is_real = False
            
            # VETO LAYER 5: Moire + any other issue
            if moire_result['score'] <= 0.30:  # Moderate moire
                # Check if any other check also failed
                other_failures = [r for r in results if r['method'] != 'moire_pattern' and not r['passed']]
                if len(other_failures) >= 1:
                    logger.warning(f"[AntiSpoofingService.detect_spoofing:212] ⚔ VETO LAYER 5: MOIRE + OTHER FAILURES "
                                 f"(moire: {moire_result['score']:.4f}, other failures: {len(other_failures)}) - FORCING SPOOF")
                    is_real = False
            
            # VETO LAYER 6: Low overall score
            if overall_score < 0.60:
                logger.warning(f"[AntiSpoofingService.detect_spoofing:217] ⚔ VETO LAYER 6: LOW OVERALL CONFIDENCE "
                             f"(score: {overall_score:.4f}) - FORCING SPOOF")
                is_real = False
            
            # VETO LAYER 7: Multiple checks failed
            if passed_count < total_checks - 1:  # More than 1 failure
                logger.warning(f"[AntiSpoofingService.detect_spoofing:222] ⚔ VETO LAYER 7: MULTIPLE CHECK FAILURES "
                             f"(passed: {passed_count}/{total_checks}) - FORCING SPOOF")
                is_real = False

            # Only if ALL veto layers pass, check the standard criteria
            if is_real:  # Still considered real after all vetos
                majority_needed = max(self.min_checks_required, (total_checks // 2) + 1)
                
                # Standard check
                is_real = (
                    (overall_score >= self.spoofing_threshold and passed_count >= majority_needed)
                    or overall_score >= 0.80  # Increased from 0.75
                )
                
                logger.info(f"[AntiSpoofingService.detect_spoofing:234] STANDARD CHECK:")
                logger.info(f"[AntiSpoofingService.detect_spoofing:235]   Overall score >= threshold ({self.spoofing_threshold}): {overall_score >= self.spoofing_threshold}")
                logger.info(f"[AntiSpoofingService.detect_spoofing:236]   Passed >= majority ({majority_needed}): {passed_count >= majority_needed}")
                logger.info(f"[AntiSpoofingService.detect_spoofing:237]   Overall >= 0.80: {overall_score >= 0.80}")

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
                    'verdict': 'REAL' if is_real else 'SPOOF'
                }
            }

            logger.info(f"[AntiSpoofingService.detect_spoofing:254] ========== FINAL VERDICT: {result['details']['verdict']} ==========")
            logger.info(f"[AntiSpoofingService.detect_spoofing:255] Score: {result['confidence']:.4f}, "
                       f"Passed: {passed_count}/{total_checks}, "
                       f"Screen indicators: {screen_attack_indicators}, "
                       f"Critical failures: {critical_failures}")
            logger.info(f"[AntiSpoofingService.detect_spoofing:259] ========== SPOOFING DETECTION COMPLETE ==========")

            return result

        except Exception as e:
            logger.error(f"[AntiSpoofingService.detect_spoofing:263] ❌ Spoofing detection FAILED: {str(e)}", exc_info=True)
            logger.critical(f"[AntiSpoofingService.detect_spoofing:264] FAILING CLOSED - SECURITY FIRST")
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
        logger.info(f"[AntiSpoofingService._make_result:278] {method}: {score:.4f} (threshold: {t:.4f}) - {'PASS' if passed else 'FAIL'}")
        return result

    # ------------------------------------------------------------------
    # AGGRESSIVE SCREEN DETECTION CHECKS
    # ------------------------------------------------------------------

    def _analyze_screen_color_artifacts(self, image: np.ndarray) -> float:
        """
        AGGRESSIVE screen color artifact detection.
        Screens have distinctive color characteristics that real faces don't.
        """
        logger.info(f"[AntiSpoofingService._analyze_screen_color_artifacts:290] Aggressive screen color analysis...")
        try:
            h, w = image.shape[:2]
            b, g, r = cv2.split(image)
            
            # 1. Check for color channel clipping (screens often clip)
            r_high_clip = np.sum(r > 240) / r.size
            g_high_clip = np.sum(g > 240) / g.size
            b_high_clip = np.sum(b > 240) / b.size
            
            high_clip_score = max(r_high_clip, g_high_clip, b_high_clip)
            
            # 2. Check for crushed blacks (screen limitation)
            r_low_clip = np.sum(r < 15) / r.size
            g_low_clip = np.sum(g < 15) / g.size
            b_low_clip = np.sum(b < 15) / b.size
            
            low_clip_score = max(r_low_clip, g_low_clip, b_low_clip)
            
            logger.info(f"[AntiSpoofingService._analyze_screen_color_artifacts:308] "
                        f"High clip: {high_clip_score:.4f}, Low clip: {low_clip_score:.4f}")

            # 3. Check for color banding
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
            
            gaps_r = count_banding_gaps(hist_r)
            gaps_g = count_banding_gaps(hist_g)
            gaps_b = count_banding_gaps(hist_b)
            total_gaps = gaps_r + gaps_g + gaps_b
            
            logger.info(f"[AntiSpoofingService._analyze_screen_color_artifacts:333] "
                        f"Color banding gaps: {total_gaps}")

            # 4. Check saturation uniformity (screens often have uniform saturation)
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            saturation = hsv[:, :, 1]
            sat_std = np.std(saturation)
            
            logger.info(f"[AntiSpoofingService._analyze_screen_color_artifacts:340] "
                        f"Saturation std: {sat_std:.2f}")

            # AGGRESSIVE SCORING
            screen_score = 0.0
            
            # High clipping = strong screen indicator
            if high_clip_score > 0.05:
                screen_score += 0.5
            elif high_clip_score > 0.02:
                screen_score += 0.3
            elif high_clip_score > 0.01:
                screen_score += 0.2
            
            # Color banding = definite screen artifact
            if total_gaps > 10:
                screen_score += 0.5
            elif total_gaps > 5:
                screen_score += 0.3
            elif total_gaps > 2:
                screen_score += 0.2
            
            # Unnatural saturation
            if sat_std < 10:  # Too uniform - likely screen
                screen_score += 0.3
            elif sat_std < 15:
                screen_score += 0.2
            
            final_score = 1.0 - min(screen_score, 1.0)
            
            logger.info(f"[AntiSpoofingService._analyze_screen_color_artifacts:363] "
                        f"Screen color score: {final_score:.4f} (raw indicators: {screen_score:.4f})")
            
            return final_score

        except Exception as e:
            logger.warning(f"[AntiSpoofingService._analyze_screen_color_artifacts:368] "
                          f"❌ Screen color analysis failed: {str(e)}")
            return 0.5

    def _detect_moire_pattern_aggressive(self, image: np.ndarray) -> float:
        """
        ULTRA-AGGRESSIVE moire pattern detection for screen recapture attacks.
        Uses multiple scales, frequency bands, and extremely sensitive thresholds.
        """
        logger.info(f"[AntiSpoofingService._detect_moire_pattern_aggressive:377] Ultra-aggressive moire detection...")
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            all_peak_ratios = []
            all_outlier_fractions = []
            all_high_freq_energies = []
            
            # Multi-scale analysis
            scales = [(128, 128), (256, 256), (512, 512)]
            
            for scale_size in scales:
                gray_scaled = cv2.resize(gray, scale_size)
                
                # Apply Hann window
                hann_window = np.outer(np.hanning(scale_size[0]), np.hanning(scale_size[1]))
                gray_windowed = gray_scaled * hann_window
                
                # FFT
                f = np.fft.fft2(gray_windowed.astype(np.float32))
                fshift = np.fft.fftshift(f)
                magnitude = np.log1p(np.abs(fshift))
                
                h, w = magnitude.shape
                cy, cx = h // 2, w // 2
                
                # Check multiple frequency bands
                freq_bands = [
                    ('very_low', 3, 8),
                    ('low', 5, 15),
                    ('mid_low', 8, 25),
                    ('mid', 12, 40),
                    ('mid_high', 15, 60),
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
                            
                            # Multiple outlier thresholds for sensitivity
                            for threshold_mult in [2.0, 2.5, 3.0]:
                                outlier_threshold = mean_energy + threshold_mult * std_energy
                                outlier_fraction = np.sum(band_energy > outlier_threshold) / len(band_energy)
                                all_outlier_fractions.append(outlier_fraction)
                        
                        all_high_freq_energies.append(mean_energy)
            
            # Get maximum values across all analyses
            max_peak_ratio = max(all_peak_ratios) if all_peak_ratios else 0
            max_outlier_fraction = max(all_outlier_fractions) if all_outlier_fractions else 0
            avg_high_freq_energy = np.mean(all_high_freq_energies) if all_high_freq_energies else 0
            
            logger.info(f"[AntiSpoofingService._detect_moire_pattern_aggressive:445] "
                        f"Max peak ratio: {max_peak_ratio:.4f}")
            logger.info(f"[AntiSpoofingService._detect_moire_pattern_aggressive:447] "
                        f"Max outlier fraction: {max_outlier_fraction:.6f}")
            logger.info(f"[AntiSpoofingService._detect_moire_pattern_aggressive:449] "
                        f"Avg high freq energy: {avg_high_freq_energy:.4f}")
            
            # ULTRA-AGGRESSIVE THRESHOLDS
            # Even slight moire patterns should trigger low scores
            if max_peak_ratio > 8.0 or max_outlier_fraction > 0.005:
                score = 0.05  # DEFINITE SCREEN - maximum penalty
            elif max_peak_ratio > 6.0 or max_outlier_fraction > 0.003:
                score = 0.10  # Very strong screen pattern
            elif max_peak_ratio > 4.0 or max_outlier_fraction > 0.001:
                score = 0.20  # Strong screen indication
            elif max_peak_ratio > 3.0:
                score = 0.35  # Moderate screen indication
            elif max_peak_ratio > 2.5:
                score = 0.50  # Weak screen indication
            elif max_peak_ratio > 2.0:
                score = 0.65  # Very weak - might be natural
            else:
                score = 0.85  # Probably natural
            
            # Additional penalty for high average energy (bright screens)
            if avg_high_freq_energy > 8.0:
                score *= 0.7  # Reduce score by 30%
            
            score = max(0.05, min(1.0, score))  # Clamp between 0.05 and 1.0
            
            logger.info(f"[AntiSpoofingService._detect_moire_pattern_aggressive:472] "
                        f"Final moire score: {score:.4f}")
            
            return score

        except Exception as e:
            logger.warning(f"[AntiSpoofingService._detect_moire_pattern_aggressive:477] "
                          f"❌ Moire detection failed: {str(e)}")
            return 0.5

    def _analyze_specular_highlights_aggressive(self, image: np.ndarray) -> float:
        """
        AGGRESSIVE specular highlight detection for screen glare.
        """
        logger.info(f"[AntiSpoofingService._analyze_specular_highlights_aggressive:485] Aggressive glare analysis...")
        try:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            v_channel = hsv[:, :, 2]
            
            # Multiple brightness thresholds for sensitivity
            thresholds = [250, 240, 230, 220]
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
                
                # Score for this threshold
                if bright_fraction > 0.06 or large_blob_fraction > 0.03:
                    scores.append(0.15)
                elif bright_fraction > 0.04 or large_blob_fraction > 0.02:
                    scores.append(0.30)
                elif bright_fraction > 0.02 or large_blob_fraction > 0.01:
                    scores.append(0.50)
                elif bright_fraction > 0.01:
                    scores.append(0.70)
                else:
                    scores.append(0.90)
            
            # Use the worst (lowest) score
            final_score = min(scores) if scores else 0.90
            
            logger.info(f"[AntiSpoofingService._analyze_specular_highlights_aggressive:520] "
                        f"Glare score: {final_score:.4f}")
            
            return final_score

        except Exception as e:
            logger.warning(f"[AntiSpoofingService._analyze_specular_highlights_aggressive:525] "
                          f"❌ Glare analysis failed: {str(e)}")
            return 0.5

    # ------------------------------------------------------------------
    # Standard checks (kept for completeness)
    # ------------------------------------------------------------------

    def _analyze_texture(self, image: np.ndarray) -> float:
        """Analyze texture for screen/replay artifacts"""
        logger.info(f"[AntiSpoofingService._analyze_texture:535] Texture analysis...")
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            logger.info(f"[AntiSpoofingService._analyze_texture:539]   Laplacian variance: {laplacian_var:.2f}")

            if laplacian_var > 300:
                score = 1.0
            elif laplacian_var > 150:
                score = 0.85
            elif laplacian_var > 80:
                score = 0.70
            elif laplacian_var > 40:
                score = 0.55
            elif laplacian_var > 20:
                score = 0.40
            else:
                score = 0.30
            
            logger.info(f"[AntiSpoofingService._analyze_texture:553]   Final texture score: {score:.4f}")
            return score

        except Exception as e:
            logger.warning(f"[AntiSpoofingService._analyze_texture:557] ❌ Texture analysis failed: {str(e)}")
            return 0.5

    def _analyze_color_distribution(self, image: np.ndarray) -> float:
        """Analyze color distribution for screen display artifacts"""
        logger.info(f"[AntiSpoofingService._analyze_color_distribution:563] Color analysis...")
        try:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            saturation = hsv[:, :, 1]
            saturation_std = np.std(saturation)
            logger.info(f"[AntiSpoofingService._analyze_color_distribution:568]   Saturation std: {saturation_std:.2f}")

            if saturation_std > 35:
                score = 1.0
            elif saturation_std > 25:
                score = 0.85
            elif saturation_std > 18:
                score = 0.70
            elif saturation_std > 12:
                score = 0.55
            elif saturation_std > 8:
                score = 0.40
            else:
                score = 0.35
            
            logger.info(f"[AntiSpoofingService._analyze_color_distribution:582]   Final color score: {score:.4f}")
            return score

        except Exception as e:
            logger.warning(f"[AntiSpoofingService._analyze_color_distribution:586] ❌ Color analysis failed: {str(e)}")
            return 0.5

    def _analyze_edges(self, image: np.ndarray) -> float:
        """Analyze edges for print artifacts"""
        logger.info(f"[AntiSpoofingService._analyze_edges:592] Edge analysis...")
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            logger.info(f"[AntiSpoofingService._analyze_edges:597]   Edge density: {edge_density:.4f}")

            if edge_density < 0.10:
                score = 1.0
            elif edge_density < 0.15:
                score = 0.85
            elif edge_density < 0.20:
                score = 0.70
            elif edge_density < 0.25:
                score = 0.55
            elif edge_density < 0.30:
                score = 0.40
            else:
                score = 0.30
            
            logger.info(f"[AntiSpoofingService._analyze_edges:611]   Final edge score: {score:.4f}")
            return score

        except Exception as e:
            logger.warning(f"[AntiSpoofingService._analyze_edges:615] ❌ Edge analysis failed: {str(e)}")
            return 0.5

    def _analyze_noise_pattern(self, image: np.ndarray) -> float:
        """Analyze noise pattern for digital artifacts"""
        logger.info(f"[AntiSpoofingService._analyze_noise_pattern:621] Noise analysis...")
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            noise = cv2.absdiff(gray, blurred)
            noise_std = np.std(noise)
            logger.info(f"[AntiSpoofingService._analyze_noise_pattern:627]   Noise std: {noise_std:.2f}")

            if 5 < noise_std < 40:
                score = 1.0
            elif 3 < noise_std < 50:
                score = 0.85
            elif 2 < noise_std < 60:
                score = 0.70
            elif 1 < noise_std < 70:
                score = 0.55
            else:
                score = 0.40
            
            logger.info(f"[AntiSpoofingService._analyze_noise_pattern:639]   Final noise score: {score:.4f}")
            return score

        except Exception as e:
            logger.warning(f"[AntiSpoofingService._analyze_noise_pattern:643] ❌ Noise analysis failed: {str(e)}")
            return 0.5

    def _analyze_texture_lbp(self, image: np.ndarray) -> float:
        """LBP texture analysis for printed photo attacks"""
        logger.info(f"[AntiSpoofingService._analyze_texture_lbp:649] LBP analysis...")
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
            logger.info(f"[AntiSpoofingService._analyze_texture_lbp:672]   LBP entropy: {entropy:.4f}, normalized: {normalized_entropy:.4f}")

            if normalized_entropy > 0.80:
                score = 1.0
            elif normalized_entropy > 0.70:
                score = 0.85
            elif normalized_entropy > 0.60:
                score = 0.65
            elif normalized_entropy > 0.50:
                score = 0.45
            else:
                score = 0.30
            
            logger.info(f"[AntiSpoofingService._analyze_texture_lbp:685]   Final LBP score: {score:.4f}")
            return score

        except Exception as e:
            logger.warning(f"[AntiSpoofingService._analyze_texture_lbp:689] ❌ LBP analysis failed: {str(e)}")
            return 0.5

    def _analyze_motion_liveness(self, frames: List[np.ndarray]) -> float:
        """Motion liveness check"""
        logger.info(f"[AntiSpoofingService._analyze_motion_liveness:695] Motion analysis with {len(frames)} frames...")
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
                flow_mean = np.mean(magnitude)
                flow_std = np.std(magnitude)
                flow_means.append(flow_mean)
                flow_stds.append(flow_std)
                logger.info(f"[AntiSpoofingService._analyze_motion_liveness:712]   Frame pair {i}-{i+1}: "
                           f"mean: {flow_mean:.4f}, std: {flow_std:.4f}")

            avg_motion = float(np.mean(flow_means))
            avg_variation = float(np.mean(flow_stds))
            logger.info(f"[AntiSpoofingService._analyze_motion_liveness:717]   Avg motion: {avg_motion:.4f}, "
                        f"avg variation: {avg_variation:.4f}")

            if avg_motion < 0.05:
                score = 0.10
                logger.info(f"[AntiSpoofingService._analyze_motion_liveness:721]   FROZEN MOTION")
            else:
                rigidity_ratio = avg_variation / max(avg_motion, 1e-6)
                logger.info(f"[AntiSpoofingService._analyze_motion_liveness:724]   Rigidity ratio: {rigidity_ratio:.4f}")
                
                if rigidity_ratio < 0.10:
                    score = 0.20
                elif rigidity_ratio < 0.20:
                    score = 0.45
                elif rigidity_ratio < 0.40:
                    score = 0.70
                else:
                    score = 0.90

            logger.info(f"[AntiSpoofingService._analyze_motion_liveness:735]   Final motion score: {score:.4f}")
            return score

        except Exception as e:
            logger.warning(f"[AntiSpoofingService._analyze_motion_liveness:739] ❌ Motion analysis failed: {str(e)}")
            return 0.5