import cv2
import numpy as np
from typing import Dict, List, Optional
from app.config import settings
from app.utils.logger import setup_logger

logger = setup_logger(__name__)


class AntiSpoofingService:
    """
    Anti-Spoofing Service - Enhanced Screen Attack Detection
    Detects presentation attacks: printed photos, screen replays
    (phone/tablet/monitor/laptop), and static images held up to the camera.

    Checks:
      1. Texture analysis (Laplacian variance)      -> blur/detail cues
      2. Color distribution analysis                -> screen saturation shift
      3. Screen-specific color artifacts             -> RGB clipping, color banding
      4. Edge analysis (Canny density)               -> print edge artifacts
      5. Noise pattern analysis                      -> sensor noise plausibility
      6. Moire pattern detection (FFT)               -> screen replay attacks (TUNED)
      7. LBP texture analysis                        -> printed photo attacks
      8. Specular highlight / glare analysis         -> screen glare
      9. Motion liveness (optional, needs >=2 frames)-> static photo/video attacks
    """

    def __init__(self):
        logger.info(f"[AntiSpoofingService.__init__:32] Initializing Enhanced AntiSpoofingService")
        self.initialized = False
        # Hard-coded thresholds - TUNED for better screen detection
        self.spoofing_threshold = 0.50
        self.min_checks_required = 2
        self.min_individual_threshold = 0.35
        
        # Screen-specific thresholds
        self.screen_moire_threshold = 0.30  # Lower = more sensitive to screens
        self.screen_color_threshold = 0.40  # Lower = more sensitive to screen colors
        
        logger.info(f"[AntiSpoofingService.__init__:42] Thresholds set - spoofing: {self.spoofing_threshold}, "
                    f"min_checks: {self.min_checks_required}, min_individual: {self.min_individual_threshold}")
        logger.info(f"[AntiSpoofingService.__init__:44] Screen detection thresholds - "
                    f"moire: {self.screen_moire_threshold}, color: {self.screen_color_threshold}")

    async def initialize(self):
        """Initialize anti-spoofing service"""
        logger.info(f"[AntiSpoofingService.initialize:49] Starting initialization")
        self.initialized = True
        logger.info("Anti-spoofing service initialized")
        logger.info(f"[AntiSpoofingService.initialize:52] Initialization complete, initialized={self.initialized}")

    async def detect_spoofing(
        self,
        image_data: np.ndarray,
        motion_frames: Optional[List[np.ndarray]] = None
    ) -> Dict:
        """
        Detect if face is real or spoofed (Enhanced for screen detection).

        Args:
            image_data: single BGR frame (required)
            motion_frames: optional sequence of >=2 BGR frames for motion liveness

        Returns:
            Spoofing detection result
        """
        logger.info(f"[AntiSpoofingService.detect_spoofing:69] ========== STARTING ENHANCED SPOOFING DETECTION ==========")
        logger.info(f"[AntiSpoofingService.detect_spoofing:70] Image shape: {image_data.shape}, "
                    f"dtype: {image_data.dtype}")
        logger.info(f"[AntiSpoofingService.detect_spoofing:72] Motion frames provided: {motion_frames is not None}, "
                    f"count: {len(motion_frames) if motion_frames else 0}")
        
        try:
            results = []
            screen_attack_indicators = 0
            
            # Check 1: Texture Analysis
            logger.info(f"[AntiSpoofingService.detect_spoofing:79] Running TEXTURE ANALYSIS (Check 1/9)...")
            texture_score = self._analyze_texture(image_data)
            results.append(self._make_result('texture_analysis', texture_score))
            logger.info(f"[AntiSpoofingService.detect_spoofing:82] ✓ Texture analysis complete - score: {texture_score:.4f}")

            # Check 2: Color Distribution Analysis
            logger.info(f"[AntiSpoofingService.detect_spoofing:85] Running COLOR DISTRIBUTION ANALYSIS (Check 2/9)...")
            color_score = self._analyze_color_distribution(image_data)
            results.append(self._make_result('color_analysis', color_score))
            logger.info(f"[AntiSpoofingService.detect_spoofing:88] ✓ Color analysis complete - score: {color_score:.4f}")

            # Check 3: Screen-Specific Color Artifacts (NEW - Critical for screen detection)
            logger.info(f"[AntiSpoofingService.detect_spoofing:91] Running SCREEN COLOR ARTIFACTS ANALYSIS (Check 3/9)...")
            screen_color_score = self._analyze_screen_color_artifacts(image_data)
            results.append(self._make_result('screen_color_artifacts', screen_color_score))
            logger.info(f"[AntiSpoofingService.detect_spoofing:94] ✓ Screen color analysis complete - score: {screen_color_score:.4f}")
            
            if screen_color_score <= self.screen_color_threshold:
                screen_attack_indicators += 1
                logger.warning(f"[AntiSpoofingService.detect_spoofing:98] ⚠ Screen color artifacts detected!")

            # Check 4: Edge Analysis
            logger.info(f"[AntiSpoofingService.detect_spoofing:101] Running EDGE ANALYSIS (Check 4/9)...")
            edge_score = self._analyze_edges(image_data)
            results.append(self._make_result('edge_analysis', edge_score))
            logger.info(f"[AntiSpoofingService.detect_spoofing:104] ✓ Edge analysis complete - score: {edge_score:.4f}")

            # Check 5: Noise Pattern Analysis
            logger.info(f"[AntiSpoofingService.detect_spoofing:107] Running NOISE PATTERN ANALYSIS (Check 5/9)...")
            noise_score = self._analyze_noise_pattern(image_data)
            results.append(self._make_result('noise_analysis', noise_score, threshold=0.35))
            logger.info(f"[AntiSpoofingService.detect_spoofing:110] ✓ Noise analysis complete - score: {noise_score:.4f}")

            # Check 6: Moire Pattern Detection (TUNED for screens)
            logger.info(f"[AntiSpoofingService.detect_spoofing:113] Running ENHANCED MOIRE PATTERN DETECTION (Check 6/9)...")
            moire_score = self._detect_moire_pattern_enhanced(image_data)
            results.append(self._make_result('moire_pattern', moire_score, threshold=self.screen_moire_threshold))
            logger.info(f"[AntiSpoofingService.detect_spoofing:116] ✓ Moire pattern complete - score: {moire_score:.4f}")
            
            if moire_score <= self.screen_moire_threshold:
                screen_attack_indicators += 1
                logger.warning(f"[AntiSpoofingService.detect_spoofing:120] ⚠ Screen moire pattern detected!")

            # Check 7: LBP Texture Analysis
            logger.info(f"[AntiSpoofingService.detect_spoofing:123] Running LBP TEXTURE ANALYSIS (Check 7/9)...")
            lbp_score = self._analyze_texture_lbp(image_data)
            results.append(self._make_result('lbp_texture', lbp_score))
            logger.info(f"[AntiSpoofingService.detect_spoofing:126] ✓ LBP texture complete - score: {lbp_score:.4f}")

            # Check 8: Specular Glare Analysis
            logger.info(f"[AntiSpoofingService.detect_spoofing:129] Running SPECULAR GLARE ANALYSIS (Check 8/9)...")
            glare_score = self._analyze_specular_highlights(image_data)
            results.append(self._make_result('specular_glare', glare_score))
            logger.info(f"[AntiSpoofingService.detect_spoofing:132] ✓ Specular glare complete - score: {glare_score:.4f}")

            # Check 9: Motion Liveness (optional)
            liveness_checked = False
            if motion_frames and len(motion_frames) >= 2:
                logger.info(f"[AntiSpoofingService.detect_spoofing:137] Running MOTION LIVENESS CHECK (Check 9/9)...")
                liveness_score = self._analyze_motion_liveness(motion_frames)
                results.append(self._make_result('motion_liveness', liveness_score))
                liveness_checked = True
                logger.info(f"[AntiSpoofingService.detect_spoofing:141] ✓ Motion liveness complete - score: {liveness_score:.4f}")
            else:
                logger.info(f"[AntiSpoofingService.detect_spoofing:143] ⊘ Motion liveness SKIPPED - insufficient frames")

            # Calculate results
            passed_count = sum(1 for r in results if r['passed'])
            total_checks = len(results)
            overall_score = sum(r['score'] for r in results) / total_checks

            logger.info(f"[AntiSpoofingService.detect_spoofing:150] RESULTS SUMMARY:")
            logger.info(f"[AntiSpoofingService.detect_spoofing:151]   Total checks: {total_checks}")
            logger.info(f"[AntiSpoofingService.detect_spoofing:152]   Passed: {passed_count}")
            logger.info(f"[AntiSpoofingService.detect_spoofing:153]   Overall score: {overall_score:.4f}")
            logger.info(f"[AntiSpoofingService.detect_spoofing:154]   Screen attack indicators: {screen_attack_indicators}")
            
            # Log individual scores
            for r in results:
                status = "✓ PASS" if r['passed'] else "✗ FAIL"
                logger.info(f"[AntiSpoofingService.detect_spoofing:158]   {r['method']}: {r['score']:.4f} {status}")

            # Majority calculation
            majority_needed = max(self.min_checks_required, (total_checks // 2) + 1)
            logger.info(f"[AntiSpoofingService.detect_spoofing:162]   Majority needed: {majority_needed}")

            # Initial verdict
            is_real = (
                (overall_score >= self.spoofing_threshold and passed_count >= majority_needed)
                or overall_score >= 0.75
            )
            logger.info(f"[AntiSpoofingService.detect_spoofing:168] INITIAL VERDICT: {'REAL' if is_real else 'SPOOF'}")
            logger.info(f"[AntiSpoofingService.detect_spoofing:169]   Overall score >= threshold ({self.spoofing_threshold}): {overall_score >= self.spoofing_threshold}")
            logger.info(f"[AntiSpoofingService.detect_spoofing:170]   Passed >= majority ({majority_needed}): {passed_count >= majority_needed}")
            logger.info(f"[AntiSpoofingService.detect_spoofing:171]   Overall >= 0.75: {overall_score >= 0.75}")

            # ENHANCED SCREEN DETECTION VETO LOGIC
            
            # Veto 1: Multiple screen indicators
            if screen_attack_indicators >= 2:
                logger.warning(f"[AntiSpoofingService.detect_spoofing:177] ⚠ MULTIPLE SCREEN INDICATORS VETO: "
                             f"{screen_attack_indicators} screen indicators found, forcing SPOOF verdict")
                is_real = False
            
            # Veto 2: Strong moire + color artifacts (screen fingerprint)
            moire_result = next(r for r in results if r['method'] == 'moire_pattern')
            screen_color_result = next(r for r in results if r['method'] == 'screen_color_artifacts')
            
            logger.info(f"[AntiSpoofingService.detect_spoofing:185] SCREEN FINGERPRINT CHECK:")
            logger.info(f"[AntiSpoofingService.detect_spoofing:186]   Moire score: {moire_result['score']:.4f}")
            logger.info(f"[AntiSpoofingService.detect_spoofing:187]   Screen color score: {screen_color_result['score']:.4f}")
            
            if moire_result['score'] <= self.screen_moire_threshold and screen_color_result['score'] <= self.screen_color_threshold:
                logger.warning(f"[AntiSpoofingService.detect_spoofing:190] ⚠ SCREEN FINGERPRINT VETO: "
                             f"Both moire and screen color artifacts detected, forcing SPOOF verdict")
                is_real = False
            
            # Veto 3: Strong moire + glare (classic screen replay)
            glare_result = next(r for r in results if r['method'] == 'specular_glare')
            logger.info(f"[AntiSpoofingService.detect_spoofing:195] MOIRE+GLARE VETO CHECK:")
            logger.info(f"[AntiSpoofingService.detect_spoofing:196]   Moire score: {moire_result['score']:.4f}")
            logger.info(f"[AntiSpoofingService.detect_spoofing:197]   Glare score: {glare_result['score']:.4f}")
            
            if moire_result['score'] <= 0.30 and glare_result['score'] <= 0.50:
                logger.warning(f"[AntiSpoofingService.detect_spoofing:200] ⚠ MOIRE+GLARE VETO TRIGGERED: "
                             f"Screen replay pattern detected, forcing SPOOF verdict")
                is_real = False
            
            # Veto 4: Motion liveness (if available)
            if liveness_checked:
                liveness_result = next(r for r in results if r['method'] == 'motion_liveness')
                logger.info(f"[AntiSpoofingService.detect_spoofing:206] MOTION VETO CHECK:")
                logger.info(f"[AntiSpoofingService.detect_spoofing:207]   Liveness score: {liveness_result['score']:.4f}")
                
                if liveness_result['score'] <= 0.15:
                    logger.warning(f"[AntiSpoofingService.detect_spoofing:210] ⚠ MOTION VETO TRIGGERED: "
                                 f"Static/rigid motion detected, forcing SPOOF verdict")
                    is_real = False

            # Veto 5: High confidence screen detection (overall pattern)
            if screen_attack_indicators >= 1 and overall_score < 0.60:
                logger.warning(f"[AntiSpoofingService.detect_spoofing:216] ⚠ SCREEN PATTERN VETO: "
                             f"Screen indicators with low overall score, forcing SPOOF verdict")
                is_real = False

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
                    'verdict': 'REAL' if is_real else 'SPOOF'
                }
            }

            logger.info(f"[AntiSpoofingService.detect_spoofing:234] FINAL VERDICT: {result['details']['verdict']} "
                       f"(score: {result['confidence']:.4f}, passed: {passed_count}/{total_checks}, "
                       f"screen indicators: {screen_attack_indicators})")
            logger.info(f"[AntiSpoofingService.detect_spoofing:237] ========== SPOOFING DETECTION COMPLETE ==========")

            return result

        except Exception as e:
            logger.error(f"[AntiSpoofingService.detect_spoofing:241] ❌ Spoofing detection FAILED: {str(e)}", exc_info=True)
            logger.critical(f"[AntiSpoofingService.detect_spoofing:242] FAILING CLOSED - Returning REJECTED_ON_ERROR for security")
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
        logger.info(f"[AntiSpoofingService._make_result:256] Method: {method}, score: {score:.4f}, "
                    f"threshold: {t:.4f}, passed: {passed}")
        return result

    # ------------------------------------------------------------------
    # ENHANCED SCREEN DETECTION CHECKS
    # ------------------------------------------------------------------

    def _analyze_screen_color_artifacts(self, image: np.ndarray) -> float:
        """
        Detect screen-specific color artifacts:
        - Color channel clipping (screens often have clipped whites/blacks)
        - Color banding (limited bit depth of displays)
        - Unnatural color distributions
        - Moire-like color patterns
        """
        logger.info(f"[AntiSpoofingService._analyze_screen_color_artifacts:270] Screen color artifacts analysis...")
        try:
            h, w = image.shape[:2]
            
            # 1. Check for color channel clipping
            b, g, r = cv2.split(image)
            
            # High-end clipping (saturated whites)
            r_high_clip = np.sum(r > 245) / r.size
            g_high_clip = np.sum(g > 245) / g.size
            b_high_clip = np.sum(b > 245) / b.size
            
            # Low-end clipping (crushed blacks)
            r_low_clip = np.sum(r < 10) / r.size
            g_low_clip = np.sum(g < 10) / g.size
            b_low_clip = np.sum(b < 10) / b.size
            
            high_clip_score = max(r_high_clip, g_high_clip, b_high_clip)
            low_clip_score = max(r_low_clip, g_low_clip, b_low_clip)
            
            logger.info(f"[AntiSpoofingService._analyze_screen_color_artifacts:289] "
                        f"High clip: R:{r_high_clip:.4f} G:{g_high_clip:.4f} B:{b_high_clip:.4f} Max:{high_clip_score:.4f}")
            logger.info(f"[AntiSpoofingService._analyze_screen_color_artifacts:291] "
                        f"Low clip: R:{r_low_clip:.4f} G:{g_low_clip:.4f} B:{b_low_clip:.4f} Max:{low_clip_score:.4f}")

            # 2. Check for color banding (histogram gaps)
            hist_r = cv2.calcHist([image], [2], None, [256], [0, 256])
            hist_g = cv2.calcHist([image], [1], None, [256], [0, 256])
            hist_b = cv2.calcHist([image], [0], None, [256], [0, 256])
            
            # Count consecutive zero bins (banding indicator)
            def count_banding_gaps(hist):
                gaps = 0
                consecutive_zeros = 0
                for i in range(len(hist)):
                    if hist[i][0] == 0:
                        consecutive_zeros += 1
                    else:
                        if consecutive_zeros >= 3:  # Gap of 3+ consecutive zero bins
                            gaps += 1
                        consecutive_zeros = 0
                if consecutive_zeros >= 3:
                    gaps += 1
                return gaps
            
            gaps_r = count_banding_gaps(hist_r)
            gaps_g = count_banding_gaps(hist_g)
            gaps_b = count_banding_gaps(hist_b)
            
            total_gaps = gaps_r + gaps_g + gaps_b
            
            logger.info(f"[AntiSpoofingService._analyze_screen_color_artifacts:318] "
                        f"Color banding gaps - R:{gaps_r} G:{gaps_g} B:{gaps_b} Total:{total_gaps}")

            # 3. Check for unnatural color saturation patterns
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            saturation = hsv[:, :, 1]
            
            # Screens often have either very uniform or strangely bimodal saturation
            sat_hist = cv2.calcHist([saturation], [0], None, [256], [0, 256])
            sat_hist_normalized = sat_hist / sat_hist.sum()
            
            # Calculate saturation distribution entropy
            sat_hist_clean = sat_hist_normalized[sat_hist_normalized > 0]
            if len(sat_hist_clean) > 0:
                sat_entropy = -np.sum(sat_hist_clean * np.log2(sat_hist_clean))
                sat_entropy_normalized = sat_entropy / 8.0  # 8 = log2(256)
            else:
                sat_entropy_normalized = 0
            
            logger.info(f"[AntiSpoofingService._analyze_screen_color_artifacts:334] "
                        f"Saturation entropy: {sat_entropy_normalized:.4f}")

            # 4. Combined screen color score
            screen_score = 0.0
            
            # High clipping is very suspicious for screens
            if high_clip_score > 0.08:
                screen_score += 0.4
            elif high_clip_score > 0.05:
                screen_score += 0.3
            elif high_clip_score > 0.02:
                screen_score += 0.2
            elif high_clip_score > 0.01:
                screen_score += 0.1
            
            # Color banding is a strong screen indicator
            if total_gaps > 15:
                screen_score += 0.4
            elif total_gaps > 10:
                screen_score += 0.3
            elif total_gaps > 5:
                screen_score += 0.2
            elif total_gaps > 2:
                screen_score += 0.1
            
            # Unnatural saturation patterns
            if sat_entropy_normalized < 0.5:
                screen_score += 0.2
            elif sat_entropy_normalized < 0.6:
                screen_score += 0.1
            
            # Convert to final score (1.0 = real, 0.0 = definite screen)
            final_score = 1.0 - min(screen_score, 1.0)
            
            logger.info(f"[AntiSpoofingService._analyze_screen_color_artifacts:359] "
                        f"Screen color score: {final_score:.4f} (raw screen indicators: {screen_score:.4f})")
            
            return final_score

        except Exception as e:
            logger.warning(f"[AntiSpoofingService._analyze_screen_color_artifacts:364] "
                          f"❌ Screen color analysis failed: {str(e)}")
            return 0.5

    def _detect_moire_pattern_enhanced(self, image: np.ndarray) -> float:
        """
        Enhanced moire pattern detection specifically tuned for screen recapture attacks.
        
        Screens create distinct FFT patterns due to:
        - Pixel grid interference
        - Refresh rate artifacts
        - Subpixel patterns (RGB stripe)
        """
        logger.info(f"[AntiSpoofingService._detect_moire_pattern_enhanced:376] Enhanced moire detection...")
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Multi-scale analysis for better detection
            scales = [(256, 256), (512, 512)]
            all_peak_ratios = []
            all_outlier_fractions = []
            
            for scale_size in scales:
                gray_scaled = cv2.resize(gray, scale_size)
                
                # Apply Hann window to reduce edge effects
                hann_window = np.outer(np.hanning(scale_size[0]), np.hanning(scale_size[1]))
                gray_windowed = gray_scaled * hann_window
                
                f = np.fft.fft2(gray_windowed.astype(np.float32))
                fshift = np.fft.fftshift(f)
                magnitude = np.log1p(np.abs(fshift))
                
                h, w = magnitude.shape
                cy, cx = h // 2, w // 2
                
                # Multiple frequency bands analysis
                freq_bands = [
                    ('low', 5, 15),
                    ('mid', 10, 40),
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
                            
                            outlier_threshold = mean_energy + 2.5 * std_energy
                            outlier_fraction = np.sum(band_energy > outlier_threshold) / len(band_energy)
                            all_outlier_fractions.append(outlier_fraction)
            
            # Use maximum peak ratio and outlier fraction across all scales/bands
            max_peak_ratio = max(all_peak_ratios) if all_peak_ratios else 0
            max_outlier_fraction = max(all_outlier_fractions) if all_outlier_fractions else 0
            
            logger.info(f"[AntiSpoofingService._detect_moire_pattern_enhanced:428] "
                        f"Max peak ratio: {max_peak_ratio:.4f}, Max outlier fraction: {max_outlier_fraction:.6f}")
            
            # TIGHTENED THRESHOLDS FOR BETTER SCREEN DETECTION
            if max_peak_ratio > 6.0 or max_outlier_fraction > 0.003:
                score = 0.10  # Definite screen pattern
            elif max_peak_ratio > 4.0 or max_outlier_fraction > 0.001:
                score = 0.25  # Strong screen indication
            elif max_peak_ratio > 3.0:
                score = 0.45  # Moderate screen indication
            elif max_peak_ratio > 2.0:
                score = 0.65  # Weak indication
            else:
                score = 0.90  # Probably not a screen
            
            logger.info(f"[AntiSpoofingService._detect_moire_pattern_enhanced:441] "
                        f"Enhanced moire score: {score:.4f}")
            
            return score

        except Exception as e:
            logger.warning(f"[AntiSpoofingService._detect_moire_pattern_enhanced:446] "
                          f"❌ Enhanced moire detection failed: {str(e)}")
            return 0.5

    # ------------------------------------------------------------------
    # Original checks (kept but slightly enhanced)
    # ------------------------------------------------------------------

    def _analyze_texture(self, image: np.ndarray) -> float:
        """Analyze texture for screen/replay artifacts"""
        logger.info(f"[AntiSpoofingService._analyze_texture:456] Texture analysis...")
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            logger.info(f"[AntiSpoofingService._analyze_texture:460]   Laplacian variance: {laplacian_var:.2f}")

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
            
            logger.info(f"[AntiSpoofingService._analyze_texture:474]   Final texture score: {score:.4f}")
            return score

        except Exception as e:
            logger.warning(f"[AntiSpoofingService._analyze_texture:478] ❌ Texture analysis failed: {str(e)}")
            return 0.5

    def _analyze_color_distribution(self, image: np.ndarray) -> float:
        """Analyze color distribution for screen display artifacts"""
        logger.info(f"[AntiSpoofingService._analyze_color_distribution:484] Color analysis...")
        try:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            saturation = hsv[:, :, 1]
            saturation_std = np.std(saturation)
            logger.info(f"[AntiSpoofingService._analyze_color_distribution:489]   Saturation std: {saturation_std:.2f}")

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
            
            logger.info(f"[AntiSpoofingService._analyze_color_distribution:503]   Final color score: {score:.4f}")
            return score

        except Exception as e:
            logger.warning(f"[AntiSpoofingService._analyze_color_distribution:507] ❌ Color analysis failed: {str(e)}")
            return 0.5

    def _analyze_edges(self, image: np.ndarray) -> float:
        """Analyze edges for print artifacts"""
        logger.info(f"[AntiSpoofingService._analyze_edges:513] Edge analysis...")
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            logger.info(f"[AntiSpoofingService._analyze_edges:518]   Edge density: {edge_density:.4f}")

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
            
            logger.info(f"[AntiSpoofingService._analyze_edges:532]   Final edge score: {score:.4f}")
            return score

        except Exception as e:
            logger.warning(f"[AntiSpoofingService._analyze_edges:536] ❌ Edge analysis failed: {str(e)}")
            return 0.5

    def _analyze_noise_pattern(self, image: np.ndarray) -> float:
        """Analyze noise pattern for digital artifacts"""
        logger.info(f"[AntiSpoofingService._analyze_noise_pattern:542] Noise analysis...")
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            noise = cv2.absdiff(gray, blurred)
            noise_std = np.std(noise)
            logger.info(f"[AntiSpoofingService._analyze_noise_pattern:548]   Noise std: {noise_std:.2f}")

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
            
            logger.info(f"[AntiSpoofingService._analyze_noise_pattern:560]   Final noise score: {score:.4f}")
            return score

        except Exception as e:
            logger.warning(f"[AntiSpoofingService._analyze_noise_pattern:564] ❌ Noise analysis failed: {str(e)}")
            return 0.5

    def _analyze_texture_lbp(self, image: np.ndarray) -> float:
        """LBP texture analysis for printed photo attacks"""
        logger.info(f"[AntiSpoofingService._analyze_texture_lbp:570] LBP analysis...")
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
            logger.info(f"[AntiSpoofingService._analyze_texture_lbp:593]   LBP entropy: {entropy:.4f}, normalized: {normalized_entropy:.4f}")

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
            
            logger.info(f"[AntiSpoofingService._analyze_texture_lbp:606]   Final LBP score: {score:.4f}")
            return score

        except Exception as e:
            logger.warning(f"[AntiSpoofingService._analyze_texture_lbp:610] ❌ LBP analysis failed: {str(e)}")
            return 0.5

    def _analyze_specular_highlights(self, image: np.ndarray) -> float:
        """Specular highlight / glare analysis for screen detection"""
        logger.info(f"[AntiSpoofingService._analyze_specular_highlights:616] Glare analysis...")
        try:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            v_channel = hsv[:, :, 2]

            bright_mask = (v_channel > 235).astype(np.uint8) * 255
            total_pixels = bright_mask.size
            bright_fraction = np.sum(bright_mask > 0) / total_pixels
            logger.info(f"[AntiSpoofingService._analyze_specular_highlights:624]   Bright fraction: {bright_fraction:.4f}")

            num_labels, _, stats, _ = cv2.connectedComponentsWithStats(bright_mask, connectivity=8)

            large_blob_fraction = 0.0
            if num_labels > 1:
                areas = stats[1:, cv2.CC_STAT_AREA]
                large_blob_fraction = areas.max() / total_pixels
                logger.info(f"[AntiSpoofingService._analyze_specular_highlights:632]   Blobs: {num_labels-1}, "
                           f"largest: {large_blob_fraction:.4f}")

            # Tightened thresholds for better screen glare detection
            if bright_fraction > 0.08 or large_blob_fraction > 0.04:
                score = 0.20
            elif bright_fraction > 0.05 or large_blob_fraction > 0.02:
                score = 0.45
            elif bright_fraction > 0.02 or large_blob_fraction > 0.01:
                score = 0.70
            else:
                score = 1.0
            
            logger.info(f"[AntiSpoofingService._analyze_specular_highlights:643]   Final glare score: {score:.4f}")
            return score

        except Exception as e:
            logger.warning(f"[AntiSpoofingService._analyze_specular_highlights:647] ❌ Glare analysis failed: {str(e)}")
            return 0.5

    def _analyze_motion_liveness(self, frames: List[np.ndarray]) -> float:
        """Motion liveness check"""
        logger.info(f"[AntiSpoofingService._analyze_motion_liveness:653] Motion analysis with {len(frames)} frames...")
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
                logger.info(f"[AntiSpoofingService._analyze_motion_liveness:670]   Frame pair {i}-{i+1}: "
                           f"mean: {flow_mean:.4f}, std: {flow_std:.4f}")

            avg_motion = float(np.mean(flow_means))
            avg_variation = float(np.mean(flow_stds))
            logger.info(f"[AntiSpoofingService._analyze_motion_liveness:675]   Avg motion: {avg_motion:.4f}, "
                        f"avg variation: {avg_variation:.4f}")

            if avg_motion < 0.05:
                score = 0.15
                logger.info(f"[AntiSpoofingService._analyze_motion_liveness:679]   FROZEN MOTION")
            else:
                rigidity_ratio = avg_variation / max(avg_motion, 1e-6)
                logger.info(f"[AntiSpoofingService._analyze_motion_liveness:682]   Rigidity ratio: {rigidity_ratio:.4f}")
                
                if rigidity_ratio < 0.15:
                    score = 0.30
                elif rigidity_ratio < 0.30:
                    score = 0.55
                elif rigidity_ratio < 0.50:
                    score = 0.75
                else:
                    score = 1.0

            logger.info(f"[AntiSpoofingService._analyze_motion_liveness:693]   Final motion score: {score:.4f}")
            return score

        except Exception as e:
            logger.warning(f"[AntiSpoofingService._analyze_motion_liveness:697] ❌ Motion analysis failed: {str(e)}")
            return 0.5