import cv2
import numpy as np
from typing import Dict, List, Optional
from app.config import settings
from app.utils.logger import setup_logger

logger = setup_logger(__name__)


class AntiSpoofingService:
    """
    Anti-Spoofing Service - Smart Screen Blocking
    Blocks screen attacks immediately but allows real faces through.
    """

    def __init__(self):
        logger.info(f"[AntiSpoofingService.__init__:18] Initializing SMART BLOCK AntiSpoofingService")
        self.initialized = False
        
        self.spoofing_threshold = 0.60
        self.min_checks_required = 2
        self.min_individual_threshold = 0.40
        
        # Screen detection thresholds
        self.screen_moire_threshold = 0.40
        self.screen_color_threshold = 0.50
        self.strong_moire_veto = 0.15
        self.extreme_moire_veto = 0.05
        
        # SMART BLOCK: Only block if BOTH moire AND glare are bad
        self.screen_moire_block = 0.10   # Moire must be this bad
        self.screen_glare_block = 0.30   # AND glare must be this bad
        
        # Real face validation
        self.real_face_min_indicators = 3
        self.real_face_min_score = 0.68
        self.real_face_indicator_threshold = 0.85
        self.low_confidence_threshold = 0.65
        
        logger.info(f"[AntiSpoofingService.__init__:38] SMART BLOCK: moire<{self.screen_moire_block} AND glare<{self.screen_glare_block}")

    async def initialize(self):
        """Initialize anti-spoofing service"""
        logger.info(f"[AntiSpoofingService.initialize:42] Starting initialization")
        self.initialized = True
        logger.info("Anti-spoofing service initialized - SMART BLOCK MODE")

    async def detect_spoofing(
        self,
        image_data: np.ndarray,
        motion_frames: Optional[List[np.ndarray]] = None
    ) -> Dict:
        """
        Detect if face is real or spoofed - SMART BLOCKING.
        Only blocks immediately if BOTH moire AND glare indicate screen.
        """
        logger.info(f"[AntiSpoofingService.detect_spoofing:55] ========== STARTING SMART BLOCK DETECTION ==========")
        logger.info(f"[AntiSpoofingService.detect_spoofing:56] Image shape: {image_data.shape}")
        
        try:
            results = []
            screen_attack_indicators = 0
            critical_failures = 0
            real_face_indicators = 0
            
            # ================================================================
            # CHECK 1: MOIRE PATTERN - Run first for early detection
            # ================================================================
            logger.info(f"[AntiSpoofingService.detect_spoofing:68] Check 1/9: MOIRE PATTERN")
            moire_score = self._detect_moire_pattern_balanced(image_data)
            results.append(self._make_result('moire_pattern', moire_score, threshold=self.screen_moire_threshold))
            
            if moire_score <= self.extreme_moire_veto:
                screen_attack_indicators += 3
                critical_failures += 2
                logger.warning(f"[AntiSpoofingService.detect_spoofing:75] ⚠ CRITICAL: Definite screen moire! ({moire_score:.4f})")
            elif moire_score <= self.strong_moire_veto:
                screen_attack_indicators += 2
                critical_failures += 1
                logger.warning(f"[AntiSpoofingService.detect_spoofing:79] ⚠ STRONG screen moire! ({moire_score:.4f})")
            elif moire_score <= self.screen_moire_threshold:
                screen_attack_indicators += 1
                logger.warning(f"[AntiSpoofingService.detect_spoofing:82] ⚠ Screen moire suspected ({moire_score:.4f})")
            
            # ================================================================
            # CHECK 2: SPECULAR GLARE - Run second for combo check
            # ================================================================
            logger.info(f"[AntiSpoofingService.detect_spoofing:87] Check 2/9: SPECULAR GLARE")
            glare_score = self._analyze_specular_highlights_balanced(image_data)
            results.append(self._make_result('specular_glare', glare_score))
            
            if glare_score <= 0.25:
                screen_attack_indicators += 1
                logger.warning(f"[AntiSpoofingService.detect_spoofing:93] ⚠ Screen glare detected ({glare_score:.4f})")
            
            # ================================================================
            # SMART BLOCK CHECK: Only block if BOTH moire AND glare are bad
            # ================================================================
            bad_moire = moire_score <= self.screen_moire_block
            bad_glare = glare_score <= self.screen_glare_block
            
            logger.info(f"[AntiSpoofingService.detect_spoofing:101] SMART BLOCK CHECK:")
            logger.info(f"[AntiSpoofingService.detect_spoofing:102]   Bad moire (<{self.screen_moire_block}): {bad_moire} ({moire_score:.4f})")
            logger.info(f"[AntiSpoofingService.detect_spoofing:103]   Bad glare (<{self.screen_glare_block}): {bad_glare} ({glare_score:.4f})")
            
            if bad_moire and bad_glare:
                # BOTH indicators are bad - this is DEFINITELY a screen
                logger.warning(f"[AntiSpoofingService.detect_spoofing:107] ⛔ SMART BLOCK: Screen confirmed by moire+glare combo!")
                logger.info(f"[AntiSpoofingService.detect_spoofing:108] ========== SCREEN ATTACK BLOCKED ==========")
                
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
                        'block_reason': f'Screen confirmed: moire={moire_score:.4f}, glare={glare_score:.4f}'
                    }
                }
            
            if bad_moire and not bad_glare:
                logger.info(f"[AntiSpoofingService.detect_spoofing:126] Bad moire but good glare - likely real face with camera artifacts, continuing...")
            
            # ================================================================
            # NOT BLOCKED - Continue with remaining checks
            # ================================================================
            
            # Check 3: Texture Analysis
            logger.info(f"[AntiSpoofingService.detect_spoofing:132] Check 3/9: TEXTURE ANALYSIS")
            texture_score = self._analyze_texture(image_data)
            results.append(self._make_result('texture_analysis', texture_score))
            if texture_score >= self.real_face_indicator_threshold:
                real_face_indicators += 1
            
            # Check 4: Color Distribution
            logger.info(f"[AntiSpoofingService.detect_spoofing:139] Check 4/9: COLOR DISTRIBUTION")
            color_score = self._analyze_color_distribution(image_data)
            results.append(self._make_result('color_analysis', color_score))
            if color_score >= self.real_face_indicator_threshold:
                real_face_indicators += 1
            
            # Check 5: Screen Color Artifacts
            logger.info(f"[AntiSpoofingService.detect_spoofing:146] Check 5/9: SCREEN COLOR ARTIFACTS")
            screen_color_score = self._analyze_screen_color_artifacts(image_data)
            results.append(self._make_result('screen_color_artifacts', screen_color_score))
            
            if screen_color_score <= 0.35:
                screen_attack_indicators += 2
                critical_failures += 1
                logger.warning(f"[AntiSpoofingService.detect_spoofing:153] ⚠ CRITICAL: Strong screen color artifacts!")
            elif screen_color_score <= self.screen_color_threshold:
                screen_attack_indicators += 1
                logger.warning(f"[AntiSpoofingService.detect_spoofing:156] ⚠ Screen color artifacts suspected")
            
            # Check 6: Edge Analysis
            logger.info(f"[AntiSpoofingService.detect_spoofing:160] Check 6/9: EDGE ANALYSIS")
            edge_score = self._analyze_edges(image_data)
            results.append(self._make_result('edge_analysis', edge_score))
            if edge_score >= self.real_face_indicator_threshold:
                real_face_indicators += 1
            
            # Check 7: Noise Pattern
            logger.info(f"[AntiSpoofingService.detect_spoofing:167] Check 7/9: NOISE PATTERN")
            noise_score = self._analyze_noise_pattern(image_data)
            results.append(self._make_result('noise_analysis', noise_score, threshold=0.40))
            if noise_score >= self.real_face_indicator_threshold:
                real_face_indicators += 1
            
            # Check 8: LBP Texture
            logger.info(f"[AntiSpoofingService.detect_spoofing:174] Check 8/9: LBP TEXTURE")
            lbp_score = self._analyze_texture_lbp(image_data)
            results.append(self._make_result('lbp_texture', lbp_score))
            if lbp_score >= self.real_face_indicator_threshold:
                real_face_indicators += 1
            
            # Check 9: Motion Liveness (optional)
            liveness_checked = False
            if motion_frames and len(motion_frames) >= 2:
                logger.info(f"[AntiSpoofingService.detect_spoofing:183] Check 9/9: MOTION LIVENESS")
                liveness_score = self._analyze_motion_liveness(motion_frames)
                results.append(self._make_result('motion_liveness', liveness_score))
                liveness_checked = True
                if liveness_score <= 0.20:
                    screen_attack_indicators += 2
                    critical_failures += 1
                    logger.warning(f"[AntiSpoofingService.detect_spoofing:190] ⚠ Static/rigid motion detected")
            else:
                logger.info(f"[AntiSpoofingService.detect_spoofing:192] ⊘ Motion liveness SKIPPED")

            # Calculate results
            passed_count = sum(1 for r in results if r['passed'])
            total_checks = len(results)
            overall_score = sum(r['score'] for r in results) / total_checks

            logger.info(f"[AntiSpoofingService.detect_spoofing:199] ========== RESULTS SUMMARY ==========")
            logger.info(f"[AntiSpoofingService.detect_spoofing:200]   Total: {total_checks}, Passed: {passed_count}, Score: {overall_score:.4f}")
            logger.info(f"[AntiSpoofingService.detect_spoofing:201]   Screen indicators: {screen_attack_indicators}, Critical: {critical_failures}, Real: {real_face_indicators}")
            
            for r in results:
                status = "✓ PASS" if r['passed'] else "✗ FAIL"
                logger.info(f"[AntiSpoofingService.detect_spoofing:204]   {r['method']}: {r['score']:.4f} {status}")

            # ================================================================
            # FINAL DECISION
            # ================================================================
            is_strong_real_face = (
                real_face_indicators >= self.real_face_min_indicators and 
                overall_score >= self.real_face_min_score
            )
            
            logger.info(f"[AntiSpoofingService.detect_spoofing:215] DECISION: Strong real face: {is_strong_real_face}")
            
            is_real = True
            
            if is_strong_real_face:
                logger.info(f"[AntiSpoofingService.detect_spoofing:220] ✓ Real face ACCEPTED ({real_face_indicators} indicators, score: {overall_score:.4f})")
                is_real = True
            else:
                if critical_failures >= 1:
                    logger.warning(f"[AntiSpoofingService.detect_spoofing:224] ⚔ VETO: Critical failure ({critical_failures})")
                    is_real = False
                elif screen_attack_indicators >= 3:
                    logger.warning(f"[AntiSpoofingService.detect_spoofing:227] ⚔ VETO: Multiple screen indicators ({screen_attack_indicators})")
                    is_real = False
                elif screen_attack_indicators >= 1 and overall_score < self.low_confidence_threshold:
                    logger.warning(f"[AntiSpoofingService.detect_spoofing:230] ⚔ VETO: Screen indicator + low confidence")
                    is_real = False
                elif moire_score <= self.strong_moire_veto:
                    logger.warning(f"[AntiSpoofingService.detect_spoofing:233] ⚔ VETO: Strong moire ({moire_score:.4f})")
                    is_real = False
                elif overall_score < 0.55:
                    logger.warning(f"[AntiSpoofingService.detect_spoofing:236] ⚔ VETO: Very low confidence ({overall_score:.4f})")
                    is_real = False
                elif passed_count < total_checks - 1:
                    logger.warning(f"[AntiSpoofingService.detect_spoofing:239] ⚔ VETO: Multiple failures ({passed_count}/{total_checks})")
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

            logger.info(f"[AntiSpoofingService.detect_spoofing:265] ========== FINAL VERDICT: {result['details']['verdict']} ==========")
            return result

        except Exception as e:
            logger.error(f"[AntiSpoofingService.detect_spoofing:269] ❌ Detection FAILED: {str(e)}", exc_info=True)
            logger.critical(f"[AntiSpoofingService.detect_spoofing:270] FAILING CLOSED")
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
        logger.info(f"[AntiSpoofingService._make_result:284] {method}: {score:.4f} (threshold: {t:.4f}) - {'PASS' if passed else 'FAIL'}")
        return result

    # ------------------------------------------------------------------
    # DETECTION METHODS
    # ------------------------------------------------------------------

    def _detect_moire_pattern_balanced(self, image: np.ndarray) -> float:
        """Balanced moire detection."""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            all_peak_ratios = []
            all_outlier_fractions = []
            all_high_freq_energies = []
            
            for scale_size in [(256, 256), (512, 512)]:
                gray_scaled = cv2.resize(gray, scale_size)
                hann_window = np.outer(np.hanning(scale_size[0]), np.hanning(scale_size[1]))
                gray_windowed = gray_scaled * hann_window
                
                f = np.fft.fft2(gray_windowed.astype(np.float32))
                fshift = np.fft.fftshift(f)
                magnitude = np.log1p(np.abs(fshift))
                
                h, w = magnitude.shape
                cy, cx = h // 2, w // 2
                
                for r_min, r_max in [(5, 15), (10, 35), (20, min(h, w)//2)]:
                    y, x = np.ogrid[:h, :w]
                    dist = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
                    mask = (dist > r_min) & (dist <= r_max)
                    
                    energy = magnitude[mask]
                    if len(energy) > 0:
                        mean_e = np.mean(energy)
                        std_e = np.std(energy)
                        if std_e > 1e-6:
                            all_peak_ratios.append((np.max(energy) - mean_e) / std_e)
                            for tm in [2.5, 3.0]:
                                all_outlier_fractions.append(np.sum(energy > mean_e + tm * std_e) / len(energy))
                        all_high_freq_energies.append(mean_e)
            
            max_pr = max(all_peak_ratios) if all_peak_ratios else 0
            max_of = max(all_outlier_fractions) if all_outlier_fractions else 0
            avg_energy = np.mean(all_high_freq_energies) if all_high_freq_energies else 0
            
            if max_pr > 8.0 or max_of > 0.008: score = 0.05
            elif max_pr > 6.0 or max_of > 0.005: score = 0.12
            elif max_pr > 4.5 or max_of > 0.002: score = 0.25
            elif max_pr > 3.5: score = 0.45
            elif max_pr > 2.5: score = 0.65
            else: score = 0.85
            
            if avg_energy > 9.0: score *= 0.8
            return max(0.05, min(1.0, score))
        except Exception as e:
            logger.warning(f"Moire failed: {str(e)}")
            return 0.5

    def _analyze_specular_highlights_balanced(self, image: np.ndarray) -> float:
        """Balanced glare detection."""
        try:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            v_channel = hsv[:, :, 2]
            scores = []
            for thresh in [250, 240, 230]:
                bright_mask = (v_channel > thresh).astype(np.uint8) * 255
                bf = np.sum(bright_mask > 0) / bright_mask.size
                _, _, stats, _ = cv2.connectedComponentsWithStats(bright_mask, connectivity=8)
                lbf = max(stats[1:, cv2.CC_STAT_AREA]) / bright_mask.size if len(stats) > 1 and stats[1:, cv2.CC_STAT_AREA].size > 0 else 0
                
                if bf > 0.08 or lbf > 0.04: scores.append(0.20)
                elif bf > 0.05 or lbf > 0.02: scores.append(0.40)
                elif bf > 0.03 or lbf > 0.01: scores.append(0.60)
                else: scores.append(0.80)
            return min(scores) if scores else 0.80
        except Exception as e:
            logger.warning(f"Glare failed: {str(e)}")
            return 0.5

    def _analyze_screen_color_artifacts(self, image: np.ndarray) -> float:
        """Detect screen color artifacts."""
        try:
            b, g, r = cv2.split(image)
            high_clip = max(np.sum(r > 240) / r.size, np.sum(g > 240) / g.size, np.sum(b > 240) / b.size)
            
            def count_gaps(hist):
                gaps = cons = 0
                for v in hist:
                    if v[0] == 0: cons += 1
                    else:
                        if cons >= 3: gaps += 1
                        cons = 0
                return gaps + (1 if cons >= 3 else 0)
            
            total_gaps = sum(count_gaps(cv2.calcHist([image], [i], None, [256], [0, 256])) for i in range(3))
            sat_std = np.std(cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[:, :, 1])
            
            score = 0.0
            if high_clip > 0.08: score += 0.5
            elif high_clip > 0.05: score += 0.3
            elif high_clip > 0.02: score += 0.2
            if total_gaps > 12: score += 0.5
            elif total_gaps > 8: score += 0.3
            elif total_gaps > 4: score += 0.2
            if sat_std < 12: score += 0.3
            elif sat_std < 18: score += 0.2
            return 1.0 - min(score, 1.0)
        except Exception as e:
            logger.warning(f"Screen color failed: {str(e)}")
            return 0.5

    def _analyze_texture(self, image: np.ndarray) -> float:
        try:
            var = cv2.Laplacian(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
            return 1.0 if var > 300 else 0.85 if var > 150 else 0.70 if var > 80 else 0.55 if var > 40 else 0.40 if var > 20 else 0.30
        except: return 0.5

    def _analyze_color_distribution(self, image: np.ndarray) -> float:
        try:
            std = np.std(cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[:, :, 1])
            return 1.0 if std > 35 else 0.85 if std > 25 else 0.70 if std > 18 else 0.55 if std > 12 else 0.40 if std > 8 else 0.35
        except: return 0.5

    def _analyze_edges(self, image: np.ndarray) -> float:
        try:
            d = np.sum(cv2.Canny(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 50, 150) > 0) / (image.shape[0] * image.shape[1])
            return 1.0 if d < 0.10 else 0.85 if d < 0.15 else 0.70 if d < 0.20 else 0.55 if d < 0.25 else 0.40 if d < 0.30 else 0.30
        except: return 0.5

    def _analyze_noise_pattern(self, image: np.ndarray) -> float:
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            std = np.std(cv2.absdiff(gray, cv2.GaussianBlur(gray, (5, 5), 0)))
            return 1.0 if 5 < std < 40 else 0.85 if 3 < std < 50 else 0.70 if 2 < std < 60 else 0.55 if 1 < std < 70 else 0.40
        except: return 0.5

    def _analyze_texture_lbp(self, image: np.ndarray) -> float:
        try:
            gray = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (200, 200)).astype(np.int32)
            center = gray[1:-1, 1:-1]
            offsets = [(-1,-1), (-1,0), (-1,1), (0,1), (1,1), (1,0), (1,-1), (0,-1)]
            lbp = np.zeros_like(center, dtype=np.uint8)
            for i, (dy, dx) in enumerate(offsets):
                lbp |= ((gray[1+dy:1+dy+center.shape[0], 1+dx:1+dx+center.shape[1]] >= center).astype(np.uint8) << i)
            hist = np.histogram(lbp, bins=256, range=(0, 256))[0].astype(np.float64)
            if hist.sum() == 0: return 0.5
            prob = hist[hist > 0] / hist.sum()
            entropy = -np.sum(prob * np.log2(prob)) / 8.0
            return 1.0 if entropy > 0.80 else 0.85 if entropy > 0.70 else 0.65 if entropy > 0.60 else 0.45 if entropy > 0.50 else 0.30
        except: return 0.5

    def _analyze_motion_liveness(self, frames: List[np.ndarray]) -> float:
        try:
            grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
            means, stds = [], []
            for i in range(len(grays) - 1):
                flow = cv2.calcOpticalFlowFarneback(grays[i], grays[i+1], None, 0.5, 3, 15, 3, 5, 1.2, 0)
                mag = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
                means.append(np.mean(mag)); stds.append(np.std(mag))
            avg_m = np.mean(means)
            return 0.15 if avg_m < 0.05 else 0.30 if (np.mean(stds)/max(avg_m,1e-6)) < 0.15 else 0.55 if (np.mean(stds)/max(avg_m,1e-6)) < 0.30 else 0.75 if (np.mean(stds)/max(avg_m,1e-6)) < 0.50 else 0.90
        except: return 0.5