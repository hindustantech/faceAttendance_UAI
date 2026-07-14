import cv2
import numpy as np
from typing import Dict, List, Optional
from app.config import settings
from app.utils.logger import setup_logger

logger = setup_logger(__name__)


class AntiSpoofingService:
    """
    Anti-Spoofing Service
    Detects presentation attacks: printed photos, screen replays
    (phone/tablet/monitor), and static images held up to the camera.

    Checks:
      1. Texture analysis (Laplacian variance)      -> blur/detail cues
      2. Color distribution analysis                -> screen saturation shift
      3. Edge analysis (Canny density)               -> print edge artifacts
      4. Noise pattern analysis                      -> sensor noise plausibility
      5. Moire pattern detection (FFT)               -> screen replay attacks
      6. LBP texture analysis                        -> printed photo attacks
      7. Specular highlight / glare analysis         -> screen glare
      8. Motion liveness (optional, needs >=2 frames)-> static photo/video attacks
    """

    def __init__(self):
        logger.info(f"[AntiSpoofingService.__init__:32] Initializing AntiSpoofingService")
        self.initialized = False
        # Hard-coded thresholds (no settings dependency)
        self.spoofing_threshold = 0.50
        self.min_checks_required = 2
        self.min_individual_threshold = 0.35
        logger.info(f"[AntiSpoofingService.__init__:38] Thresholds set - spoofing: {self.spoofing_threshold}, "
                    f"min_checks: {self.min_checks_required}, min_individual: {self.min_individual_threshold}")

    async def initialize(self):
        """Initialize anti-spoofing service"""
        logger.info(f"[AntiSpoofingService.initialize:43] Starting initialization")
        self.initialized = True
        logger.info("Anti-spoofing service initialized")
        logger.info(f"[AntiSpoofingService.initialize:46] Initialization complete, initialized={self.initialized}")

    async def detect_spoofing(
        self,
        image_data: np.ndarray,
        motion_frames: Optional[List[np.ndarray]] = None
    ) -> Dict:
        """
        Detect if face is real or spoofed.

        Args:
            image_data: single BGR frame (required) used for all
                image-level checks (texture, color, edges, noise,
                moire, LBP, glare).
            motion_frames: optional sequence of >=2 BGR frames captured
                a few hundred ms apart, used for the motion liveness
                check. If omitted, liveness is skipped and the verdict
                relies on the 7 single-frame checks only -- callers
                doing high-security verification should supply frames.

        Returns:
            Spoofing detection result
        """
        logger.info(f"[AntiSpoofingService.detect_spoofing:67] ========== STARTING SPOOFING DETECTION ==========")
        logger.info(f"[AntiSpoofingService.detect_spoofing:68] Image shape: {image_data.shape}, "
                    f"dtype: {image_data.dtype}")
        logger.info(f"[AntiSpoofingService.detect_spoofing:70] Motion frames provided: {motion_frames is not None}, "
                    f"count: {len(motion_frames) if motion_frames else 0}")
        
        try:
            results = []

            # Check 1: Texture Analysis
            logger.info(f"[AntiSpoofingService.detect_spoofing:77] Running TEXTURE ANALYSIS (Check 1/7)...")
            texture_score = self._analyze_texture(image_data)
            results.append(self._make_result('texture_analysis', texture_score))
            logger.info(f"[AntiSpoofingService.detect_spoofing:80] ✓ Texture analysis complete - score: {texture_score:.4f}")

            # Check 2: Color Analysis
            logger.info(f"[AntiSpoofingService.detect_spoofing:83] Running COLOR DISTRIBUTION ANALYSIS (Check 2/7)...")
            color_score = self._analyze_color_distribution(image_data)
            results.append(self._make_result('color_analysis', color_score))
            logger.info(f"[AntiSpoofingService.detect_spoofing:86] ✓ Color analysis complete - score: {color_score:.4f}")

            # Check 3: Edge Analysis
            logger.info(f"[AntiSpoofingService.detect_spoofing:89] Running EDGE ANALYSIS (Check 3/7)...")
            edge_score = self._analyze_edges(image_data)
            results.append(self._make_result('edge_analysis', edge_score))
            logger.info(f"[AntiSpoofingService.detect_spoofing:92] ✓ Edge analysis complete - score: {edge_score:.4f}")

            # Check 4: Noise Analysis
            logger.info(f"[AntiSpoofingService.detect_spoofing:95] Running NOISE PATTERN ANALYSIS (Check 4/7)...")
            noise_score = self._analyze_noise_pattern(image_data)
            results.append(self._make_result('noise_analysis', noise_score, threshold=0.35))
            logger.info(f"[AntiSpoofingService.detect_spoofing:98] ✓ Noise analysis complete - score: {noise_score:.4f}")

            # Check 5: Moire Pattern
            logger.info(f"[AntiSpoofingService.detect_spoofing:101] Running MOIRE PATTERN DETECTION (Check 5/7)...")
            moire_score = self._detect_moire_pattern(image_data)
            results.append(self._make_result('moire_pattern', moire_score))
            logger.info(f"[AntiSpoofingService.detect_spoofing:104] ✓ Moire pattern complete - score: {moire_score:.4f}")

            # Check 6: LBP Texture
            logger.info(f"[AntiSpoofingService.detect_spoofing:107] Running LBP TEXTURE ANALYSIS (Check 6/7)...")
            lbp_score = self._analyze_texture_lbp(image_data)
            results.append(self._make_result('lbp_texture', lbp_score))
            logger.info(f"[AntiSpoofingService.detect_spoofing:110] ✓ LBP texture complete - score: {lbp_score:.4f}")

            # Check 7: Specular Glare
            logger.info(f"[AntiSpoofingService.detect_spoofing:113] Running SPECULAR GLARE ANALYSIS (Check 7/7)...")
            glare_score = self._analyze_specular_highlights(image_data)
            results.append(self._make_result('specular_glare', glare_score))
            logger.info(f"[AntiSpoofingService.detect_spoofing:116] ✓ Specular glare complete - score: {glare_score:.4f}")

            # Check 8: Motion Liveness (optional)
            liveness_checked = False
            if motion_frames and len(motion_frames) >= 2:
                logger.info(f"[AntiSpoofingService.detect_spoofing:121] Running MOTION LIVENESS CHECK (Check 8/8)...")
                liveness_score = self._analyze_motion_liveness(motion_frames)
                results.append(self._make_result('motion_liveness', liveness_score))
                liveness_checked = True
                logger.info(f"[AntiSpoofingService.detect_spoofing:125] ✓ Motion liveness complete - score: {liveness_score:.4f}")
            else:
                logger.info(f"[AntiSpoofingService.detect_spoofing:127] ⊘ Motion liveness SKIPPED - insufficient frames")

            # Calculate results
            passed_count = sum(1 for r in results if r['passed'])
            total_checks = len(results)
            overall_score = sum(r['score'] for r in results) / total_checks

            logger.info(f"[AntiSpoofingService.detect_spoofing:133] RESULTS SUMMARY:")
            logger.info(f"[AntiSpoofingService.detect_spoofing:134]   Total checks: {total_checks}")
            logger.info(f"[AntiSpoofingService.detect_spoofing:135]   Passed: {passed_count}")
            logger.info(f"[AntiSpoofingService.detect_spoofing:136]   Overall score: {overall_score:.4f}")
            
            # Log individual scores
            for r in results:
                status = "✓ PASS" if r['passed'] else "✗ FAIL"
                logger.info(f"[AntiSpoofingService.detect_spoofing:140]   {r['method']}: {r['score']:.4f} {status}")

            # Majority scales with however many checks actually ran
            majority_needed = max(self.min_checks_required, (total_checks // 2) + 1)
            logger.info(f"[AntiSpoofingService.detect_spoofing:144]   Majority needed: {majority_needed}")

            is_real = (
                (overall_score >= self.spoofing_threshold and passed_count >= majority_needed)
                or overall_score >= 0.75
            )
            logger.info(f"[AntiSpoofingService.detect_spoofing:149] INITIAL VERDICT: {'REAL' if is_real else 'SPOOF'}")
            logger.info(f"[AntiSpoofingService.detect_spoofing:150]   Overall score >= threshold ({self.spoofing_threshold}): {overall_score >= self.spoofing_threshold}")
            logger.info(f"[AntiSpoofingService.detect_spoofing:151]   Passed >= majority ({majority_needed}): {passed_count >= majority_needed}")
            logger.info(f"[AntiSpoofingService.detect_spoofing:152]   Overall >= 0.75: {overall_score >= 0.75}")

            # Hard veto: a strong moire signature together with a strong
            # glare signature is close to a definitive screen-replay
            # fingerprint.
            moire_result = next(r for r in results if r['method'] == 'moire_pattern')
            glare_result = next(r for r in results if r['method'] == 'specular_glare')
            logger.info(f"[AntiSpoofingService.detect_spoofing:158] MOIRE+GLARE VETO CHECK:")
            logger.info(f"[AntiSpoofingService.detect_spoofing:159]   Moire score: {moire_result['score']:.4f}")
            logger.info(f"[AntiSpoofingService.detect_spoofing:160]   Glare score: {glare_result['score']:.4f}")
            
            if moire_result['score'] <= 0.20 and glare_result['score'] <= 0.20:
                logger.warning(f"[AntiSpoofingService.detect_spoofing:163] ⚠ MOIRE+GLARE VETO TRIGGERED: "
                             f"Both scores <= 0.20, forcing SPOOF verdict")
                is_real = False
            else:
                logger.info(f"[AntiSpoofingService.detect_spoofing:166]   Moire+Glare veto NOT triggered")

            # Hard veto: if a motion check ran and shows essentially zero
            # motion or purely rigid (whole-frame-uniform) motion
            if liveness_checked:
                liveness_result = next(r for r in results if r['method'] == 'motion_liveness')
                logger.info(f"[AntiSpoofingService.detect_spoofing:172] MOTION VETO CHECK:")
                logger.info(f"[AntiSpoofingService.detect_spoofing:173]   Liveness score: {liveness_result['score']:.4f}")
                
                if liveness_result['score'] <= 0.15:
                    logger.warning(f"[AntiSpoofingService.detect_spoofing:176] ⚠ MOTION VETO TRIGGERED: "
                                 f"Liveness score <= 0.15, forcing SPOOF verdict")
                    is_real = False
                else:
                    logger.info(f"[AntiSpoofingService.detect_spoofing:179]   Motion veto NOT triggered")
            else:
                logger.info(f"[AntiSpoofingService.detect_spoofing:181]   Motion veto NOT checked (no liveness data)")

            result = {
                'is_real': is_real,
                'confidence': round(overall_score, 4),
                'threshold': self.spoofing_threshold,
                'details': {
                    'results': results,
                    'passed_checks': passed_count,
                    'total_checks': total_checks,
                    'liveness_checked': liveness_checked,
                    'verdict': 'REAL' if is_real else 'SPOOF'
                }
            }

            logger.info(f"[AntiSpoofingService.detect_spoofing:195] FINAL VERDICT: {result['details']['verdict']} "
                       f"(score: {result['confidence']:.4f}, passed: {passed_count}/{total_checks})")
            logger.info(f"[AntiSpoofingService.detect_spoofing:197] ========== SPOOFING DETECTION COMPLETE ==========")

            return result

        except Exception as e:
            logger.error(f"[AntiSpoofingService.detect_spoofing:201] ❌ Spoofing detection FAILED: {str(e)}", exc_info=True)
            logger.critical(f"[AntiSpoofingService.detect_spoofing:202] FAILING CLOSED - Returning REJECTED_ON_ERROR for security")
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
        logger.info(f"[AntiSpoofingService._make_result:213] Method: {method}, score: {score:.4f}, "
                    f"threshold: {t:.4f}, passed: {passed}")
        return result

    # ------------------------------------------------------------------
    # Original single-frame checks
    # ------------------------------------------------------------------

    def _analyze_texture(self, image: np.ndarray) -> float:
        """Analyze texture for screen/replay artifacts"""
        logger.info(f"[AntiSpoofingService._analyze_texture:223] Texture analysis - computing Laplacian variance...")
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            logger.info(f"[AntiSpoofingService._analyze_texture:227]   Laplacian variance: {laplacian_var:.2f}")

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
            
            logger.info(f"[AntiSpoofingService._analyze_texture:241]   Final texture score: {score:.4f}")
            return score

        except Exception as e:
            logger.warning(f"[AntiSpoofingService._analyze_texture:245] ❌ Texture analysis failed: {str(e)}")
            return 0.5

    def _analyze_color_distribution(self, image: np.ndarray) -> float:
        """Analyze color distribution for screen display artifacts"""
        logger.info(f"[AntiSpoofingService._analyze_color_distribution:251] Color analysis - computing saturation std...")
        try:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            saturation = hsv[:, :, 1]
            saturation_std = np.std(saturation)
            logger.info(f"[AntiSpoofingService._analyze_color_distribution:256]   Saturation std: {saturation_std:.2f}")

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
            
            logger.info(f"[AntiSpoofingService._analyze_color_distribution:270]   Final color score: {score:.4f}")
            return score

        except Exception as e:
            logger.warning(f"[AntiSpoofingService._analyze_color_distribution:274] ❌ Color analysis failed: {str(e)}")
            return 0.5

    def _analyze_edges(self, image: np.ndarray) -> float:
        """Analyze edges for print artifacts"""
        logger.info(f"[AntiSpoofingService._analyze_edges:280] Edge analysis - computing edge density...")
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            logger.info(f"[AntiSpoofingService._analyze_edges:285]   Edge density: {edge_density:.4f}")

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
            
            logger.info(f"[AntiSpoofingService._analyze_edges:299]   Final edge score: {score:.4f}")
            return score

        except Exception as e:
            logger.warning(f"[AntiSpoofingService._analyze_edges:303] ❌ Edge analysis failed: {str(e)}")
            return 0.5

    def _analyze_noise_pattern(self, image: np.ndarray) -> float:
        """Analyze noise pattern for digital artifacts"""
        logger.info(f"[AntiSpoofingService._analyze_noise_pattern:309] Noise analysis - computing noise std...")
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            noise = cv2.absdiff(gray, blurred)
            noise_std = np.std(noise)
            logger.info(f"[AntiSpoofingService._analyze_noise_pattern:315]   Noise std: {noise_std:.2f}")

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
            
            logger.info(f"[AntiSpoofingService._analyze_noise_pattern:327]   Final noise score: {score:.4f}")
            return score

        except Exception as e:
            logger.warning(f"[AntiSpoofingService._analyze_noise_pattern:331] ❌ Noise analysis failed: {str(e)}")
            return 0.5

    # ------------------------------------------------------------------
    # New checks
    # ------------------------------------------------------------------

    def _detect_moire_pattern(self, image: np.ndarray) -> float:
        """
        Moire pattern detection via FFT.
        """
        logger.info(f"[AntiSpoofingService._detect_moire_pattern:341] Moire detection - computing FFT...")
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (256, 256))
            logger.info(f"[AntiSpoofingService._detect_moire_pattern:345]   Resized to 256x256 for FFT")

            f = np.fft.fft2(gray.astype(np.float32))
            fshift = np.fft.fftshift(f)
            magnitude = np.log1p(np.abs(fshift))

            h, w = magnitude.shape
            cy, cx = h // 2, w // 2

            low_freq_radius = 12
            y, x = np.ogrid[:h, :w]
            dist_from_center = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
            high_freq_mask = dist_from_center > low_freq_radius

            high_freq_energy = magnitude[high_freq_mask]
            mean_energy = np.mean(high_freq_energy)
            std_energy = np.std(high_freq_energy)
            max_energy = np.max(high_freq_energy)
            logger.info(f"[AntiSpoofingService._detect_moire_pattern:362]   High freq stats - mean: {mean_energy:.4f}, "
                        f"std: {std_energy:.4f}, max: {max_energy:.4f}")

            if std_energy < 1e-6:
                peak_ratio = 0.0
            else:
                peak_ratio = (max_energy - mean_energy) / std_energy

            outlier_threshold = mean_energy + 3 * std_energy
            outlier_fraction = np.sum(high_freq_energy > outlier_threshold) / high_freq_energy.size
            logger.info(f"[AntiSpoofingService._detect_moire_pattern:371]   Peak ratio: {peak_ratio:.4f}, "
                        f"outlier fraction: {outlier_fraction:.6f}")

            if peak_ratio > 15 or outlier_fraction > 0.01:
                score = 0.20
            elif peak_ratio > 10 or outlier_fraction > 0.005:
                score = 0.40
            elif peak_ratio > 7:
                score = 0.60
            elif peak_ratio > 5:
                score = 0.80
            else:
                score = 1.0
            
            logger.info(f"[AntiSpoofingService._detect_moire_pattern:384]   Final moire score: {score:.4f}")
            return score

        except Exception as e:
            logger.warning(f"[AntiSpoofingService._detect_moire_pattern:388] ❌ Moire pattern analysis failed: {str(e)}")
            return 0.5

    def _analyze_texture_lbp(self, image: np.ndarray) -> float:
        """
        Local Binary Pattern (LBP) texture analysis
        """
        logger.info(f"[AntiSpoofingService._analyze_texture_lbp:396] LBP analysis - computing texture entropy...")
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (200, 200)).astype(np.int32)
            logger.info(f"[AntiSpoofingService._analyze_texture_lbp:400]   Resized to 200x200 for LBP")

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
                logger.warning(f"[AntiSpoofingService._analyze_texture_lbp:414]   LBP histogram sum is 0")
                return 0.5

            prob = hist / hist_sum
            prob = prob[prob > 0]
            entropy = -np.sum(prob * np.log2(prob))
            normalized_entropy = entropy / 8.0
            logger.info(f"[AntiSpoofingService._analyze_texture_lbp:422]   LBP entropy: {entropy:.4f}, "
                        f"normalized: {normalized_entropy:.4f}")

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
            
            logger.info(f"[AntiSpoofingService._analyze_texture_lbp:435]   Final LBP score: {score:.4f}")
            return score

        except Exception as e:
            logger.warning(f"[AntiSpoofingService._analyze_texture_lbp:439] ❌ LBP texture analysis failed: {str(e)}")
            return 0.5

    def _analyze_specular_highlights(self, image: np.ndarray) -> float:
        """
        Specular highlight / glare analysis
        """
        logger.info(f"[AntiSpoofingService._analyze_specular_highlights:447] Specular glare analysis...")
        try:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            v_channel = hsv[:, :, 2]

            bright_mask = (v_channel > 235).astype(np.uint8) * 255
            total_pixels = bright_mask.size
            bright_fraction = np.sum(bright_mask > 0) / total_pixels
            logger.info(f"[AntiSpoofingService._analyze_specular_highlights:455]   Bright fraction: {bright_fraction:.4f}")

            num_labels, _, stats, _ = cv2.connectedComponentsWithStats(bright_mask, connectivity=8)

            large_blob_fraction = 0.0
            if num_labels > 1:
                areas = stats[1:, cv2.CC_STAT_AREA]
                large_blob_fraction = areas.max() / total_pixels
                logger.info(f"[AntiSpoofingService._analyze_specular_highlights:463]   Number of blobs: {num_labels-1}, "
                           f"largest blob fraction: {large_blob_fraction:.4f}")

            if bright_fraction > 0.12 or large_blob_fraction > 0.06:
                score = 0.20
            elif bright_fraction > 0.07 or large_blob_fraction > 0.03:
                score = 0.45
            elif bright_fraction > 0.03 or large_blob_fraction > 0.015:
                score = 0.70
            else:
                score = 1.0
            
            logger.info(f"[AntiSpoofingService._analyze_specular_highlights:474]   Final glare score: {score:.4f}")
            return score

        except Exception as e:
            logger.warning(f"[AntiSpoofingService._analyze_specular_highlights:478] ❌ Specular highlight analysis failed: {str(e)}")
            return 0.5

    def _analyze_motion_liveness(self, frames: List[np.ndarray]) -> float:
        """
        Motion liveness check
        """
        logger.info(f"[AntiSpoofingService._analyze_motion_liveness:486] Motion liveness analysis with {len(frames)} frames...")
        try:
            grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
            logger.info(f"[AntiSpoofingService._analyze_motion_liveness:489]   Converted {len(grays)} frames to grayscale")

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
                logger.info(f"[AntiSpoofingService._analyze_motion_liveness:503]   Frame pair {i}-{i+1}: "
                           f"mean flow: {flow_mean:.4f}, std flow: {flow_std:.4f}")

            avg_motion = float(np.mean(flow_means))
            avg_variation = float(np.mean(flow_stds))
            logger.info(f"[AntiSpoofingService._analyze_motion_liveness:508]   Average motion: {avg_motion:.4f}, "
                        f"average variation: {avg_variation:.4f}")

            if avg_motion < 0.05:
                score = 0.15
                logger.info(f"[AntiSpoofingService._analyze_motion_liveness:512]   FROZEN MOTION detected")
            else:
                rigidity_ratio = avg_variation / max(avg_motion, 1e-6)
                logger.info(f"[AntiSpoofingService._analyze_motion_liveness:515]   Rigidity ratio: {rigidity_ratio:.4f}")
                
                if rigidity_ratio < 0.15:
                    score = 0.30
                    logger.info(f"[AntiSpoofingService._analyze_motion_liveness:519]   HIGHLY RIGID motion detected")
                elif rigidity_ratio < 0.30:
                    score = 0.55
                    logger.info(f"[AntiSpoofingService._analyze_motion_liveness:522]   MODERATELY RIGID motion")
                elif rigidity_ratio < 0.50:
                    score = 0.75
                    logger.info(f"[AntiSpoofingService._analyze_motion_liveness:525]   SOME non-rigid motion")
                else:
                    score = 1.0
                    logger.info(f"[AntiSpoofingService._analyze_motion_liveness:528]   NATURAL non-rigid motion detected")

            logger.info(f"[AntiSpoofingService._analyze_motion_liveness:530]   Final motion liveness score: {score:.4f}")
            return score

        except Exception as e:
            logger.warning(f"[AntiSpoofingService._analyze_motion_liveness:534] ❌ Motion liveness analysis failed: {str(e)}")
            return 0.5