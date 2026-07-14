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
        logger.debug(f"[AntiSpoofingService.__init__:32] Initializing AntiSpoofingService")
        self.initialized = False
        # Hard-coded thresholds (no settings dependency)
        self.spoofing_threshold = 0.50
        self.min_checks_required = 2
        self.min_individual_threshold = 0.35
        logger.debug(f"[AntiSpoofingService.__init__:38] Thresholds set - spoofing: {self.spoofing_threshold}, "
                    f"min_checks: {self.min_checks_required}, min_individual: {self.min_individual_threshold}")

    async def initialize(self):
        """Initialize anti-spoofing service"""
        logger.debug(f"[AntiSpoofingService.initialize:43] Starting initialization")
        self.initialized = True
        logger.info("Anti-spoofing service initialized")
        logger.debug(f"[AntiSpoofingService.initialize:46] Initialization complete, initialized={self.initialized}")

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
        logger.debug(f"[AntiSpoofingService.detect_spoofing:67] Starting spoofing detection")
        logger.debug(f"[AntiSpoofingService.detect_spoofing:68] Image shape: {image_data.shape}, "
                    f"dtype: {image_data.dtype}")
        logger.debug(f"[AntiSpoofingService.detect_spoofing:70] Motion frames provided: {motion_frames is not None}, "
                    f"count: {len(motion_frames) if motion_frames else 0}")
        
        try:
            results = []

            # Check 1: Texture Analysis
            logger.debug(f"[AntiSpoofingService.detect_spoofing:77] Running texture analysis (Check 1/7)")
            texture_score = self._analyze_texture(image_data)
            results.append(self._make_result('texture_analysis', texture_score))
            logger.debug(f"[AntiSpoofingService.detect_spoofing:80] Texture analysis score: {texture_score:.4f}")

            # Check 2: Color Analysis
            logger.debug(f"[AntiSpoofingService.detect_spoofing:83] Running color analysis (Check 2/7)")
            color_score = self._analyze_color_distribution(image_data)
            results.append(self._make_result('color_analysis', color_score))
            logger.debug(f"[AntiSpoofingService.detect_spoofing:86] Color analysis score: {color_score:.4f}")

            # Check 3: Edge Analysis
            logger.debug(f"[AntiSpoofingService.detect_spoofing:89] Running edge analysis (Check 3/7)")
            edge_score = self._analyze_edges(image_data)
            results.append(self._make_result('edge_analysis', edge_score))
            logger.debug(f"[AntiSpoofingService.detect_spoofing:92] Edge analysis score: {edge_score:.4f}")

            # Check 4: Noise Analysis
            logger.debug(f"[AntiSpoofingService.detect_spoofing:95] Running noise analysis (Check 4/7)")
            noise_score = self._analyze_noise_pattern(image_data)
            results.append(self._make_result('noise_analysis', noise_score, threshold=0.35))
            logger.debug(f"[AntiSpoofingService.detect_spoofing:98] Noise analysis score: {noise_score:.4f}")

            # Check 5: Moire Pattern
            logger.debug(f"[AntiSpoofingService.detect_spoofing:101] Running moire pattern detection (Check 5/7)")
            moire_score = self._detect_moire_pattern(image_data)
            results.append(self._make_result('moire_pattern', moire_score))
            logger.debug(f"[AntiSpoofingService.detect_spoofing:104] Moire pattern score: {moire_score:.4f}")

            # Check 6: LBP Texture
            logger.debug(f"[AntiSpoofingService.detect_spoofing:107] Running LBP texture analysis (Check 6/7)")
            lbp_score = self._analyze_texture_lbp(image_data)
            results.append(self._make_result('lbp_texture', lbp_score))
            logger.debug(f"[AntiSpoofingService.detect_spoofing:110] LBP texture score: {lbp_score:.4f}")

            # Check 7: Specular Glare
            logger.debug(f"[AntiSpoofingService.detect_spoofing:113] Running specular glare analysis (Check 7/7)")
            glare_score = self._analyze_specular_highlights(image_data)
            results.append(self._make_result('specular_glare', glare_score))
            logger.debug(f"[AntiSpoofingService.detect_spoofing:116] Specular glare score: {glare_score:.4f}")

            # Check 8: Motion Liveness (optional)
            liveness_checked = False
            if motion_frames and len(motion_frames) >= 2:
                logger.debug(f"[AntiSpoofingService.detect_spoofing:121] Running motion liveness (Check 8/8)")
                liveness_score = self._analyze_motion_liveness(motion_frames)
                results.append(self._make_result('motion_liveness', liveness_score))
                liveness_checked = True
                logger.debug(f"[AntiSpoofingService.detect_spoofing:125] Motion liveness score: {liveness_score:.4f}")
            else:
                logger.debug(f"[AntiSpoofingService.detect_spoofing:127] Skipping motion liveness - insufficient frames")

            passed_count = sum(1 for r in results if r['passed'])
            total_checks = len(results)
            overall_score = sum(r['score'] for r in results) / total_checks

            logger.debug(f"[AntiSpoofingService.detect_spoofing:133] Results summary - "
                        f"passed: {passed_count}/{total_checks}, overall_score: {overall_score:.4f}")

            # Majority scales with however many checks actually ran
            # (7 or 8), instead of the old hard-coded ">= 3".
            majority_needed = max(self.min_checks_required, (total_checks // 2) + 1)
            logger.debug(f"[AntiSpoofingService.detect_spoofing:138] Majority needed: {majority_needed}")

            is_real = (
                (overall_score >= self.spoofing_threshold and passed_count >= majority_needed)
                or overall_score >= 0.75
            )
            logger.debug(f"[AntiSpoofingService.detect_spoofing:143] Initial verdict: {'REAL' if is_real else 'SPOOF'} "
                        f"(overall_score >= threshold: {overall_score >= self.spoofing_threshold}, "
                        f"passed >= majority: {passed_count >= majority_needed}, "
                        f"overall >= 0.75: {overall_score >= 0.75})")

            # Hard veto: a strong moire signature together with a strong
            # glare signature is close to a definitive screen-replay
            # fingerprint. Don't let it get averaged away by four other
            # checks that happen to score well.
            moire_result = next(r for r in results if r['method'] == 'moire_pattern')
            glare_result = next(r for r in results if r['method'] == 'specular_glare')
            logger.debug(f"[AntiSpoofingService.detect_spoofing:154] Moire veto check - "
                        f"moire_score: {moire_result['score']:.4f}, glare_score: {glare_result['score']:.4f}")
            
            if moire_result['score'] <= 0.20 and glare_result['score'] <= 0.20:
                logger.warning(f"[AntiSpoofingService.detect_spoofing:157] MOIRE+GLARE VETO TRIGGERED: "
                             f"Both scores <= 0.20, forcing SPOOF verdict")
                is_real = False

            # Hard veto: if a motion check ran and shows essentially zero
            # motion or purely rigid (whole-frame-uniform) motion, that's
            # a strong static-photo/flat-surface signal.
            if liveness_checked:
                liveness_result = next(r for r in results if r['method'] == 'motion_liveness')
                logger.debug(f"[AntiSpoofingService.detect_spoofing:165] Motion veto check - "
                           f"liveness_score: {liveness_result['score']:.4f}")
                
                if liveness_result['score'] <= 0.15:
                    logger.warning(f"[AntiSpoofingService.detect_spoofing:168] MOTION VETO TRIGGERED: "
                                 f"Liveness score <= 0.15, forcing SPOOF verdict")
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
                    'verdict': 'REAL' if is_real else 'SPOOF'
                }
            }

            logger.info(f"Anti-spoofing: {result['details']['verdict']} "
                       f"(score: {result['confidence']:.4f}, passed: {passed_count}/{total_checks})")
            logger.debug(f"[AntiSpoofingService.detect_spoofing:187] Final result details: {result['details']}")

            return result

        except Exception as e:
            logger.error(f"[AntiSpoofingService.detect_spoofing:191] Spoofing detection failed: {str(e)}", exc_info=True)
            # SECURITY: fail CLOSED, not open. The previous version
            # returned is_real=True/confidence=1.0 here, which means any
            # exception (malformed image, edge-case input, etc.) bypassed
            # every spoofing check. An attacker who can trigger an
            # exception on demand would get an automatic pass. If
            # detection breaks, reject the attempt instead.
            logger.critical(f"[AntiSpoofingService.detect_spoofing:197] FAILING CLOSED - Returning REJECTED_ON_ERROR")
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
        logger.debug(f"[AntiSpoofingService._make_result:210] Method: {method}, score: {score:.4f}, "
                    f"threshold: {t:.4f}, passed: {passed}")
        return result

    # ------------------------------------------------------------------
    # Original single-frame checks
    # ------------------------------------------------------------------

    def _analyze_texture(self, image: np.ndarray) -> float:
        """Analyze texture for screen/replay artifacts"""
        logger.debug(f"[AntiSpoofingService._analyze_texture:220] Starting texture analysis")
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            logger.debug(f"[AntiSpoofingService._analyze_texture:223] Converted to grayscale, shape: {gray.shape}")
            
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            logger.debug(f"[AntiSpoofingService._analyze_texture:226] Laplacian variance: {laplacian_var:.2f}")

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
            
            logger.debug(f"[AntiSpoofingService._analyze_texture:240] Texture score: {score:.4f} "
                        f"(laplacian_var: {laplacian_var:.2f})")
            return score

        except Exception as e:
            logger.warning(f"[AntiSpoofingService._analyze_texture:244] Texture analysis failed: {str(e)}", exc_info=True)
            return 0.5

    def _analyze_color_distribution(self, image: np.ndarray) -> float:
        """Analyze color distribution for screen display artifacts"""
        logger.debug(f"[AntiSpoofingService._analyze_color_distribution:250] Starting color distribution analysis")
        try:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            saturation = hsv[:, :, 1]
            saturation_std = np.std(saturation)
            logger.debug(f"[AntiSpoofingService._analyze_color_distribution:255] Saturation std: {saturation_std:.2f}")

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
            
            logger.debug(f"[AntiSpoofingService._analyze_color_distribution:269] Color score: {score:.4f} "
                        f"(saturation_std: {saturation_std:.2f})")
            return score

        except Exception as e:
            logger.warning(f"[AntiSpoofingService._analyze_color_distribution:273] Color analysis failed: {str(e)}", exc_info=True)
            return 0.5

    def _analyze_edges(self, image: np.ndarray) -> float:
        """Analyze edges for print artifacts"""
        logger.debug(f"[AntiSpoofingService._analyze_edges:279] Starting edge analysis")
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            logger.debug(f"[AntiSpoofingService._analyze_edges:284] Edge density: {edge_density:.4f}")

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
            
            logger.debug(f"[AntiSpoofingService._analyze_edges:298] Edge score: {score:.4f} "
                        f"(edge_density: {edge_density:.4f})")
            return score

        except Exception as e:
            logger.warning(f"[AntiSpoofingService._analyze_edges:302] Edge analysis failed: {str(e)}", exc_info=True)
            return 0.5

    def _analyze_noise_pattern(self, image: np.ndarray) -> float:
        """Analyze noise pattern for digital artifacts"""
        logger.debug(f"[AntiSpoofingService._analyze_noise_pattern:308] Starting noise pattern analysis")
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            noise = cv2.absdiff(gray, blurred)
            noise_std = np.std(noise)
            logger.debug(f"[AntiSpoofingService._analyze_noise_pattern:314] Noise std: {noise_std:.2f}")

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
            
            logger.debug(f"[AntiSpoofingService._analyze_noise_pattern:326] Noise score: {score:.4f} "
                        f"(noise_std: {noise_std:.2f})")
            return score

        except Exception as e:
            logger.warning(f"[AntiSpoofingService._analyze_noise_pattern:330] Noise analysis failed: {str(e)}", exc_info=True)
            return 0.5

    # ------------------------------------------------------------------
    # New checks
    # ------------------------------------------------------------------

    def _detect_moire_pattern(self, image: np.ndarray) -> float:
        """
        Moire pattern detection via FFT.

        When a screen (phone/tablet/monitor) displaying a face is
        recaptured by a camera, interference between the screen's pixel
        grid and the camera sensor's grid produces periodic high-frequency
        energy concentrated in narrow bands -- visible as sharp, isolated
        peaks in the FFT magnitude spectrum away from the DC (center)
        component. Natural (non-screen) faces don't produce this.
        """
        logger.debug(f"[AntiSpoofingService._detect_moire_pattern:346] Starting moire pattern detection")
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (256, 256))  # normalize spectrum size
            logger.debug(f"[AntiSpoofingService._detect_moire_pattern:350] Resized to 256x256")

            f = np.fft.fft2(gray.astype(np.float32))
            fshift = np.fft.fftshift(f)
            magnitude = np.log1p(np.abs(fshift))
            logger.debug(f"[AntiSpoofingService._detect_moire_pattern:355] FFT computed, magnitude shape: {magnitude.shape}")

            h, w = magnitude.shape
            cy, cx = h // 2, w // 2

            # Exclude the low-frequency center -- that's just normal
            # image content (smooth shading, coarse structure).
            low_freq_radius = 12
            y, x = np.ogrid[:h, :w]
            dist_from_center = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
            high_freq_mask = dist_from_center > low_freq_radius

            high_freq_energy = magnitude[high_freq_mask]
            mean_energy = np.mean(high_freq_energy)
            std_energy = np.std(high_freq_energy)
            max_energy = np.max(high_freq_energy)
            logger.debug(f"[AntiSpoofingService._detect_moire_pattern:371] High freq stats - "
                        f"mean: {mean_energy:.4f}, std: {std_energy:.4f}, max: {max_energy:.4f}")

            if std_energy < 1e-6:
                peak_ratio = 0.0
            else:
                peak_ratio = (max_energy - mean_energy) / std_energy

            outlier_threshold = mean_energy + 3 * std_energy
            outlier_fraction = np.sum(high_freq_energy > outlier_threshold) / high_freq_energy.size
            logger.debug(f"[AntiSpoofingService._detect_moire_pattern:380] Peak ratio: {peak_ratio:.4f}, "
                        f"outlier fraction: {outlier_fraction:.6f}")

            # Stronger/more isolated peaks -> more likely a screen replay
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
            
            logger.debug(f"[AntiSpoofingService._detect_moire_pattern:393] Moire score: {score:.4f}")
            return score

        except Exception as e:
            logger.warning(f"[AntiSpoofingService._detect_moire_pattern:397] Moire pattern analysis failed: {str(e)}", exc_info=True)
            return 0.5

    def _analyze_texture_lbp(self, image: np.ndarray) -> float:
        """
        Local Binary Pattern (LBP) texture analysis, targeted at printed
        photo attacks.

        Real skin has fine, irregular micro-texture (pores, fine hair,
        blemishes) that produces a rich, high-entropy LBP histogram.
        Printed photos lose this high-frequency detail through the print
        process (dot-gain, paper fiber texture, ink diffusion), producing
        a flatter, lower-entropy LBP histogram dominated by a handful of
        uniform patterns.
        """
        logger.debug(f"[AntiSpoofingService._analyze_texture_lbp:412] Starting LBP texture analysis")
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (200, 200)).astype(np.int32)
            logger.debug(f"[AntiSpoofingService._analyze_texture_lbp:416] Resized to 200x200 for LBP")

            center = gray[1:-1, 1:-1]
            offsets = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]

            lbp = np.zeros_like(center, dtype=np.uint8)
            for i, (dy, dx) in enumerate(offsets):
                neighbor = gray[1 + dy: 1 + dy + center.shape[0], 1 + dx: 1 + dx + center.shape[1]]
                lbp |= ((neighbor >= center).astype(np.uint8) << i)
            
            logger.debug(f"[AntiSpoofingService._analyze_texture_lbp:426] LBP codes computed")

            hist, _ = np.histogram(lbp, bins=256, range=(0, 256))
            hist = hist.astype(np.float64)
            hist_sum = hist.sum()
            
            if hist_sum == 0:
                logger.warning(f"[AntiSpoofingService._analyze_texture_lbp:433] LBP histogram sum is 0")
                return 0.5

            prob = hist / hist_sum
            prob = prob[prob > 0]
            entropy = -np.sum(prob * np.log2(prob))
            normalized_entropy = entropy / 8.0  # log2(256) = 8
            logger.debug(f"[AntiSpoofingService._analyze_texture_lbp:441] LBP entropy: {entropy:.4f}, "
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
            
            logger.debug(f"[AntiSpoofingService._analyze_texture_lbp:454] LBP score: {score:.4f}")
            return score

        except Exception as e:
            logger.warning(f"[AntiSpoofingService._analyze_texture_lbp:458] LBP texture analysis failed: {str(e)}", exc_info=True)
            return 0.5

    def _analyze_specular_highlights(self, image: np.ndarray) -> float:
        """
        Specular highlight / glare analysis, targeted at screen glare
        (phones, laptops, monitors held up to the camera).

        Real skin under normal lighting produces small, soft specular
        highlights (forehead, nose tip, cheekbones). A screen or glossy
        printed photo recaptured by a camera tends to produce large,
        sharp-edged, often clipped (near pure white) glare regions that
        cover a much larger fraction of the frame.
        """
        logger.debug(f"[AntiSpoofingService._analyze_specular_highlights:473] Starting specular highlight analysis")
        try:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            v_channel = hsv[:, :, 2]

            bright_mask = (v_channel > 235).astype(np.uint8) * 255
            total_pixels = bright_mask.size
            bright_fraction = np.sum(bright_mask > 0) / total_pixels
            logger.debug(f"[AntiSpoofingService._analyze_specular_highlights:481] Bright fraction: {bright_fraction:.4f}")

            num_labels, _, stats, _ = cv2.connectedComponentsWithStats(bright_mask, connectivity=8)

            large_blob_fraction = 0.0
            if num_labels > 1:
                areas = stats[1:, cv2.CC_STAT_AREA]  # skip background label 0
                large_blob_fraction = areas.max() / total_pixels
                logger.debug(f"[AntiSpoofingService._analyze_specular_highlights:489] Number of blobs: {num_labels-1}, "
                           f"largest blob fraction: {large_blob_fraction:.4f}")

            if bright_fraction > 0.12 or large_blob_fraction > 0.06:
                score = 0.20  # large glare region -> likely screen/glossy photo
            elif bright_fraction > 0.07 or large_blob_fraction > 0.03:
                score = 0.45
            elif bright_fraction > 0.03 or large_blob_fraction > 0.015:
                score = 0.70
            else:
                score = 1.0  # small/no specular highlights -> consistent with real skin
            
            logger.debug(f"[AntiSpoofingService._analyze_specular_highlights:500] Specular glare score: {score:.4f}")
            return score

        except Exception as e:
            logger.warning(f"[AntiSpoofingService._analyze_specular_highlights:504] Specular highlight analysis failed: {str(e)}", exc_info=True)
            return 0.5

    def _analyze_motion_liveness(self, frames: List[np.ndarray]) -> float:
        """
        Motion liveness check. Requires >= 2 frames captured roughly
        0.3-1.0s apart (caller's responsibility to supply these -- e.g.
        sample frames from a short capture window during verification).

        A live face in front of a camera shows small, irregular,
        non-rigid motion (breathing, blinking, micro head movement) that
        varies across different facial regions. A printed photo or a
        phone/tablet held up produces either near-zero motion (held
        still) or purely rigid/uniform motion (the whole frame moves
        together because it's one flat rigid surface, e.g. hand shake
        while holding the photo/device).
        """
        logger.debug(f"[AntiSpoofingService._analyze_motion_liveness:521] Starting motion liveness analysis "
                    f"with {len(frames)} frames")
        try:
            grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
            logger.debug(f"[AntiSpoofingService._analyze_motion_liveness:525] Converted {len(grays)} frames to grayscale")

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
                logger.debug(f"[AntiSpoofingService._analyze_motion_liveness:539] Frame pair {i}-{i+1}: "
                           f"mean flow: {flow_mean:.4f}, std flow: {flow_std:.4f}")

            avg_motion = float(np.mean(flow_means))
            avg_variation = float(np.mean(flow_stds))  # non-rigidity across the frame
            logger.debug(f"[AntiSpoofingService._analyze_motion_liveness:544] Average motion: {avg_motion:.4f}, "
                        f"average variation: {avg_variation:.4f}")

            # Essentially frozen -> static photo held motionless, or a
            # paused/looped video frame.
            if avg_motion < 0.05:
                score = 0.15
                logger.debug(f"[AntiSpoofingService._analyze_motion_liveness:550] Frozen motion detected")

            # Motion present but perfectly uniform across the frame ->
            # rigid object (a photo/phone panning or shaking as a whole),
            # not a live, non-rigidly-deforming face.
            rigidity_ratio = avg_variation / max(avg_motion, 1e-6)
            logger.debug(f"[AntiSpoofingService._analyze_motion_liveness:556] Rigidity ratio: {rigidity_ratio:.4f}")
            
            if rigidity_ratio < 0.15:
                score = 0.30
                logger.debug(f"[AntiSpoofingService._analyze_motion_liveness:560] Highly rigid motion detected")
            elif rigidity_ratio < 0.30:
                score = 0.55
                logger.debug(f"[AntiSpoofingService._analyze_motion_liveness:563] Moderately rigid motion")
            elif rigidity_ratio < 0.50:
                score = 0.75
                logger.debug(f"[AntiSpoofingService._analyze_motion_liveness:566] Some non-rigid motion")
            else:
                score = 1.0  # natural, non-rigid micro-motion -> consistent with a live face
                logger.debug(f"[AntiSpoofingService._analyze_motion_liveness:569] Natural non-rigid motion detected")

            logger.debug(f"[AntiSpoofingService._analyze_motion_liveness:571] Motion liveness score: {score:.4f}")
            return score

        except Exception as e:
            logger.warning(f"[AntiSpoofingService._analyze_motion_liveness:575] Motion liveness analysis failed: {str(e)}", exc_info=True)
            return 0.5