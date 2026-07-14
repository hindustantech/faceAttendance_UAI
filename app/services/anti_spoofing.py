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
        self.initialized = False
        # Hard-coded thresholds (no settings dependency)
        self.spoofing_threshold = 0.50
        self.min_checks_required = 2
        self.min_individual_threshold = 0.35

    async def initialize(self):
        """Initialize anti-spoofing service"""
        self.initialized = True
        logger.info("Anti-spoofing service initialized")

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
        try:
            results = []

            results.append(self._make_result('texture_analysis', self._analyze_texture(image_data)))
            results.append(self._make_result('color_analysis', self._analyze_color_distribution(image_data)))
            results.append(self._make_result('edge_analysis', self._analyze_edges(image_data)))
            results.append(self._make_result('noise_analysis', self._analyze_noise_pattern(image_data), threshold=0.35))
            results.append(self._make_result('moire_pattern', self._detect_moire_pattern(image_data)))
            results.append(self._make_result('lbp_texture', self._analyze_texture_lbp(image_data)))
            results.append(self._make_result('specular_glare', self._analyze_specular_highlights(image_data)))

            liveness_checked = False
            if motion_frames and len(motion_frames) >= 2:
                liveness_score = self._analyze_motion_liveness(motion_frames)
                results.append(self._make_result('motion_liveness', liveness_score))
                liveness_checked = True

            passed_count = sum(1 for r in results if r['passed'])
            total_checks = len(results)
            overall_score = sum(r['score'] for r in results) / total_checks

            # Majority scales with however many checks actually ran
            # (7 or 8), instead of the old hard-coded ">= 3".
            majority_needed = max(self.min_checks_required, (total_checks // 2) + 1)

            is_real = (
                (overall_score >= self.spoofing_threshold and passed_count >= majority_needed)
                or overall_score >= 0.75
            )

            # Hard veto: a strong moire signature together with a strong
            # glare signature is close to a definitive screen-replay
            # fingerprint. Don't let it get averaged away by four other
            # checks that happen to score well.
            moire_result = next(r for r in results if r['method'] == 'moire_pattern')
            glare_result = next(r for r in results if r['method'] == 'specular_glare')
            if moire_result['score'] <= 0.20 and glare_result['score'] <= 0.20:
                is_real = False

            # Hard veto: if a motion check ran and shows essentially zero
            # motion or purely rigid (whole-frame-uniform) motion, that's
            # a strong static-photo/flat-surface signal.
            if liveness_checked:
                liveness_result = next(r for r in results if r['method'] == 'motion_liveness')
                if liveness_result['score'] <= 0.15:
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

            return result

        except Exception as e:
            logger.error(f"Spoofing detection failed: {str(e)}")
            # SECURITY: fail CLOSED, not open. The previous version
            # returned is_real=True/confidence=1.0 here, which means any
            # exception (malformed image, edge-case input, etc.) bypassed
            # every spoofing check. An attacker who can trigger an
            # exception on demand would get an automatic pass. If
            # detection breaks, reject the attempt instead.
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
        return {'method': method, 'score': score, 'passed': score >= t}

    # ------------------------------------------------------------------
    # Original single-frame checks
    # ------------------------------------------------------------------

    def _analyze_texture(self, image: np.ndarray) -> float:
        """Analyze texture for screen/replay artifacts"""
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
        """Analyze color distribution for screen display artifacts"""
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
        """Analyze edges for print artifacts"""
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
        """Analyze noise pattern for digital artifacts"""
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
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (256, 256))  # normalize spectrum size

            f = np.fft.fft2(gray.astype(np.float32))
            fshift = np.fft.fftshift(f)
            magnitude = np.log1p(np.abs(fshift))

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

            if std_energy < 1e-6:
                peak_ratio = 0.0
            else:
                peak_ratio = (max_energy - mean_energy) / std_energy

            outlier_threshold = mean_energy + 3 * std_energy
            outlier_fraction = np.sum(high_freq_energy > outlier_threshold) / high_freq_energy.size

            # Stronger/more isolated peaks -> more likely a screen replay
            if peak_ratio > 15 or outlier_fraction > 0.01:
                return 0.20
            elif peak_ratio > 10 or outlier_fraction > 0.005:
                return 0.40
            elif peak_ratio > 7:
                return 0.60
            elif peak_ratio > 5:
                return 0.80
            else:
                return 1.0

        except Exception as e:
            logger.warning(f"Moire pattern analysis failed: {str(e)}")
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
            normalized_entropy = entropy / 8.0  # log2(256) = 8

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
            logger.warning(f"LBP texture analysis failed: {str(e)}")
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
        try:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            v_channel = hsv[:, :, 2]

            bright_mask = (v_channel > 235).astype(np.uint8) * 255
            total_pixels = bright_mask.size
            bright_fraction = np.sum(bright_mask > 0) / total_pixels

            num_labels, _, stats, _ = cv2.connectedComponentsWithStats(bright_mask, connectivity=8)

            large_blob_fraction = 0.0
            if num_labels > 1:
                areas = stats[1:, cv2.CC_STAT_AREA]  # skip background label 0
                large_blob_fraction = areas.max() / total_pixels

            if bright_fraction > 0.12 or large_blob_fraction > 0.06:
                return 0.20  # large glare region -> likely screen/glossy photo
            elif bright_fraction > 0.07 or large_blob_fraction > 0.03:
                return 0.45
            elif bright_fraction > 0.03 or large_blob_fraction > 0.015:
                return 0.70
            else:
                return 1.0  # small/no specular highlights -> consistent with real skin

        except Exception as e:
            logger.warning(f"Specular highlight analysis failed: {str(e)}")
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
            avg_variation = float(np.mean(flow_stds))  # non-rigidity across the frame

            # Essentially frozen -> static photo held motionless, or a
            # paused/looped video frame.
            if avg_motion < 0.05:
                return 0.15

            # Motion present but perfectly uniform across the frame ->
            # rigid object (a photo/phone panning or shaking as a whole),
            # not a live, non-rigidly-deforming face.
            rigidity_ratio = avg_variation / max(avg_motion, 1e-6)
            if rigidity_ratio < 0.15:
                return 0.30
            elif rigidity_ratio < 0.30:
                return 0.55
            elif rigidity_ratio < 0.50:
                return 0.75
            else:
                return 1.0  # natural, non-rigid micro-motion -> consistent with a live face

        except Exception as e:
            logger.warning(f"Motion liveness analysis failed: {str(e)}")
            return 0.5