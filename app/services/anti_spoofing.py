import cv2
import numpy as np
from typing import Dict
from app.config import settings
from app.utils.logger import setup_logger

logger = setup_logger(__name__)


class AntiSpoofingService:
    """
    Anti-Spoofing Service

    Detects presentation attacks:
    1. Printed photos (paper)         -> halftone / print-dot FFT signature
    2. Phone / monitor screen replays -> moire pattern FFT signature
    3. Screen glare / specular glass  -> specular highlight analysis
    4. General texture/noise cues     -> supporting signals only, not decisive alone

    Design principle: screen and print detectors are the PRIMARY signal because
    they target the actual physical artifact of the attack. Texture/color/noise
    are secondary corroborating signals only - they should never by themselves
    let a spoof pass, and should never by themselves block a real face either.
    """

    def __init__(self):
        self.initialized = False

        # Primary attack-specific detectors - these are the ones that actually
        # look for "is this a screen" / "is this paper", not generic texture.
        self.moire_reject_threshold = 0.45     # score below this => moire pattern detected => SPOOF
        self.print_reject_threshold = 0.45     # score below this => halftone dot pattern      => SPOOF
        self.glare_reject_threshold = 0.45     # score below this => specular screen glare      => SPOOF

        # Secondary/general signals - corroborating only
        self.min_individual_threshold = 0.35

    async def initialize(self):
        self.initialized = True
        logger.info("Anti-spoofing service initialized")

    async def detect_spoofing(self, image_data: np.ndarray) -> Dict:
        """
        Detect if face is real or a presentation attack (screen/photo).

        Returns dict with is_real, confidence, and per-check breakdown.
        """
        try:
            results = []

            # ---- PRIMARY: attack-specific detectors ----

            moire_score = self._detect_screen_moire(image_data)
            results.append({
                'method': 'screen_moire',
                'score': moire_score,
                'passed': moire_score >= self.moire_reject_threshold,
                'critical': True
            })

            glare_score = self._detect_specular_glare(image_data)
            results.append({
                'method': 'specular_glare',
                'score': glare_score,
                'passed': glare_score >= self.glare_reject_threshold,
                'critical': True
            })

            print_score = self._detect_print_artifacts(image_data)
            results.append({
                'method': 'print_halftone',
                'score': print_score,
                'passed': print_score >= self.print_reject_threshold,
                'critical': True
            })

            # ---- SECONDARY: general supporting signals ----

            texture_score = self._analyze_texture(image_data)
            results.append({
                'method': 'texture_analysis',
                'score': texture_score,
                'passed': texture_score >= self.min_individual_threshold,
                'critical': False
            })

            color_score = self._analyze_color_distribution(image_data)
            results.append({
                'method': 'color_analysis',
                'score': color_score,
                'passed': color_score >= self.min_individual_threshold,
                'critical': False
            })

            noise_score = self._analyze_noise_pattern(image_data)
            results.append({
                'method': 'noise_analysis',
                'score': noise_score,
                'passed': noise_score >= self.min_individual_threshold,
                'critical': False
            })

            critical_results = [r for r in results if r['critical']]
            secondary_results = [r for r in results if not r['critical']]

            critical_passed = sum(1 for r in critical_results if r['passed'])
            secondary_passed = sum(1 for r in secondary_results if r['passed'])

            overall_score = sum(r['score'] for r in results) / len(results)

            # STRICT decision logic:
            # - ALL critical (attack-specific) checks must pass. A single
            #   failed critical check (e.g. moire detected) is enough to
            #   reject, regardless of how the secondary checks score.
            # - Secondary checks only matter as a tie-breaker among faces
            #   that already cleared every critical check - they can push
            #   a borderline-real face over the line, they can never rescue
            #   a face that failed a critical check.
            all_critical_passed = (critical_passed == len(critical_results))
            secondary_ok = secondary_passed >= (len(secondary_results) - 1)  # allow 1 secondary miss

            is_real = all_critical_passed and secondary_ok

            result = {
                'is_real': is_real,
                'confidence': round(overall_score, 4),
                'details': {
                    'results': results,
                    'critical_passed': critical_passed,
                    'critical_total': len(critical_results),
                    'secondary_passed': secondary_passed,
                    'secondary_total': len(secondary_results),
                    'verdict': 'REAL' if is_real else 'SPOOF'
                }
            }

            logger.info(
                f"Anti-spoofing: {result['details']['verdict']} "
                f"(critical: {critical_passed}/{len(critical_results)}, "
                f"secondary: {secondary_passed}/{len(secondary_results)}, "
                f"score: {result['confidence']:.4f})"
            )

            return result

        except Exception as e:
            logger.error(f"Spoofing detection failed: {str(e)}", exc_info=True)
            # FAIL-CLOSED: if the detector itself breaks, do not silently
            # wave the request through as REAL. Treat it as unverifiable
            # and let the caller decide (e.g. ask the user to retry) rather
            # than skipping the anti-spoofing gate entirely.
            return {
                'is_real': False,
                'confidence': 0.0,
                'details': {
                    'error': str(e),
                    'verdict': 'ERROR_UNVERIFIABLE'
                }
            }

    # ------------------------------------------------------------------
    # PRIMARY: attack-specific detectors
    # ------------------------------------------------------------------

    def _detect_screen_moire(self, image: np.ndarray) -> float:
        """
        Detect moire interference patterns caused by photographing a
        phone/monitor screen. A screen's sub-pixel grid + the camera's own
        sensor grid interfere and create a periodic pattern that is NOT
        present when photographing a real, continuous-surface face.

        Method: FFT magnitude spectrum -> look for strong, narrow-band
        peaks away from the low-frequency (DC) center. Real faces produce
        a smooth, gradually-decaying spectrum. Screens produce sharp,
        localized high-energy peaks at specific frequencies.
        """
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (512, 512))  # normalize size for consistent freq bins

            # 2D FFT
            f = np.fft.fft2(gray.astype(np.float64))
            fshift = np.fft.fftshift(f)
            magnitude = np.log(np.abs(fshift) + 1)

            h, w = magnitude.shape
            cy, cx = h // 2, w // 2

            # Mask out the DC/low-frequency center (natural image content
            # lives here) - we only care about mid/high frequency energy,
            # which is where moire peaks show up.
            radius_inner = min(h, w) // 16
            radius_outer = min(h, w) // 3

            y, x = np.ogrid[:h, :w]
            dist_from_center = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
            band_mask = (dist_from_center >= radius_inner) & (dist_from_center <= radius_outer)

            band_energy = magnitude[band_mask]

            if band_energy.size == 0:
                return 0.5

            mean_energy = np.mean(band_energy)
            std_energy = np.std(band_energy)
            max_energy = np.max(band_energy)

            # Moire shows up as sharp peaks well above the local mean -
            # a high (max - mean) / std "peakiness" ratio.
            if std_energy < 1e-6:
                peakiness = 0.0
            else:
                peakiness = (max_energy - mean_energy) / std_energy

            # Calibrated: real faces typically show peakiness < ~4-5.
            # Screen photographs commonly show peakiness > ~7-8 due to
            # the regular sub-pixel/refresh grid.
            if peakiness < 4.0:
                return 1.0
            elif peakiness < 5.5:
                return 0.75
            elif peakiness < 7.0:
                return 0.55
            elif peakiness < 8.5:
                return 0.35
            else:
                return 0.15  # strong moire signature -> almost certainly a screen

        except Exception as e:
            logger.warning(f"Moire detection failed: {str(e)}")
            return 0.5

    def _detect_specular_glare(self, image: np.ndarray) -> float:
        """
        Detect sharp specular highlights typical of glossy phone/monitor
        glass. Real skin produces soft, diffuse reflection; glass/screens
        produce small, very bright, hard-edged highlight spots (glare)
        because of the reflective coating.
        """
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Very bright, near-saturated pixels
            bright_mask = (gray > 235).astype(np.uint8) * 255

            # Find connected bright regions
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
                bright_mask, connectivity=8
            )

            total_pixels = gray.size
            hard_glare_pixels = 0
            hard_glare_blobs = 0

            for i in range(1, num_labels):  # skip background label 0
                area = stats[i, cv2.CC_STAT_AREA]
                # Small, isolated, very bright blobs = glare spots.
                # (Large bright regions are more likely just good lighting
                # on skin/background, not glass reflection.)
                if 3 <= area <= (total_pixels * 0.01):
                    hard_glare_blobs += 1
                    hard_glare_pixels += area

            glare_ratio = hard_glare_pixels / total_pixels

            # Calibrated: a handful of small glare blobs from normal room
            # lighting is expected. Many small hard-edged bright blobs, or
            # a glare ratio that's unusually high for the image size,
            # indicates glass/screen reflection.
            if hard_glare_blobs <= 2 and glare_ratio < 0.003:
                return 1.0
            elif hard_glare_blobs <= 4 and glare_ratio < 0.008:
                return 0.8
            elif hard_glare_blobs <= 7 and glare_ratio < 0.015:
                return 0.55
            elif hard_glare_blobs <= 12 and glare_ratio < 0.03:
                return 0.35
            else:
                return 0.15  # heavy specular glare -> likely glass/screen

        except Exception as e:
            logger.warning(f"Specular glare detection failed: {str(e)}")
            return 0.5

    def _detect_print_artifacts(self, image: np.ndarray) -> float:
        """
        Detect halftone/dot-matrix patterns typical of printed photographs.
        Most printers (inkjet, laser, and especially photo-lab prints)
        reproduce continuous-tone images using regularly spaced dot
        patterns, which - like screen moire - produces a periodic signal
        in the frequency domain, but usually at different frequency bands
        and orientations than screen moire (screens are dominated by a
        grid; halftone dots are dominated by diagonal/rosette patterns).
        """
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (512, 512))

            f = np.fft.fft2(gray.astype(np.float64))
            fshift = np.fft.fftshift(f)
            magnitude = np.log(np.abs(fshift) + 1)

            h, w = magnitude.shape
            cy, cx = h // 2, w // 2

            # Halftone dot patterns typically show up as strong energy
            # concentrated along diagonal axes (45/135 degrees) in a
            # mid-frequency ring, distinct from the axis-aligned grid
            # signature of screens.
            y, x = np.ogrid[:h, :w]
            dist_from_center = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
            radius_inner = min(h, w) // 10
            radius_outer = min(h, w) // 2.5

            angle = np.degrees(np.arctan2(y - cy, x - cx)) % 180
            diagonal_mask = (
                ((angle >= 30) & (angle <= 60)) |
                ((angle >= 120) & (angle <= 150))
            )
            band_mask = (dist_from_center >= radius_inner) & (dist_from_center <= radius_outer)

            diagonal_band = magnitude[band_mask & diagonal_mask]
            full_band = magnitude[band_mask]

            if diagonal_band.size == 0 or full_band.size == 0:
                return 0.5

            diagonal_energy = np.mean(diagonal_band)
            overall_band_energy = np.mean(full_band)

            if overall_band_energy < 1e-6:
                ratio = 1.0
            else:
                ratio = diagonal_energy / overall_band_energy

            # Calibrated: real faces distribute frequency energy fairly
            # evenly across angles. A pronounced concentration along the
            # diagonal axes indicates a halftone rosette pattern.
            if ratio < 1.05:
                return 1.0
            elif ratio < 1.15:
                return 0.75
            elif ratio < 1.25:
                return 0.55
            elif ratio < 1.35:
                return 0.35
            else:
                return 0.15  # strong diagonal concentration -> likely halftone print

        except Exception as e:
            logger.warning(f"Print artifact detection failed: {str(e)}")
            return 0.5

    # ------------------------------------------------------------------
    # SECONDARY: general supporting signals (kept from original,
    # unchanged in method - only their role in the final decision changed)
    # ------------------------------------------------------------------

    def _analyze_texture(self, image: np.ndarray) -> float:
        """Analyze texture for screen/replay artifacts (supporting signal)"""
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
        """Analyze color distribution for screen display artifacts (supporting signal)"""
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

    def _analyze_noise_pattern(self, image: np.ndarray) -> float:
        """Analyze noise pattern for digital artifacts (supporting signal)"""
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