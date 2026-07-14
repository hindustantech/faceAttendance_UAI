import os
import shutil
import numpy as np
from typing import Dict, Optional

from app.config import settings
from app.utils.logger import setup_logger
from models.fas.anti_spoof_predict import AntiSpoofPredict

logger = setup_logger(__name__)

# Default locations -- override via settings/env if your deployment lays
# the models directory out differently.
_DEFAULT_MODELS_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models", "fas")


class AntiSpoofingService:
    """
    Anti-Spoofing Service - MiniFASNet-backed liveness check.
    """

    def __init__(self):
        logger.info("[AntiSpoofingService.__init__] Initializing AntiSpoofingService (MiniFASNet)")
        self.initialized = False

        models_root = getattr(settings, "FAS_MODELS_ROOT", _DEFAULT_MODELS_ROOT)
        self.model_dir = os.path.join(models_root, "weights")
        
        # Define paths
        self.detection_dir = os.path.join(models_root, "detection_model")
        self.caffemodel_path = os.path.join(self.detection_dir, "Widerface-RetinaFace.caffemodel")
        self.deploy_path = os.path.join(self.detection_dir, "deploy.prototxt")
        
        self.device_id = getattr(settings, "FAS_DEVICE_ID", 0)

        # Require the REAL class to win by a comfortable margin
        self.real_confidence_threshold = getattr(settings, "FAS_REAL_CONFIDENCE_THRESHOLD", 0.80)

        self._predictor: Optional[AntiSpoofPredict] = None

        logger.info(f"  Model dir: {self.model_dir}")
        logger.info(f"  Detection dir: {self.detection_dir}")
        logger.info(f"  Real-confidence threshold: {self.real_confidence_threshold}")

    async def initialize(self):
        """Load the RetinaFace detector and all MiniFASNet weights once."""
        logger.info("[AntiSpoofingService.initialize] Loading detector + FAS models...")
        
        # --- FIX START: Auto-create missing folder and find files ---
        
        # 1. Create the missing detection_model folder if it doesn't exist
        if not os.path.exists(self.detection_dir):
            logger.warning(f"  Directory does not exist: {self.detection_dir}. Creating it...")
            try:
                os.makedirs(self.detection_dir)
                logger.info("  Successfully created 'detection_model' folder.")
            except Exception as e:
                raise RuntimeError(f"Failed to create directory {self.detection_dir}. Please create it manually. Error: {e}")

        # 2. Handle missing deploy.prototxt (Check for common locations and move them)
        if not os.path.exists(self.deploy_path):
            # Check if the .txt version exists in the target folder
            txt_path = self.deploy_path + ".txt"
            if os.path.exists(txt_path):
                logger.warning(f"  Found {txt_path} instead of expected {self.deploy_path}. Renaming automatically...")
                try:
                    os.rename(txt_path, self.deploy_path)
                    logger.info("  Successfully renamed file to deploy.prototxt")
                except Exception as rename_e:
                    raise RuntimeError(f"Failed to rename {txt_path} to {self.deploy_path}. Please rename it manually. Error: {rename_e}")
            else:
                # The folder was just created, so the file is definitely missing. Give clear instructions.
                raise FileNotFoundError(
                    f"CRITICAL ERROR: Cannot find 'deploy.prototxt'.\n"
                    f"I just created the folder: {self.detection_dir}\n"
                    f"Please copy 'deploy.prototxt' into that folder and restart the server."
                )

        # 3. Handle missing caffemodel
        if not os.path.exists(self.caffemodel_path):
            # Sometimes users put it in the root fas folder. Let's check and move it.
            possible_root_path = os.path.join(os.path.dirname(self.detection_dir), "Widerface-RetinaFace.caffemodel")
            if os.path.exists(possible_root_path):
                logger.warning(f"  Found caffemodel in wrong location: {possible_root_path}. Moving to {self.caffemodel_path}...")
                try:
                    shutil.move(possible_root_path, self.caffemodel_path)
                    logger.info("  Successfully moved caffemodel into detection_model folder.")
                except Exception as move_e:
                    raise RuntimeError(f"Failed to move caffemodel. Please move it manually. Error: {move_e}")
            else:
                raise FileNotFoundError(
                    f"CRITICAL ERROR: Cannot find 'Widerface-RetinaFace.caffemodel'.\n"
                    f"Please place this file inside: {self.detection_dir}"
                )
            
        # --- FIX END ---

        try:
            self._predictor = AntiSpoofPredict(
                model_dir=self.model_dir,
                caffemodel_path=self.caffemodel_path,
                deploy_path=self.deploy_path,
                device_id=self.device_id,
            )
            self.initialized = True
            logger.info("Anti-spoofing service initialized - MiniFASNet MODE")
        except Exception as e:
            logger.critical(f"[AntiSpoofingService.initialize] FAILED to load FAS models: {e}", exc_info=True)
            raise

    async def detect_spoofing(self, image_data: np.ndarray, motion_frames=None) -> Dict:
        """
        Detect if a face is real or spoofed.
        """
        logger.info("[AntiSpoofingService.detect_spoofing] ========== STARTING DETECTION ==========")
        logger.info(f"[AntiSpoofingService.detect_spoofing] Image shape: {image_data.shape}, dtype: {image_data.dtype}")

        if not self.initialized or self._predictor is None:
            logger.critical("[AntiSpoofingService.detect_spoofing] Service not initialized!")
            return self._fail_safe_result("Service not initialized")

        try:
            bbox = self._predictor.get_bbox(image_data)
            label, confidence, raw_probs = self._predictor.predict_fused(image_data, bbox)

            is_real = (label == 1) and (confidence >= self.real_confidence_threshold)
            attack_type = "NONE" if is_real else "SPOOF_ATTACK"

            logger.info("[AntiSpoofingService.detect_spoofing] ========== RESULTS SUMMARY ==========")
            logger.info(f"  bbox: {bbox}")
            logger.info(f"  raw class probs (summed over models): {raw_probs}")
            logger.info(f"  argmax label: {label} (1 == real) | avg confidence: {confidence:.4f} "
                        f"| required >= {self.real_confidence_threshold}")
            logger.info(f"  Attack type: {attack_type}")
            logger.info(f"  Verdict: {'REAL' if is_real else 'SPOOF'}")

            result = {
                'is_real': is_real,
                'confidence': round(float(confidence), 4),
                'threshold': self.real_confidence_threshold,
                'attack_type': attack_type,
                'details': {
                    'label': label,
                    'raw_class_probs': raw_probs.tolist(),
                    'bbox': bbox,
                    'attack_type': attack_type,
                    'verdict': 'REAL' if is_real else 'SPOOF',
                    'model': 'MiniFASNetV2+MiniFASNetV1SE (fused)',
                }
            }

            logger.info("[AntiSpoofingService.detect_spoofing] ========== DETECTION COMPLETE ==========")
            return result

        except Exception as e:
            logger.error(f"[AntiSpoofingService.detect_spoofing] Detection FAILED: {str(e)}", exc_info=True)
            return self._fail_safe_result(str(e))

    def _fail_safe_result(self, error: str) -> Dict:
        logger.critical("[AntiSpoofingService] FAILING SAFE - allowing real on error")
        return {
            'is_real': True,  # Fail safe - don't block real users on a system error
            'confidence': 0.5,
            'threshold': self.real_confidence_threshold,
            'attack_type': 'UNKNOWN',
            'details': {
                'error': error,
                'attack_type': 'UNKNOWN',
                'verdict': 'PASSED_ON_ERROR',
            }
        }