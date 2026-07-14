# app/services/anti_spoofing.py (FIXED VERSION)

import os
import shutil
import numpy as np
from typing import Dict, Optional
from pathlib import Path

from app.config import settings
from app.utils.logger import setup_logger
from models.fas.anti_spoof_predict import AntiSpoofPredict

logger = setup_logger(__name__)

class AntiSpoofingService:
    """Anti-Spoofing Service with proper singleton pattern"""
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        # Only initialize once
        if not hasattr(self, '_config_loaded'):
            self._config_loaded = True
            self._predictor: Optional[AntiSpoofPredict] = None
            self._load_config()
    
    def _load_config(self):
        """Load configuration and validate paths"""
        models_root = getattr(settings, "FAS_MODELS_ROOT", None)
        if not models_root:
            # Default to models/fas relative to project root
            models_root = os.path.join(
                Path(__file__).parent.parent.parent,
                "models", "fas"
            )
        
        self.model_dir = os.path.join(models_root, "weights")
        self.detection_dir = os.path.join(models_root, "detection_model")
        self.caffemodel_path = os.path.join(self.detection_dir, "Widerface-RetinaFace.caffemodel")
        self.deploy_path = os.path.join(self.detection_dir, "deploy.prototxt")
        
        self.device_id = getattr(settings, "FAS_DEVICE_ID", 0)
        self.real_confidence_threshold = getattr(settings, "FAS_REAL_CONFIDENCE_THRESHOLD", 0.80)
        self.min_confidence_threshold = getattr(settings, "FAS_MIN_CONFIDENCE_THRESHOLD", 0.35)
        
        logger.info(f"Anti-spoofing config loaded:")
        logger.info(f"  Model dir: {self.model_dir}")
        logger.info(f"  Detection dir: {self.detection_dir}")
        logger.info(f"  Real confidence threshold: {self.real_confidence_threshold}")
    
    async def initialize(self) -> bool:
        """Initialize the anti-spoofing service"""
        if self._initialized and self._predictor is not None:
            logger.info("Anti-spoofing service already initialized")
            return True
        
        try:
            # Validate and prepare directories
            self._prepare_directories()
            
            # Initialize predictor
            self._predictor = AntiSpoofPredict(
                model_dir=self.model_dir,
                caffemodel_path=self.caffemodel_path,
                deploy_path=self.deploy_path,
                device_id=self.device_id,
            )
            
            self._initialized = True
            logger.info("Anti-spoofing service initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize anti-spoofing: {e}", exc_info=True)
            self._predictor = None
            self._initialized = False
            return False
    
    def _prepare_directories(self):
        """Ensure all required directories and files exist"""
        # Create detection_model directory if missing
        os.makedirs(self.detection_dir, exist_ok=True)
        
        # Check for required model files
        required_files = {
            'deploy.prototxt': self.deploy_path,
            'caffemodel': self.caffemodel_path
        }
        
        missing_files = []
        for name, path in required_files.items():
            if not os.path.exists(path):
                # Check with .txt extension
                if os.path.exists(path + ".txt"):
                    shutil.move(path + ".txt", path)
                else:
                    missing_files.append(f"{name} at {path}")
        
        if missing_files:
            raise FileNotFoundError(
                f"Missing required anti-spoofing model files:\n" + 
                "\n".join(f"  - {f}" for f in missing_files) +
                "\nPlease ensure these files are present before starting the service."
            )
    
    def is_initialized(self) -> bool:
        """Check if service is properly initialized"""
        return self._initialized and self._predictor is not None
    
    async def detect_spoofing(self, image_data: np.ndarray) -> Dict:
        """Detect if a face is real or spoofed"""
        logger.info("Starting anti-spoofing detection")
        
        if not self.is_initialized():
            logger.error("Anti-spoofing service not initialized")
            return self._create_error_result("Service not initialized")
        
        try:
            # Validate input
            if image_data is None or image_data.size == 0:
                return self._create_error_result("Invalid image data")
            
            # Get face bounding box
            bbox = self._predictor.get_bbox(image_data)
            if bbox is None:
                logger.warning("No face detected in image")
                return {
                    'is_real': False,
                    'confidence': 0.0,
                    'threshold': self.real_confidence_threshold,
                    'attack_type': 'NO_FACE_DETECTED',
                    'details': {
                        'label': -1,
                        'raw_class_probs': [],
                        'bbox': None,
                        'verdict': 'SPOOF',
                        'model': 'MiniFASNet'
                    }
                }
            
            # Get prediction
            label, confidence, raw_probs = self._predictor.predict_fused(image_data, bbox)
            
            # Determine if real or spoof
            is_real = (label == 1) and (confidence >= self.real_confidence_threshold)
            attack_type = "NONE" if is_real else "SPOOF_ATTACK"
            
            logger.info(f"Anti-spoofing result: label={label}, confidence={confidence:.4f}, is_real={is_real}")
            
            return {
                'is_real': is_real,
                'confidence': round(float(confidence), 4),
                'threshold': self.real_confidence_threshold,
                'attack_type': attack_type,
                'details': {
                    'label': int(label),
                    'raw_class_probs': raw_probs.tolist() if hasattr(raw_probs, 'tolist') else list(raw_probs),
                    'bbox': bbox,
                    'verdict': 'REAL' if is_real else 'SPOOF',
                    'model': 'MiniFASNetV2+MiniFASNetV1SE (fused)'
                }
            }
            
        except Exception as e:
            logger.error(f"Anti-spoofing detection failed: {e}", exc_info=True)
            return self._create_error_result(str(e))
    
    def _create_error_result(self, error: str) -> Dict:
        """Create error result - FAIL CLOSED for security"""
        return {
            'is_real': False,  # Changed to False for security
            'confidence': 0.0,
            'threshold': self.real_confidence_threshold,
            'attack_type': 'DETECTION_ERROR',
            'details': {
                'error': error,
                'verdict': 'ERROR',
                'model': 'MiniFASNet'
            }
        }