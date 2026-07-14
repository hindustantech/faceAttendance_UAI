# Adapted from the official Silent-Face-Anti-Spoofing repo (Apache-2.0, Minivision):
#   https://github.com/minivision-ai/Silent-Face-Anti-Spoofing/blob/master/src/anti_spoof_predict.py
#
# What's different from upstream, and why:
#   1. Detection/AntiSpoofPredict take absolute paths as constructor args instead
#      of hardcoded relative paths ("./resources/...") -- the original assumes you
#      always run from the repo root, which breaks under a FastAPI process running
#      from a different working directory.
#   2. Model weights are loaded ONCE in `load_models()` and kept in memory, instead
#      of upstream's `predict()` calling `torch.load(model_path)` from disk on every
#      single call. Fine for a one-shot CLI demo; would add real latency (disk IO +
#      state_dict load) on every attendance punch in a server.
#   3. `predict_fused()` reproduces upstream's test.py fusion logic exactly (crop
#      per-model according to its filename-encoded scale/size, sum softmax outputs,
#      argmax, divide by number of models) but as a reusable method instead of
#      script-level code.
#
# The model architecture (MiniFASNet.py) and crop math (generate_patches.py) are
# used unmodified -- those encode the actual trained behavior and must not drift
# from upstream.

import os
import math
import torch
import numpy as np
import torch.nn.functional as F

from models.fas.model_lib.MiniFASNet import MiniFASNetV1, MiniFASNetV2, MiniFASNetV1SE, MiniFASNetV2SE
from models.fas.generate_patches import CropImage
from models.fas.utility import get_kernel, parse_model_name

import cv2

MODEL_MAPPING = {
    'MiniFASNetV1': MiniFASNetV1,
    'MiniFASNetV2': MiniFASNetV2,
    'MiniFASNetV1SE': MiniFASNetV1SE,
    'MiniFASNetV2SE': MiniFASNetV2SE,
}


class Detection:
    """RetinaFace (Caffe) face detector -- used only to get a bounding box for
    cropping, not for face recognition. Same model upstream uses."""

    def __init__(self, caffemodel_path: str, deploy_path: str):
        self.detector = cv2.dnn.readNetFromCaffe(deploy_path, caffemodel_path)
        self.detector_confidence = 0.6

    def get_bbox(self, img: np.ndarray):
        height, width = img.shape[0], img.shape[1]
        aspect_ratio = width / height
        if img.shape[1] * img.shape[0] >= 192 * 192:
            img = cv2.resize(
                img,
                (int(192 * math.sqrt(aspect_ratio)), int(192 / math.sqrt(aspect_ratio))),
                interpolation=cv2.INTER_LINEAR,
            )

        blob = cv2.dnn.blobFromImage(img, 1, mean=(104, 117, 123))
        self.detector.setInput(blob, 'data')
        out = self.detector.forward('detection_out').squeeze()
        max_conf_index = np.argmax(out[:, 2])
        left, top, right, bottom = (
            out[max_conf_index, 3] * width,
            out[max_conf_index, 4] * height,
            out[max_conf_index, 5] * width,
            out[max_conf_index, 6] * height,
        )
        bbox = [int(left), int(top), int(right - left + 1), int(bottom - top + 1)]
        return bbox


def _to_tensor_no_norm(img_hwc_bgr: np.ndarray) -> torch.Tensor:
    """
    Reproduces upstream's data_io/transform.py ToTensor() -> functional.to_tensor()
    EXACTLY for the numpy-array code path: HWC -> CHW, cast to float, and
    deliberately NOT divided by 255 (upstream comment: "backward compatibility
    ... return img.float().div(255) modify by zkx" -- they removed the /255).
    The trained weights expect this exact (unnormalized, 0-255 float) input;
    silently "fixing" this to a standard /255 normalization would change the
    model's input distribution and quietly wreck accuracy.
    """
    if img_hwc_bgr.ndim == 2:
        img_hwc_bgr = img_hwc_bgr.reshape((img_hwc_bgr.shape[0], img_hwc_bgr.shape[1], 1))
    tensor = torch.from_numpy(img_hwc_bgr.transpose((2, 0, 1))).float()
    return tensor


class AntiSpoofPredict(Detection):
    """
    Loads all MiniFASNet model files found in `model_dir` ONCE, and exposes
    `predict_fused()` which reproduces upstream test.py's multi-model fusion:
    crop per-model according to its filename-encoded scale, sum the softmax
    outputs across models, and argmax.
    """

    def __init__(self, model_dir: str, caffemodel_path: str, deploy_path: str, device_id: int = 0):
        super().__init__(caffemodel_path, deploy_path)
        self.device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
        self.image_cropper = CropImage()
        self._models = []  # list of dicts: {model, scale, out_w, out_h, crop, name}
        self._load_models(model_dir)

    def _load_models(self, model_dir: str):
        for model_name in sorted(os.listdir(model_dir)):
            if not model_name.endswith('.pth'):
                continue
            model_path = os.path.join(model_dir, model_name)
            h_input, w_input, model_type, scale = parse_model_name(model_name)
            kernel_size = get_kernel(h_input, w_input)

            model = MODEL_MAPPING[model_type](conv6_kernel=kernel_size).to(self.device)

            state_dict = torch.load(model_path, map_location=self.device)
            keys = iter(state_dict)
            first_layer_name = next(keys)
            if first_layer_name.find('module.') >= 0:
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                for key, value in state_dict.items():
                    new_state_dict[key[7:]] = value
                model.load_state_dict(new_state_dict)
            else:
                model.load_state_dict(state_dict)

            model.eval()

            self._models.append({
                'name': model_name,
                'model': model,
                'scale': scale,
                'out_w': w_input,
                'out_h': h_input,
                'crop': scale is not None,
            })

        if not self._models:
            raise RuntimeError(f"No .pth model files found in {model_dir}")

    @torch.no_grad()
    def predict_fused(self, org_img: np.ndarray, bbox):
        """
        Returns (label, confidence, raw_probs) where:
          label: argmax class index (int) -- empirically verified against this
                 repo's own sample images: label == 1 means REAL. Labels 0 and 2
                 are both "fake", pooled from different attack styles in the
                 training set; this repo does not document a reliable, stable
                 mapping of 0 vs 2 to "print" vs "screen" specifically, so we
                 don't assert one (see anti_spoofing.py for details).
          confidence: the winning class's summed-softmax score, averaged over
                 the number of fused models (matches upstream test.py: value =
                 prediction[0][label] / num_models).
          raw_probs: the raw (unaveraged) summed 3-class score, for logging.
        """
        prediction = np.zeros((1, 3))
        for m in self._models:
            patch = self.image_cropper.crop(
                org_img=org_img,
                bbox=bbox,
                scale=m['scale'],
                out_w=m['out_w'],
                out_h=m['out_h'],
                crop=m['crop'],
            )
            tensor = _to_tensor_no_norm(patch).unsqueeze(0).to(self.device)
            result = F.softmax(m['model'].forward(tensor), dim=1).cpu().numpy()
            prediction += result

        label = int(np.argmax(prediction))
        confidence = float(prediction[0][label] / len(self._models))
        return label, confidence, prediction
