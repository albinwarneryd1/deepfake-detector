from dataclasses import dataclass
from typing import Optional
import numpy as np

from .preprocessing import ImagePreprocessor

@dataclass(frozen=True)
class PredictionResult:
    label: str
    fake_prob: float
    real_prob: float
    threshold: float
    is_fake: bool
    used_face_crop: bool

class DeepfakePredictor:
    def __init__(self, model, preprocessor: ImagePreprocessor, class_names=("Fake", "Real"), face_cropper=None):
        self.model = model
        self.preprocessor = preprocessor
        self.class_names = class_names
        self.face_cropper = face_cropper

    def predict_bgr(self, image_bgr: np.ndarray, threshold: float = 0.60, use_face_crop: bool = True) -> PredictionResult:
        used_crop = False
        img = image_bgr

        # Optional face crop
        if use_face_crop and self.face_cropper is not None:
            try:
                cropped = self.face_cropper.crop_largest_face(img)
                if cropped is not None and cropped.size > 0:
                    img = cropped
                    used_crop = True
            except Exception:
                # If face-crop fails, just fallback to original image
                used_crop = False
                img = image_bgr

        x = self.preprocessor.preprocess_bgr(img)  # (1, H, W, 3)

        pred = self.model.predict(x, verbose=0)
        pred = np.asarray(pred).reshape(-1)

        # Handle softmax(2) or sigmoid(1)
        if pred.shape[0] == 2:
            fake_prob = float(pred[0])
            real_prob = float(pred[1])
        else:
            fake_prob = float(pred[0])
            real_prob = 1.0 - fake_prob

        is_fake = fake_prob >= threshold
        label = self.class_names[0] if is_fake else self.class_names[1]

        return PredictionResult(
            label=label,
            fake_prob=fake_prob,
            real_prob=real_prob,
            threshold=threshold,
            is_fake=is_fake,
            used_face_crop=used_crop
        )
