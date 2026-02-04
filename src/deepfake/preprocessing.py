from dataclasses import dataclass
import numpy as np
import cv2

@dataclass
class ImagePreprocessor:
    target_size: tuple[int, int] = (96, 96)

    def preprocess_bgr(self, image_bgr: np.ndarray) -> np.ndarray:
        """
        Input: BGR image (OpenCV)
        Output: float32 tensor (1, H, W, 3) normalized to [0,1]
        """
        if image_bgr is None or image_bgr.size == 0:
         raise ValueError("Received empty image for preprocessing")


        resized = cv2.resize(image_bgr, self.target_size, interpolation=cv2.INTER_AREA)
        x = resized.astype(np.float32) / 255.0
        x = np.expand_dims(x, axis=0)
        return x
