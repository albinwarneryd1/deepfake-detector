from dataclasses import dataclass
import cv2
import numpy as np

@dataclass
class FaceCropper:
    scaleFactor: float = 1.1
    minNeighbors: int = 5
    minSize: tuple[int, int] = (60, 60)

    def __post_init__(self):
        self.cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

    def crop_largest_face(self, image_bgr: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        faces = self.cascade.detectMultiScale(
            gray,
            scaleFactor=self.scaleFactor,
            minNeighbors=self.minNeighbors,
            minSize=self.minSize
        )
        if len(faces) == 0:
            return image_bgr

        x, y, w, h = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[0]
        pad = int(0.1 * max(w, h))
        x0 = max(x - pad, 0)
        y0 = max(y - pad, 0)
        x1 = min(x + w + pad, image_bgr.shape[1])
        y1 = min(y + h + pad, image_bgr.shape[0])
        return image_bgr[y0:y1, x0:x1]
