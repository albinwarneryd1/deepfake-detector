import numpy as np
import cv2

def decode_uploaded_image(file_bytes: bytes) -> np.ndarray:
    """
    Decode uploaded bytes -> BGR image (OpenCV)
    """
    arr = np.frombuffer(file_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image. Unsupported format or corrupted file.")
    return img

def bgr_to_rgb(image_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
