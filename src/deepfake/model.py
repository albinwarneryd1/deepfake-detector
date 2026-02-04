from dataclasses import dataclass
from pathlib import Path
from typing import Union

import tensorflow as tf

PathLike = Union[str, Path]

@dataclass
class KerasModelLoader:
    model_path: Path

    def __init__(self, model_path: PathLike):
        self.model_path = Path(model_path)

def load(self) -> tf.keras.Model:
    if not self.model_path.exists():
        raise FileNotFoundError(
            f"Model file not found at {self.model_path}. "
            "Run `python train.py` first."
        )
    return tf.keras.models.load_model(str(self.model_path))

print(f"[ModelLoader] Loading model from {self.model_path}")
