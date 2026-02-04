from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tensorflow.keras.models import load_model


@dataclass(frozen=True)
class KerasModelLoader:
    model_path: Path

    def load(self) -> Any:
        """Load and return a Keras model from disk."""
        path = Path(self.model_path)
        print(f"[ModelLoader] Loading model from: {path}")
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        return load_model(str(path))
