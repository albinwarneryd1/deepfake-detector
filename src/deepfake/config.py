from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class AppConfig:
    model_path: Path = Path("models/deepfake_detection_model.h5")
    assets_dir: Path = Path("assets")
    input_size: tuple[int, int] = (96, 96)
    class_names: tuple[str, str] = ("Fake", "Real")
