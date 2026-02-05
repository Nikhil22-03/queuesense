"""Run settings, paths, and lightweight environment details."""

from __future__ import annotations

import json
import platform
from dataclasses import asdict, dataclass, fields
from pathlib import Path


def project_root() -> Path:
    """Return the repository root whether the package is imported from src or an editable install."""
    here = Path(__file__).resolve()
    candidate = here.parents[2]
    if (candidate / "configs" / "default.json").is_file():
        return candidate
    cwd = Path.cwd()
    if (cwd / "configs" / "default.json").is_file():
        return cwd
    return candidate


@dataclass(frozen=True)
class Timing:
    """Video-time durations for the queue state machine. Units are seconds."""

    join_confirm_sec: float = 1.5
    service_confirm_sec: float = 1.0
    exit_confirm_sec: float = 2.0
    missing_grace_sec: float = 3.0


@dataclass
class Settings:
    """Everything that changes the result of a run, stored with that run."""

    model_path: str = "yolo26n.pt"
    confidence: float = 0.25
    device: str = "auto"
    frame_stride: int = 1
    image_size: int = 640
    join_confirm_sec: float = 1.5
    service_confirm_sec: float = 1.0
    exit_confirm_sec: float = 2.0
    missing_grace_sec: float = 3.0
    min_service_samples: int = 3

    def __post_init__(self) -> None:
        self.model_path = str(self.model_path).strip()
        self.confidence = float(self.confidence)
        self.device = str(self.device).strip().lower()
        self.frame_stride = int(self.frame_stride)
        self.image_size = int(self.image_size)
        self.join_confirm_sec = float(self.join_confirm_sec)
        self.service_confirm_sec = float(self.service_confirm_sec)
        self.exit_confirm_sec = float(self.exit_confirm_sec)
        self.missing_grace_sec = float(self.missing_grace_sec)
        self.min_service_samples = int(self.min_service_samples)

    def validate(self) -> None:
        if not self.model_path:
            raise ValueError("Model path is empty.")
        if not 0.0 < self.confidence <= 1.0:
            raise ValueError("Confidence must be greater than 0 and at most 1.")
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("Device must be auto, cpu, or cuda.")
        if self.frame_stride < 1:
            raise ValueError("Frame stride must be an integer of at least 1.")
        if self.image_size % 32 != 0 or not 160 <= self.image_size <= 1280:
            raise ValueError("Image size must be a multiple of 32 between 160 and 1280.")
        for name in (
            "join_confirm_sec",
            "service_confirm_sec",
            "exit_confirm_sec",
            "missing_grace_sec",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be zero or greater.")
        if self.min_service_samples < 1:
            raise ValueError("The estimator needs a minimum sample count of at least 1.")

    def timing(self) -> Timing:
        return Timing(
            join_confirm_sec=self.join_confirm_sec,
            service_confirm_sec=self.service_confirm_sec,
            exit_confirm_sec=self.exit_confirm_sec,
            missing_grace_sec=self.missing_grace_sec,
        )

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict | None) -> Settings:
        incoming = dict(data or {})
        known = {item.name for item in fields(cls)}
        return cls(**{key: value for key, value in incoming.items() if key in known})


def load_settings(path: Path | None = None) -> Settings:
    config_path = Path(path) if path else project_root() / "configs" / "default.json"
    payload: dict = {}
    if config_path.is_file():
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    settings = Settings.from_dict(payload)
    settings.validate()
    return settings


def hardware_info(selected_device: str) -> dict:
    """Describe the machine that produced a run. Missing optional libraries stay explicit."""
    info: dict = {
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "python": platform.python_version(),
        "selected_device": selected_device,
        "torch": None,
        "cuda_available": False,
        "gpu_name": None,
        "opencv": None,
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        if info["cuda_available"]:
            info["gpu_name"] = torch.cuda.get_device_name(0)
    except Exception as exc:  # pragma: no cover - depends on the local install
        info["torch_note"] = str(exc)
    try:
        import cv2

        info["opencv"] = cv2.__version__
    except Exception as exc:  # pragma: no cover
        info["opencv_note"] = str(exc)
    return info
