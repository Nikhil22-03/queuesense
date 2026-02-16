"""Person detection with an Ultralytics model and ByteTrack.

The person class id is read from the model name map. COCO detectors usually
use 0, and this code still looks up the name so a custom model can differ.
Ultralytics and the weights are imported only when a detector is constructed,
which keeps unit tests from downloading anything.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from queuesense.config import project_root
from queuesense.tracker import Detection, detections_from_result, reset_model_trackers


def tracker_config_path() -> str:
    """Project ByteTrack file. Falls back to Ultralytics' built-in name if it is missing."""
    path = project_root() / "configs" / "bytetrack.yaml"
    return str(path) if path.is_file() else "bytetrack.yaml"


def resolve_person_class_ids(names) -> list[int]:
    """Return every class id whose label is 'person', using the model's own names."""
    pairs: list[tuple[int, str]] = []
    if isinstance(names, dict):
        for key, value in names.items():
            pairs.append((int(key), str(value)))
    else:
        pairs = [(index, str(value)) for index, value in enumerate(names)]
    person_ids = [class_id for class_id, name in pairs if name.strip().lower() == "person"]
    if not person_ids:
        available = ", ".join(f"{class_id}:{name}" for class_id, name in pairs) or "(none)"
        raise RuntimeError(f"This model has no class named 'person'. Classes: {available}")
    return person_ids


def resolve_device(preference: str) -> str:
    """Map auto/cpu/cuda onto a device string Ultralytics accepts."""
    choice = preference.strip().lower()
    if choice not in {"auto", "cpu", "cuda"}:
        raise ValueError("Device must be auto, cpu, or cuda.")
    cuda = False
    try:
        import torch

        cuda = bool(torch.cuda.is_available())
    except Exception:
        cuda = False
    if choice == "cpu":
        return "cpu"
    if choice == "cuda":
        if not cuda:
            raise RuntimeError(
                "CUDA was requested, but this PyTorch install cannot see a GPU. "
                "Choose CPU, or install a CUDA build of PyTorch."
            )
        return "cuda"
    return "cuda" if cuda else "cpu"


class PersonDetector:
    """One loaded model. Call reset_tracker() before every independent video."""

    def __init__(self, model_path: str, device: str, image_size: int = 640) -> None:
        self.requested_model_path = model_path
        self.device = resolve_device(device)
        self.image_size = int(image_size)
        self.model, self.resolved_model_path, self.weight_source = self._open_model(model_path)
        self.person_class_ids = resolve_person_class_ids(self.model.names)
        import ultralytics

        self.ultralytics_version = ultralytics.__version__

    def reset_tracker(self) -> None:
        reset_model_trackers(self.model)

    def track_frame(self, frame, confidence: float) -> list[Detection]:
        results = self.model.track(
            source=frame,
            persist=True,
            tracker=tracker_config_path(),
            classes=self.person_class_ids,
            conf=float(confidence),
            iou=0.7,
            imgsz=self.image_size,
            device=self.device,
            verbose=False,
            stream=False,
        )
        if not results:
            return []
        return detections_from_result(results[0], set(self.person_class_ids))

    @property
    def metadata(self) -> dict:
        names = {str(key): str(value) for key, value in dict(self.model.names).items()}
        return {
            "requested_model_path": self.requested_model_path,
            "resolved_model_path": self.resolved_model_path,
            "weight_source": self.weight_source,
            "ultralytics_version": self.ultralytics_version,
            "class_names": names,
            "person_class_ids": list(self.person_class_ids),
            "device": self.device,
            "image_size": self.image_size,
            "tracker": tracker_config_path(),
        }

    def _open_model(self, model_path: str):
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "Ultralytics is not installed. Install requirements.txt before running analysis."
            ) from exc
        requested = Path(model_path)
        models_dir = project_root() / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        try:
            if requested.is_file():
                chosen = requested.resolve()
                return YOLO(str(chosen)), str(chosen), "local file"
            cached = models_dir / requested.name
            if cached.is_file():
                return YOLO(str(cached)), str(cached.resolve()), "models directory"
            before = {path.resolve() for path in Path.cwd().glob("*.pt")}
            model = YOLO(requested.name)
            after = {path.resolve() for path in Path.cwd().glob("*.pt")}
            downloaded = sorted(after - before)
            if cached.is_file():
                return model, str(cached.resolve()), "models directory"
            if downloaded:
                shutil.copy2(downloaded[0], cached)
                try:
                    downloaded[0].unlink()
                except OSError:
                    pass
                return model, str(cached.resolve()), "downloaded from Ultralytics into models/"
            return model, requested.name, "loaded by Ultralytics"
        except RuntimeError:
            raise
        except Exception as exc:
            raise RuntimeError(
                f"Could not load model '{model_path}'. "
                "The first use of an official name such as yolo26n.pt downloads weights from Ultralytics "
                "and needs a network connection. "
                f"Original error: {exc}"
            ) from exc
