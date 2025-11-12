# src/vision_system/detectors.py (final v5: vectorized boxes, robust scaling, imgsz)
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Dict, Sequence, Union, Tuple

# --- Force-disable Rich markup globally ---
import os as _os
_os.environ.setdefault("RICH_DISABLE", "1")
_os.environ.setdefault("TERM", "dumb")

# Monkeypatch Rich.Console to always use markup=False (in case libs ignore RICH_DISABLE)
try:
    import rich.console as _rc  # type: ignore
    _orig_Console = _rc.Console
    def _patched_Console(*a, **kw):
        kw["markup"] = False
        return _orig_Console(*a, **kw)
    _rc.Console = _patched_Console  # type: ignore
except Exception:
    pass

import cv2
import numpy as np
import os
import json

# ----------------------------------------
# Torch / Ultralytics safe loader fix (PyTorch 2.6+)
# ----------------------------------------
try:
    from torch.serialization import add_safe_globals  # type: ignore
    from ultralytics.nn.tasks import DetectionModel  # type: ignore
    add_safe_globals([DetectionModel])
except Exception:
    pass

# --- Monkeypatch Ultralytics AutoBackend to tolerate non-Python metadata ---
try:
    import importlib
    ab = importlib.import_module("ultralytics.nn.autobackend")
    try:
        import yaml as _yaml  # type: ignore
    except Exception:
        _yaml = None  # type: ignore
    _orig_lit = ab.ast.literal_eval
    def _safe_lit_eval(s: str):
        try:
            return _orig_lit(s)
        except Exception:
            if _yaml is not None:
                try:
                    return _yaml.safe_load(s)
                except Exception:
                    pass
            try:
                return json.loads(s)
            except Exception:
                return {}
    ab.ast.literal_eval = _safe_lit_eval  # type: ignore
except Exception:
    pass

# ----------------------------------------
# Ultralytics YOLO (obsługuje .pt i .tflite)
# ----------------------------------------
from ultralytics import YOLO  # type: ignore


@dataclass
class Detection:
    x1: int
    y1: int
    x2: int
    y2: int
    score: float
    class_id: int
    label: str = ""


def _normalize_names(names: Union[None, Dict[int, str], Sequence[str]]) -> Dict[int, str]:
    if names is None:
        return {}
    if isinstance(names, dict):
        out: Dict[int, str] = {}
        for k, v in names.items():
            try:
                out[int(k)] = str(v)
            except Exception:
                continue
        return out
    if isinstance(names, (list, tuple)):
        return {i: str(n) for i, n in enumerate(names)}
    return {}


def _load_names_from_env() -> Dict[int, str]:
    keys = ("DETECTOR_CLASSES", "CLASSES", "NAMES", "LABELS")
    raw: Optional[str] = None
    for k in keys:
        val = os.getenv(k)
        if val:
            raw = val
            break
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, (list, tuple)):
            return {i: str(x) for i, x in enumerate(parsed) if str(x).strip()}
        if isinstance(parsed, dict):
            return _normalize_names(parsed)
    except Exception:
        pass
    items = [x.strip() for x in raw.split(",") if x.strip()]
    if items:
        return {i: s for i, s in enumerate(items)}
    return {}


def draw_detections(img: np.ndarray, dets: List[Detection]) -> np.ndarray:
    for d in dets:
        color = (0, 255, 0)
        cv2.rectangle(img, (d.x1, d.y1), (d.x2, d.y2), color, 2)
        txt = f"{d.label or d.class_id}:{d.score:.2f}"
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        y1 = d.y1 - th - 6
        if y1 < 0:
            y1 = d.y1 + 2
        cv2.rectangle(img, (d.x1, y1), (d.x1 + tw + 4, y1 + th + 6), color, -1)
        cv2.putText(img, txt, (d.x1 + 2, y1 + th + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    return img


class UltralyticsDetector:
    def __init__(
        self,
        model_path: str,
        conf: float = 0.25,
        iou: float = 0.45,
        labels_path: Optional[str] = None,
        fallback_single_class: Optional[str] = "monster",
        imgsz: Union[int, Tuple[int, int]] = 416,
    ):
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Nie znaleziono modelu: {model_path}")
        try:
            self.model = YOLO(model_path, task="detect")
        except Exception as e:
            raise RuntimeError(
                (
                    f"Nie udało się wczytać modelu '{model_path}'. "
                    f"Dla .tflite upewnij się, że 'tflite-runtime' jest zainstalowany i zgodny z Python/arch. "
                    f"Szczegóły: {e}"
                )
            )
        self.conf = float(conf)
        self.iou = float(iou)
        self.imgsz: Union[int, Tuple[int, int]] = imgsz

        names = _normalize_names(getattr(self.model, "names", {}))
        if labels_path and os.path.isfile(labels_path):
            try:
                with open(labels_path, "r", encoding="utf-8") as f:
                    ls = [ln.strip() for ln in f if ln.strip()]
                names = {i: name for i, name in enumerate(ls)}
            except Exception:
                pass
        env_names = _load_names_from_env()
        if env_names:
            names = env_names
        if not names and fallback_single_class:
            names = {0: str(fallback_single_class)}
        self.names: Dict[int, str] = names

    def detect(self, frame_bgr: np.ndarray) -> List[Detection]:
        try:
            results = self.model(
                frame_bgr,
                conf=self.conf,
                iou=self.iou,
                imgsz=self.imgsz,
                verbose=False,
            )
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"Detection inference failed: {e}")

        H, W = frame_bgr.shape[:2]
        dets: List[Detection] = []

        for r in results:
            boxes = getattr(r, "boxes", None)
            if boxes is None:
                continue

            # weź wszystkie naraz jako NumPy
            try:
                xyxy = boxes.xyxy.detach().cpu().numpy().astype(float)
            except Exception:
                xyxy = boxes.xyxy.cpu().numpy().astype(float)
            try:
                conf = boxes.conf.detach().cpu().numpy().astype(float).reshape(-1)
            except Exception:
                conf = boxes.conf.cpu().numpy().astype(float).reshape(-1)
            try:
                cls  = boxes.cls.detach().cpu().numpy().astype(int).reshape(-1)
            except Exception:
                cls  = boxes.cls.cpu().numpy().astype(int).reshape(-1)

            if xyxy.size == 0:
                continue

            # jeśli wygląda na znormalizowane (max <= 1.0) – przeskaluj
            if np.nanmax(xyxy[:, [0, 2]]) <= 1.0 and np.nanmax(xyxy[:, [1, 3]]) <= 1.0:
                xyxy[:, [0, 2]] *= W
                xyxy[:, [1, 3]] *= H

            # klamrowanie i minimalny rozmiar
            xyxy[:, [0, 2]] = np.clip(np.rint(xyxy[:, [0, 2]]), 0, W - 1)
            xyxy[:, [1, 3]] = np.clip(np.rint(xyxy[:, [1, 3]]), 0, H - 1)
            xyxy = xyxy.astype(int)
            # zapewnij dodatni rozmiar
            w = xyxy[:, 2] - xyxy[:, 0]
            h = xyxy[:, 3] - xyxy[:, 1]
            xyxy[w <= 0, 2] = np.minimum(W - 1, xyxy[w <= 0, 0] + 1)
            xyxy[h <= 0, 3] = np.minimum(H - 1, xyxy[h <= 0, 1] + 1)

            for i in range(xyxy.shape[0]):
                x1, y1, x2, y2 = map(int, xyxy[i])
                score = float(conf[i]) if i < len(conf) else 0.0
                cls_id = int(cls[i]) if i < len(cls) else -1
                label = self.names.get(cls_id, str(cls_id))
                dets.append(Detection(x1, y1, x2, y2, score, cls_id, label))

        return dets


class HailoDetector:
    def __init__(self, hef_path: str, conf: float = 0.25, iou: float = 0.45):
        self.hef_path = hef_path
        self.conf = float(conf)
        self.iou = float(iou)

    def detect(self, frame_bgr: np.ndarray) -> List[Detection]:
        raise NotImplementedError("Wstaw implementację Hailo dla Twojej wersji SDK.")


def build_detector(
    backend: str,
    model_path: str,
    conf: float = 0.25,
    iou: float = 0.45,
    labels_path: Optional[str] = None,
    imgsz: Union[int, Tuple[int, int]] = 416,
):
    backend = (backend or "cpu").lower()
    if backend in ("cpu", "ultralytics", "yolo"):
        return UltralyticsDetector(model_path, conf, iou, labels_path=labels_path, imgsz=imgsz)
    if backend in ("hailo", "hef"):
        return HailoDetector(model_path, conf, iou)
    raise ValueError(f"Nieznany backend: {backend}")


__all__ = [
    "Detection",
    "draw_detections",
    "UltralyticsDetector",
    "HailoDetector",
    "build_detector",
]
