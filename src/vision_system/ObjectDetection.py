# src/vision_system/detectors.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional
import cv2
import numpy as np
import os

# ----------------------------------------
# Torch / Ultralytics safe loader fix (PyTorch 2.6+)
# Umożliwia bezpieczne wczytywanie .pt z klasą DetectionModel.
# Nie wpływa na .tflite, ale nie przeszkadza.
# ----------------------------------------
try:
    from torch.serialization import add_safe_globals
    from ultralytics.nn.tasks import DetectionModel
    add_safe_globals([DetectionModel])
except Exception:
    pass

# ----------------------------------------
# Ultralytics YOLO (obsługuje .pt i .tflite)
# ----------------------------------------
try:
    from ultralytics import YOLO
    _HAS_ULTRA = True
except Exception:
    _HAS_ULTRA = False


@dataclass
class Detection:
    x1: int
    y1: int
    x2: int
    y2: int
    score: float
    class_id: int
    label: str = ""


def draw_detections(img: np.ndarray, dets: List[Detection]) -> np.ndarray:
    """Rysuje prostokąty i etykiety detekcji na ramce."""
    for d in dets:
        color = (0, 255, 0)
        cv2.rectangle(img, (d.x1, d.y1), (d.x2, d.y2), color, 2)
        txt = f"{d.label or d.class_id}:{d.score:.2f}"
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img, (d.x1, d.y1 - th - 6), (d.x1 + tw + 4, d.y1), color, -1)
        cv2.putText(img, txt, (d.x1 + 2, d.y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    return img


# =================================================================
# Ultralytics backend (.pt oraz .tflite) z opcjonalnym labels.txt
# =================================================================
class UltralyticsDetector:
    """
    Detekcja obiektów przy użyciu Ultralytics YOLO.
    Obsługiwane formaty modeli:
      - .pt      (PyTorch)
      - .tflite  (TensorFlow Lite; wymagany 'tflite-runtime' dla arch/OS)
    Możesz nadpisać nazwy klas podając ścieżkę do pliku labels.txt
    (jedna etykieta na linię).
    """

    def __init__(self, model_path: str, conf: float = 0.25, iou: float = 0.45, labels_path: Optional[str] = None):
        if not _HAS_ULTRA:
            raise RuntimeError(
                "Pakiet 'ultralytics' nie jest zainstalowany. Zainstaluj go, aby używać YOLO (.pt/.tflite)."
            )
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Nie znaleziono modelu: {model_path}")

        # task='detect' → nie polegaj na zgadywaniu po YAML/metadanych, co bywa źródłem błędów
        try:
            self.model = YOLO(model_path, task='detect')
        except Exception as e:
            raise RuntimeError(
                f"Nie udało się wczytać modelu '{model_path}'. "
                f"Dla .tflite upewnij się, że 'tflite-runtime' jest zainstalowany i zgodny z Python/arch. "
                f"Szczegóły: {e}"
            )

        self.conf = float(conf)
        self.iou = float(iou)

        # 1) spróbuj nazwy klas z modelu
        names = {}
        try:
            names = self.model.names if hasattr(self.model, "names") else {}
        except Exception:
            names = {}

        # 2) jeśli podano labels.txt, nadpisz nazwy
        if labels_path and os.path.isfile(labels_path):
            with open(labels_path, "r", encoding="utf-8") as f:
                ls = [ln.strip() for ln in f if ln.strip()]
            names = {i: name for i, name in enumerate(ls)}

        self.names = names

    def detect(self, frame_bgr: np.ndarray) -> List[Detection]:
        """Wykonuje detekcję na ramce (OpenCV BGR)."""
        results = self.model(frame_bgr, conf=self.conf, iou=self.iou, verbose=False)
        dets: List[Detection] = []
        for r in results:
            if getattr(r, "boxes", None) is None:
                continue
            for b in r.boxes:
                # współrzędne XYXY
                xyxy = b.xyxy[0].cpu().numpy().astype(int)
                x1, y1, x2, y2 = map(int, xyxy)
                # score / cls
                score = float(b.conf[0].item()) if getattr(b, "conf", None) is not None else 0.0
                cls = int(b.cls[0].item()) if getattr(b, "cls", None) is not None else -1
                # label
                if isinstance(self.names, dict):
                    label = self.names.get(cls, str(cls))
                elif isinstance(self.names, (list, tuple)) and 0 <= cls < len(self.names):
                    label = str(self.names[cls])
                else:
                    label = str(cls)
                dets.append(Detection(x1, y1, x2, y2, score, cls, label))
        return dets


# ======================================================
# Hailo backend (stub – do wypełnienia w Twoim SDK)
# ======================================================
class HailoDetector:
    """Detektor działający na urządzeniu Hailo (.hef)."""

    def __init__(self, hef_path: str, conf: float = 0.25, iou: float = 0.45):
        self.hef_path = hef_path
        self.conf = float(conf)
        self.iou = float(iou)
        # TODO: zainicjuj HailoRT / PySDK zgodnie z Twoją wersją SDK
        # np. hailo.Hef(...) lub hailo_platform.InferVDevice(...)

    def detect(self, frame_bgr: np.ndarray) -> List[Detection]:
        """Wykonuje detekcję z użyciem Hailo (.hef)."""
        # TODO: preprocess -> infer -> postprocess (NMS jeśli potrzeba)
        # Zwróć listę Detection (x1, y1, x2, y2, score, class_id, label)
        raise NotImplementedError("Wstaw implementację Hailo dla Twojej wersji SDK.")


# ======================================================
# Fabryka detektorów
# ======================================================
def build_detector(
    backend: str,
    model_path: str,
    conf: float = 0.25,
    iou: float = 0.45,
    labels_path: Optional[str] = None,
):
    """
    Tworzy odpowiedni detektor w zależności od backendu.
      backend = 'cpu' / 'ultralytics' / 'yolo'  -> Ultralytics (obsługuje .pt i .tflite)
      backend = 'hailo' / 'hef'                 -> Hailo (.hef, do uzupełnienia)
    """
    backend = (backend or "cpu").lower()
    if backend in ("cpu", "ultralytics", "yolo"):
        return UltralyticsDetector(model_path, conf, iou, labels_path=labels_path)
    elif backend in ("hailo", "hef"):
        return HailoDetector(model_path, conf, iou)
    else:
        raise ValueError(f"Nieznany backend: {backend}")
