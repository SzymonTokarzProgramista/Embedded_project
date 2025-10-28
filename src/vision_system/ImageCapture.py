# ImageCapture.py
"""
Capture obrazu z Intel RealSense D435i (kolor) z opcjonalnym fallbackiem do OpenCV.
- Domyślnie używa pyrealsense2 (pipeline RealSense), zwracając klatkę BGR (numpy array).
- Jeśli przekażesz urządzenie w stylu /dev/video0 lub indeks kamery, użyje OpenCV.
- API zgodne z wcześniejszymi wywołaniami: capture_image() i release().

Wymagania:
- pyrealsense2 (do RealSense)
- numpy, opencv-python(-headless)
"""

from __future__ import annotations
from typing import Optional, Union
import time
import numpy as np
import cv2

try:
    import pyrealsense2 as rs  # RealSense SDK
    _HAS_RS = True
except Exception:
    rs = None  # type: ignore
    _HAS_RS = False

def _rs_available() -> bool:
    if not _HAS_RS:
        return False
    try:
        ctx = rs.context()
        return len(ctx.query_devices()) > 0
    except Exception:
        return False


class ImageCapture:
    """
    Ujednolicony interfejs do przechwytywania klatek:
    - RealSense (pyrealsense2): gdy device=None lub "realsense" albo serial kamery.
    - OpenCV: gdy device to int lub ścieżka typu "/dev/video0".

    Przykłady:
        # RealSense (domyślnie)
        cap = ImageCapture()
        frame = cap.capture_image()

        # RealSense po serialu
        cap = ImageCapture(device="f0123456")  # serial_number
        frame = cap.capture_image()

        # Fallback do UVC/OpenCV
        cap = ImageCapture(device="/dev/video0")

    Parametry:
        device: None/"realsense"/serial albo /dev/videoN lub indeks int
        width, height, fps: parametry kolorowego streamu
        warmup_frames: ile klatek odrzucić na starcie (AE/AGC się ustabilizuje)
    """

    def __init__(
        self,
        device: Optional[Union[str, int]] = None,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        warmup_frames: int = 5,
    ) -> None:
        self.device = device
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self.warmup_frames = int(max(0, warmup_frames))

        # RealSense
        self._rs_pipeline: Optional["rs.pipeline"] = None
        self._rs_profile = None

        # OpenCV
        self._cv_cap: Optional["cv2.VideoCapture"] = None

        self._mode = self._decide_mode()
        self._start()

    # -------- public API --------

    def capture_image(self) -> np.ndarray:
        """
        Zwraca pojedynczą klatkę BGR (numpy array) z RealSense lub OpenCV.
        W razie chwilowych problemów spróbuje kilka razy zanim podniesie wyjątek.
        """
        if self._mode == "realsense":
            return self._capture_realsense_bgr()
        elif self._mode == "opencv":
            return self._capture_opencv_bgr()
        else:
            raise RuntimeError("ImageCapture: nieznany tryb działania.")

    def release(self) -> None:
        """Zwolnij zasoby."""
        if self._mode == "realsense":
            self._stop_realsense()
        elif self._mode == "opencv":
            self._stop_opencv()

    # -------- internal: mode selection --------

    def _decide_mode(self) -> str:
        # Jeśli user podał ścieżkę /dev/videoN lub int -> OpenCV
        if isinstance(self.device, int):
            return "opencv"
        if isinstance(self.device, str):
            if self.device.startswith("/dev/video"):
                return "opencv"
            # serial albo 'realsense'
            return "realsense"
        # None -> preferuj RealSense, ale tylko gdy jest dostępny
        if _rs_available():
            return "realsense"
        # brak RealSense i brak wskazanego /dev/video => nie zgaduj "0"
        raise RuntimeError(
            "Nie wykryto urządzenia RealSense (pyrealsense2), a nie podano kamery UVC. "
            "Wywołaj z ?camera=/dev/videoN lub podłącz RealSense."
        )
    

    def _start(self) -> None:
        if self._mode == "realsense":
            if not _HAS_RS:
                raise ImportError(
                    "pyrealsense2 nie jest zainstalowane. "
                    "Zainstaluj librealsense + pyrealsense2 albo użyj /dev/videoN (OpenCV)."
                )
            self._start_realsense()
        elif self._mode == "opencv":
            self._start_opencv()
        else:
            raise RuntimeError("ImageCapture: nieznany tryb uruchomienia.")

        # Wstępne „rozgrzanie” ekspozycji (opcjonalnie)
        for _ in range(self.warmup_frames):
            try:
                _ = self.capture_image()
            except Exception:
                break

    # -------- RealSense pipeline --------

    def _start_realsense(self) -> None:
        pipeline = rs.pipeline()
        cfg = rs.config()

        # Jeśli device to serial_number, przypnij urządzenie:
        if isinstance(self.device, str) and self.device not in ("realsense", "") and not self.device.startswith("/dev/video"):
            cfg.enable_device(self.device)

        # Strumień kolorowy w BGR, żeby pasowało do OpenCV bez konwersji
        cfg.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)

        # Start
        profile = pipeline.start(cfg)

        # Zachowaj do użycia
        self._rs_pipeline = pipeline
        self._rs_profile = profile

        # Krótka pauza, żeby strumień się ustabilizował
        time.sleep(0.2)

    def _capture_realsense_bgr(self) -> np.ndarray:
        assert self._rs_pipeline is not None, "RealSense pipeline nie jest uruchomiony."
        retries = 5
        for _ in range(retries):
            frames = self._rs_pipeline.wait_for_frames(timeout_ms=2000)
            color = frames.get_color_frame()
            if color:
                bgr = np.asanyarray(color.get_data())  # już w BGR (rs.format.bgr8)
                if bgr is not None and bgr.size:
                    return bgr
            # jeśli brak klatki, spróbuj ponownie
        raise RuntimeError("Nie udało się pobrać klatki kolorowej z RealSense.")

    def _stop_realsense(self) -> None:
        if self._rs_pipeline is not None:
            try:
                self._rs_pipeline.stop()
            except Exception:
                pass
            self._rs_pipeline = None
            self._rs_profile = None

    # -------- OpenCV fallback --------

    def _start_opencv(self) -> None:
        # Zamień "/dev/videoN" na indeks N
        cam_index = self.device
        if isinstance(self.device, str) and self.device.startswith("/dev/video"):
            try:
                cam_index = int(self.device.replace("/dev/video", ""))
            except ValueError:
                cam_index = 0

        cap = cv2.VideoCapture(cam_index, cv2.CAP_V4L2)
        if not cap.isOpened():
            raise RuntimeError(f"Nie mogę otworzyć kamery {self.device}")

        # Ustaw parametry, jeśli urządzenie wspiera
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, self.fps)

        self._cv_cap = cap
        time.sleep(0.1)

    def _capture_opencv_bgr(self) -> np.ndarray:
        assert self._cv_cap is not None, "OpenCV VideoCapture nie jest otwarty."
        retries = 5
        for _ in range(retries):
            ok, frame = self._cv_cap.read()
            if ok and frame is not None and frame.size:
                # frame już w BGR (OpenCV)
                return frame
            time.sleep(0.01)
        raise RuntimeError("Nie udało się pobrać klatki z kamery UVC/OpenCV.")

    def _stop_opencv(self) -> None:
        if self._cv_cap is not None:
            try:
                self._cv_cap.release()
            except Exception:
                pass
            self._cv_cap = None
