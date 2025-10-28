# ImageCapture.py
from __future__ import annotations
from typing import Optional, Union
import time, threading
import numpy as np
import cv2

try:
    import pyrealsense2 as rs
    _HAS_RS = True
except Exception:
    rs = None  # type: ignore
    _HAS_RS = False

_RS_LOCK = threading.Lock()  # chroni start/stop RS

def _rs_available() -> bool:
    if not _HAS_RS:
        return False
    try:
        ctx = rs.context()
        return len(ctx.query_devices()) > 0
    except Exception:
        return False

class ImageCapture:
    def __init__(
        self,
        device: Optional[Union[str, int]] = None,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        warmup_frames: int = 5,
    ) -> None:
        self.device = device
        self.width = int(width); self.height = int(height); self.fps = int(fps)
        self.warmup_frames = int(max(0, warmup_frames))
        self._rs_pipeline: Optional["rs.pipeline"] = None
        self._rs_profile = None
        self._cv_cap: Optional["cv2.VideoCapture"] = None
        self._mode = self._decide_mode()
        self._start()
        for _ in range(self.warmup_frames):
            try: _ = self.capture_image()
            except Exception: break

    def __enter__(self) -> "ImageCapture": return self
    def __exit__(self, exc_type, exc, tb) -> None: self.release()

    def capture_image(self) -> np.ndarray:
        if self._mode == "realsense": return self._capture_realsense_bgr()
        if self._mode == "opencv":    return self._capture_opencv_bgr()
        raise RuntimeError("ImageCapture: nieznany tryb działania.")

    def release(self) -> None:
        if self._mode == "realsense": self._stop_realsense()
        elif self._mode == "opencv":  self._stop_opencv()

    def _decide_mode(self) -> str:
        if isinstance(self.device, int): return "opencv"
        if isinstance(self.device, str):
            if self.device.startswith("/dev/video"): return "opencv"
            return "realsense"  # serial albo "realsense"
        if _rs_available(): return "realsense"
        raise RuntimeError(
            "Nie wykryto RealSense (pyrealsense2), a nie podano kamery UVC. "
            "Wywołaj z ?camera=/dev/videoN lub podłącz RealSense."
        )

    def _start(self) -> None:
        if self._mode == "realsense":
            if not _HAS_RS:
                raise ImportError("Brak pyrealsense2. Zainstaluj librealsense + pyrealsense2 lub użyj /dev/videoN.")
            self._start_realsense()
        elif self._mode == "opencv":
            self._start_opencv()
        else:
            raise RuntimeError("ImageCapture: nieznany tryb uruchomienia.")

    # ---------- RealSense ----------
    def _start_realsense(self) -> None:
        attempts = 2
        last_err: Optional[Exception] = None
        with _RS_LOCK:
            for attempt in range(1, attempts + 1):
                try:
                    pipeline = rs.pipeline()
                    cfg = rs.config()
                    if isinstance(self.device, str) and self.device not in ("realsense","") and not self.device.startswith("/dev/video"):
                        cfg.enable_device(self.device)
                    cfg.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
                    profile = pipeline.start(cfg)
                    self._rs_pipeline = pipeline
                    self._rs_profile = profile
                    time.sleep(0.2)
                    return
                except RuntimeError as e:
                    last_err = e
                    if "failed to set power state" in str(e).lower() and attempt < attempts:
                        try:
                            ctx = rs.context()
                            devs = ctx.query_devices()
                            if not devs: raise RuntimeError("Brak urządzeń RealSense.")
                            target = None
                            if isinstance(self.device, str) and self.device not in ("realsense","") and not self.device.startswith("/dev/video"):
                                for d in devs:
                                    if d.get_info(rs.camera_info.serial_number) == self.device:
                                        target = d; break
                            if target is None: target = devs[0]
                            target.hardware_reset()
                            time.sleep(3.0)  # daj czas na ponowne enumerowanie
                            continue
                        except Exception as re:
                            last_err = re
                            break
                    else:
                        break
        raise RuntimeError(
            f"RealSense nie wystartował: {last_err}. "
            "Upewnij się, że tylko jedno żądanie /camera/stream jest aktywne; "
            "sprawdź USB3/kabel; ewentualnie użyj UVC: ?camera=/dev/video2"
        )

    def _capture_realsense_bgr(self) -> np.ndarray:
        assert self._rs_pipeline is not None, "RealSense pipeline nie jest uruchomiony."
        for _ in range(5):
            frames = self._rs_pipeline.wait_for_frames(timeout_ms=2000)
            color = frames.get_color_frame()
            if color:
                bgr = np.asanyarray(color.get_data())
                if bgr is not None and bgr.size: return bgr
        raise RuntimeError("Nie udało się pobrać klatki kolorowej z RealSense.")

    def _stop_realsense(self) -> None:
        with _RS_LOCK:
            if self._rs_pipeline is not None:
                try: self._rs_pipeline.stop()
                except Exception: pass
                self._rs_pipeline = None; self._rs_profile = None

    # ---------- OpenCV / UVC ----------
    def _start_opencv(self) -> None:
        cam_index = self.device
        if isinstance(self.device, str) and self.device.startswith("/dev/video"):
            try: cam_index = int(self.device.replace("/dev/video",""))
            except ValueError: cam_index = 0
        cap = cv2.VideoCapture(cam_index, cv2.CAP_V4L2)
        if not cap.isOpened():
            raise RuntimeError(f"Nie mogę otworzyć kamery {self.device}")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        self._cv_cap = cap
        time.sleep(0.1)

    def _capture_opencv_bgr(self) -> np.ndarray:
        assert self._cv_cap is not None, "OpenCV VideoCapture nie jest otwarty."
        for _ in range(5):
            ok, frame = self._cv_cap.read()
            if ok and frame is not None and frame.size: return frame
            time.sleep(0.01)
        raise RuntimeError("Nie udało się pobrać klatki z kamery UVC/OpenCV.")

    def _stop_opencv(self) -> None:
        if self._cv_cap is not None:
            try: self._cv_cap.release()
            except Exception: pass
            self._cv_cap = None
