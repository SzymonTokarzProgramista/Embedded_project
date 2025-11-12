# src/api/blueprints/camera.py
from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse, Response, JSONResponse
from vision_system.ImageCapture import ImageCapture
import cv2, time, glob, threading, os
from typing import Optional

router = APIRouter(prefix="/camera", tags=["Camera"])

# -----------------------------
# Globalny stan (kamera i detektor)
# -----------------------------
_shared_lock = threading.Lock()
_shared_cap: Optional[ImageCapture] = None
_shared_clients = 0

# próba importu detektora (działa od ręki z Ultralytics .pt; Hailo jako stub)
try:
    from vision_system.ObjectDetection import build_detector, draw_detections
    _HAS_DETECTOR = True
    _DET_IMPORT_ERROR = ""
except Exception as _e:
    build_detector = None  # type: ignore
    draw_detections = None  # type: ignore
    _HAS_DETECTOR = False
    _DET_IMPORT_ERROR = str(_e)

# współdzielony detektor + jego parametry (można nadpisać ENV lub przez query)
_shared_detector = None
_shared_backend = os.environ.get("OD_BACKEND", "cpu")  # 'cpu' (Ultralytics) albo 'hailo'
_shared_model_path = os.environ.get("OD_MODEL", "src/models/monster_v1.0.tflite")
_shared_conf = float(os.environ.get("OD_CONF", "0.25"))
_shared_iou = float(os.environ.get("OD_IOU", "0.45"))

SAVE_DIR = "/Embedded_project/data/captures"
os.makedirs(SAVE_DIR, exist_ok=True)


# -----------------------------
# Pomocnicze
# -----------------------------
def _normalize_device(camera):
    if camera is None or camera == "" or camera == "realsense":
        return None
    if isinstance(camera, str) and camera.startswith("/dev/video"):
        try:
            return int(camera.replace("/dev/video", ""))
        except ValueError:
            return camera
    return camera


def _get_detector(backend: Optional[str] = None,
                  model: Optional[str] = None,
                  conf: Optional[float] = None,
                  iou: Optional[float] = None):
    """
    Zwraca współdzielony obiekt detektora. Jeśli parametry różnią się od obecnych,
    tworzy nowy detektor i podmienia współdzielony.
    """
    global _shared_detector, _shared_backend, _shared_model_path, _shared_conf, _shared_iou

    b = (backend or _shared_backend)
    m = (model or _shared_model_path)
    c = float(conf if conf is not None else _shared_conf)
    i = float(iou if iou is not None else _shared_iou)

    changed = (
        _shared_detector is None
        or b != _shared_backend
        or m != _shared_model_path
        or abs(c - _shared_conf) > 1e-9
        or abs(i - _shared_iou) > 1e-9
    )

    if changed:
        if not _HAS_DETECTOR:
            raise RuntimeError(
                f"Moduł detekcji nie jest dostępny: {_DET_IMPORT_ERROR}.\n"
                "Upewnij się, że dodałeś vision_system/detectors.py oraz wymagane pakiety."
            )
        # utwórz nowy detektor
        det = build_detector(backend=b, model_path=m, conf=c, iou=i)  # type: ignore
        _shared_detector = det
        _shared_backend, _shared_model_path, _shared_conf, _shared_iou = b, m, c, i

    return _shared_detector


# -----------------------------
# Endpoints
# -----------------------------
@router.get("/devices")
def devices():
    info = {"dev_video": sorted(glob.glob("/dev/video*"))}
    try:
        import pyrealsense2 as rs
        ctx = rs.context()
        info["realsense"] = [{
            "name": d.get_info(rs.camera_info.name),
            "serial": d.get_info(rs.camera_info.serial_number),
            "usb_type": d.get_info(rs.camera_info.usb_type_descriptor),
            "product_line": d.get_info(rs.camera_info.product_line),
        } for d in ctx.query_devices()]
    except Exception as e:
        info["realsense_error"] = str(e)
    return JSONResponse(info)


@router.get("/snapshot.jpg")
def snapshot(camera: str | None = None, quality: int = 85, w: int = 640, h: int = 480, fps: int = 30):
    cam_arg = _normalize_device(camera)
    cap = ImageCapture(device=cam_arg, width=int(w), height=int(h), fps=int(fps))
    frame = cap.capture_image()
    cap.release()
    ok, jpg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    return Response(content=jpg.tobytes(), media_type="image/jpeg") if ok else Response(status_code=500)


@router.post("/save")
def save_snapshot(camera: str | None = None, w: int = 1280, h: int = 720, fps: int = 30):
    """Zapisz pojedyncze zdjęcie do katalogu data/captures/"""
    cam_arg = _normalize_device(camera)
    cap = ImageCapture(device=cam_arg, width=int(w), height=int(h), fps=int(fps))
    frame = cap.capture_image()
    cap.release()

    filename = f"capture_{int(time.time())}.jpg"
    path = os.path.join(SAVE_DIR, filename)
    cv2.imwrite(path, frame)
    return {"status": "ok", "file": filename}


@router.get("/stream")
def stream(
    camera: str | None = None,
    fps: int = 15,
    quality: int = 80,
    w: int = 1280,
    h: int = 720,
    # ---- sterowanie detekcją przez URL ----
    detect: int = Query(1, description="0=bez detekcji, 1=z detekcją"),
    backend: Optional[str] = Query(None, description="cpu|hailo (opcjonalnie nadpisuje backend)"),
    model: Optional[str] = Query(None, description="ścieżka do .pt (CPU) lub .hef (Hailo)"),
    conf: Optional[float] = Query(None, description="próg pewności, np. 0.25"),
    iou: Optional[float] = Query(None, description="próg NMS/IOU, np. 0.45"),
):
    """
    Strumień MJPEG. Jeżeli detect=1 (domyślnie), ramki będą opatrzone detekcjami.
    Parametry backend/model/conf/iou mogą być nadpisywane per request.
    """
    global _shared_cap, _shared_clients

    cam_arg = _normalize_device(camera)

    with _shared_lock:
        if _shared_cap is None:
            _shared_cap = ImageCapture(device=cam_arg, width=int(w), height=int(h), fps=int(fps))
        _shared_clients += 1

    def gen():
        global _shared_cap, _shared_clients
        try:
            interval = 1.0 / max(1, int(fps))
            while True:
                with _shared_lock:
                    cap = _shared_cap
                frame = cap.capture_image()  # type: ignore

                if detect:
                    # jeśli mamy moduł detekcji, użyj go; w przeciwnym razie pokaż klatkę z ostrzeżeniem
                    if _HAS_DETECTOR:
                        try:
                            det = _get_detector(backend=backend, model=model, conf=conf, iou=iou)
                            dets = det.detect(frame)  # type: ignore
                            vis = draw_detections(frame.copy(), dets)  # type: ignore
                        except Exception as e:
                            # narysuj ostrzeżenie na ramce, zamiast wywalać stream
                            vis = frame.copy()
                            warn = f"Detection error: {e}"
                            print(f"[WARN] {warn}")
                            cv2.putText(vis, warn, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    else:
                        vis = frame.copy()
                        warn = "Detector module not available"
                        if _DET_IMPORT_ERROR:
                            warn += f": {_DET_IMPORT_ERROR}"
                        cv2.putText(vis, warn, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                else:
                    vis = frame

                ok, jpg = cv2.imencode(".jpg", vis, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
                if ok:
                    yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg.tobytes() + b"\r\n")
                time.sleep(interval)
        finally:
            with _shared_lock:
                _shared_clients -= 1
                if _shared_clients <= 0 and _shared_cap is not None:
                    _shared_cap.release()
                    _shared_cap = None
                    _shared_clients = 0

    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")


@router.post("/detector/reload")
def detector_reload(
    backend: str = "cpu",
    model: str = "models/yolov5s.pt",
    conf: float = 0.25,
    iou: float = 0.45
):
    """
    Wymuś przeładowanie współdzielonego detektora (bez restartu serwera).
    """
    global _shared_detector, _shared_backend, _shared_model_path, _shared_conf, _shared_iou
    with _shared_lock:
        _shared_detector = None
        _shared_backend = backend
        _shared_model_path = model
        _shared_conf = float(conf)
        _shared_iou = float(iou)
    return {
        "status": "ok",
        "backend": _shared_backend,
        "model": _shared_model_path,
        "conf": _shared_conf,
        "iou": _shared_iou
    }
