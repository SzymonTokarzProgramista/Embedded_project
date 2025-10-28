# src/api/blueprints/camera.py
from fastapi import APIRouter
from fastapi.responses import StreamingResponse, Response, JSONResponse
from vision_system.ImageCapture import ImageCapture
import cv2, time, glob, threading

router = APIRouter(prefix="/camera", tags=["Camera"])

_shared_lock = threading.Lock()
_shared_cap: ImageCapture | None = None
_shared_clients = 0

def _normalize_device(camera):
    if camera is None or camera == "" or camera == "realsense": return None
    if isinstance(camera, str) and camera.startswith("/dev/video"):
        try: return int(camera.replace("/dev/video",""))
        except ValueError: return camera
    return camera

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
    with ImageCapture(device=cam_arg, width=int(w), height=int(h), fps=int(fps)) as cap:
        frame = cap.capture_image()
    ok, jpg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    return Response(content=jpg.tobytes(), media_type="image/jpeg") if ok else Response(status_code=500)

@router.get("/stream")
def stream(camera: str | None = None, fps: int = 15, quality: int = 80, w: int = 640, h: int = 480):
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
                ok, jpg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
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
