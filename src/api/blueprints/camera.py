# src/api/blueprints/camera.py
from fastapi import APIRouter, HTTPException, Response
from fastapi.responses import StreamingResponse, JSONResponse
from vision_system.ImageCapture import ImageCapture
import cv2, time, glob

router = APIRouter(prefix="/camera", tags=["Camera"])

def _normalize_device(camera):
    # None => użyj RealSense (pyrealsense2)
    if camera is None or camera == "" or camera == "realsense":
        return None
    # /dev/videoN => zamień na int N
    if isinstance(camera, str) and camera.startswith("/dev/video"):
        try:
            return int(camera.replace("/dev/video", ""))
        except ValueError:
            return camera
    # może być int lub serial RS
    return camera

@router.get("/devices", summary="Diagnostyka: dostępne urządzenia")
def list_devices():
    info = {"dev_video": sorted(glob.glob("/dev/video*"))}
    try:
        import pyrealsense2 as rs
        ctx = rs.context()
        devices = []
        for d in ctx.query_devices():
            devices.append({
                "name": d.get_info(rs.camera_info.name),
                "serial": d.get_info(rs.camera_info.serial_number),
                "usb_type": d.get_info(rs.camera_info.usb_type_descriptor),
                "product_line": d.get_info(rs.camera_info.product_line),
            })
        info["realsense"] = devices
    except Exception as e:
        info["realsense_error"] = str(e)
    return JSONResponse(info)

@router.get("/snapshot.jpg", summary="Pojedyncza klatka (JPEG)")
def snapshot(camera: str | None = None, quality: int = 85, w: int = 640, h: int = 480, fps: int = 30):
    cam_arg = _normalize_device(camera)
    cap = ImageCapture(device=cam_arg, width=int(w), height=int(h), fps=int(fps))
    try:
        frame = cap.capture_image()
    finally:
        cap.release()
    ok, jpg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise HTTPException(500, "Nie udało się zakodować JPEG.")
    return Response(content=jpg.tobytes(), media_type="image/jpeg")

@router.get("/stream", summary="Ciągły podgląd (MJPEG)")
def stream(camera: str | None = None, fps: int = 15, quality: int = 80, w: int = 640, h: int = 480):
    """
    Domyślnie (camera=None) używa RealSense (pyrealsense2).
    UVC wymuś jawnie: ?camera=/dev/video2 lub indeks ?camera=2.
    RS po serialu: ?camera=F0123456
    """
    cam_arg = _normalize_device(camera)
    cap = ImageCapture(device=cam_arg, width=int(w), height=int(h), fps=int(fps))

    def gen():
        try:
            interval = 1.0 / max(1, int(fps))
            while True:
                frame = cap.capture_image()
                ok, jpg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
                if not ok:
                    continue
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n" +
                       jpg.tobytes() + b"\r\n")
                time.sleep(interval)
        finally:
            cap.release()

    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")
