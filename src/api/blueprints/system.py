# src/api/blueprints/system.py
from fastapi import APIRouter
import psutil, platform, os, time
from datetime import datetime

router = APIRouter(prefix="/system", tags=["System"])
_START = time.time()

@router.get("/info")
def info():
    return {
        "system": platform.system(),
        "node": platform.node(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "boot_time": datetime.fromtimestamp(psutil.boot_time()).isoformat(),
    }

@router.get("/cpu")
def cpu():
    return {
        "cpu_percent": psutil.cpu_percent(interval=0.3),
        "cpus": psutil.cpu_count(),
        "load_avg": os.getloadavg() if hasattr(os, "getloadavg") else None,
    }

@router.get("/memory")
def memory():
    m = psutil.virtual_memory()
    return {"total": m.total, "used": m.used, "available": m.available, "percent": m.percent}

@router.get("/disk")
def disk():
    d = psutil.disk_usage("/")
    return {"total": d.total, "used": d.used, "free": d.free, "percent": d.percent}

@router.get("/uptime")
def uptime():
    return {"seconds": round(time.time() - _START, 1)}
