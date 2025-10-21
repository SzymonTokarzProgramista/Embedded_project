from fastapi import APIRouter
import platform
import psutil

router = APIRouter(
    prefix="/system",
    tags=["System"]
)


@router.get("/info")
def system_info():
    """Basic system information."""
    return {
        "hostname": platform.node(),
        "os": platform.system(),
        "os_version": platform.version(),
        "architecture": platform.machine(),
        "python_version": platform.python_version()
    }


@router.get("/memory")
def memory_usage():
    """RAM usage statistics."""
    mem = psutil.virtual_memory()
    return {
        "total_mb": round(mem.total / 1024**2, 2),
        "used_mb": round(mem.used / 1024**2, 2),
        "available_mb": round(mem.available / 1024**2, 2),
        "percent": mem.percent
    }


@router.get("/disk")
def disk_usage():
    """Disk usage for root partition."""
    disk = psutil.disk_usage('/')
    return {
        "total_gb": round(disk.total / 1024**3, 2),
        "used_gb": round(disk.used / 1024**3, 2),
        "free_gb": round(disk.free / 1024**3, 2),
        "percent": disk.percent
    }


@router.get("/cpu")
def cpu_stats():
    """CPU utilization and temperature (if available)."""
    cpu_temp = None
    try:
        with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
            cpu_temp = round(int(f.read()) / 1000.0, 1)
    except FileNotFoundError:
        pass

    return {
        "cpu_percent": psutil.cpu_percent(interval=0.5),
        "cpu_cores": psutil.cpu_count(logical=False),
        "cpu_threads": psutil.cpu_count(logical=True),
        "cpu_temp_celsius": cpu_temp
    }