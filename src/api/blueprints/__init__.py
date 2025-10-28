from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from .camera import router as camera_router
from .preview import router as preview_router
import os

app = FastAPI(title="Embedded Camera API")

# Routers
app.include_router(camera_router)
app.include_router(preview_router)

# Ścieżka do UI:  <repo>/src/ui
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
UI_DIR = os.path.join(BASE_DIR, "ui")

print(f"[UI] Static dir: {UI_DIR}  (exists={os.path.isdir(UI_DIR)})")

# Montuj pliki statyczne
app.mount("/static", StaticFiles(directory=UI_DIR), name="static")
