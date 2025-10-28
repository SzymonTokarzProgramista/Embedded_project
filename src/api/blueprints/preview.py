from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
import os

router = APIRouter(tags=["Preview"])

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
UI_DIR = os.path.join(BASE_DIR, "ui")

@router.get("/preview")
def preview_ui():
    index_path = os.path.join(UI_DIR, "index.html")
    if not os.path.isfile(index_path):
        raise HTTPException(500, f"Brak pliku index.html w {UI_DIR}")
    return FileResponse(index_path)
