# src/api/blueprints/main.py
from fastapi import FastAPI
from fastapi.responses import RedirectResponse, HTMLResponse

from .camera import router as camera_router
from .system import router as system_router

app = FastAPI(title="Embedded Camera API", version="0.1.0")

app.include_router(system_router)
app.include_router(camera_router)

@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse("/docs")

# prosta stronka podglądu (możesz usunąć jeśli niepotrzebna)
@app.get("/preview", response_class=HTMLResponse, include_in_schema=False)
def preview():
    return """
    <html><body style="margin:0;background:#111;color:#eee;font:16px sans-serif">
      <div style="padding:10px">Live preview:</div>
      <img src="/camera/stream" style="width:100%;height:auto;display:block"/>
    </body></html>
    """
