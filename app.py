# app.py  (główny plik uruchamiany przez Dockerfile)
import os
import sys
import uvicorn

# Upewnij się, że katalog 'src' jest na PYTHONPATH, gdy app.py jest poza 'src'
ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

# Importuj bezpośrednio obiekt FastAPI
from src.api.blueprints.main import app  # <- to jest src/api/blueprints/main.py: app

if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "5000"))
    reload_enabled = os.getenv("RELOAD", "false").lower() in ("1", "true", "yes")
    log_level = os.getenv("LOG_LEVEL", "info")

    # Bezpieczniej przekazać obiekt app, a nie string "moduł:app"
    uvicorn.run(
        app,
        host=host,
        port=port,
        reload=reload_enabled,
        log_level=log_level,
    )
