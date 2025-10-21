from fastapi import FastAPI
from .system import router as system_router

app = FastAPI(
    title="Mecanum API",
    description=" ",
    version="1.0.0"
)

app.include_router(system_router)