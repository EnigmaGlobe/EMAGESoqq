# app/main.py
from fastapi import FastAPI
from app.logging_config import setup_logging
from app.routes import router
from app.deps import load_bundle

def create_app() -> FastAPI:
    setup_logging()
    app = FastAPI()
    app.include_router(router)

    @app.on_event("startup")
    def _startup():
        load_bundle()

    return app

app = create_app()
