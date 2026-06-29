# app/main.py
import time
import logging
from fastapi import FastAPI, Request

from app.deps import load_bundle
from app.routes import router
from app.logging_config import setup_logging

setup_logging()
log = logging.getLogger("http")

app = FastAPI()
app.include_router(router)


@app.on_event("startup")
async def preload_model_bundle() -> None:
    # Warm model load once during startup; /ready will report 503 if this fails.
    try:
        load_bundle()
    except Exception:
        log.exception("Model preload failed at startup")

@app.middleware("http")
async def timing_middleware(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    dt_ms = (time.perf_counter() - t0) * 1000.0

    # similar vibe to morgan/express
    log.info('%s %s -> %d (%.1fms)',
             request.method,
             request.url.path + (f"?{request.url.query}" if request.url.query else ""),
             response.status_code,
             dt_ms)

    return response
