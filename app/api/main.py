"""Run with one worker: python -m uvicorn app.api.main:app --host 127.0.0.1 --port 8000."""
from contextlib import asynccontextmanager
import os

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes import discover_printers, router
from app.api.session_manager import SessionError, SessionManager, public_value
from app.workflow.runner import BASE_DIR


def create_app(runs_root=None, prepare_fn=None, printers_fn=None, upload_limit=None):
    load_dotenv(BASE_DIR / ".env")

    @asynccontextmanager
    async def lifespan(app):
        kwargs = {"prepare_fn": prepare_fn} if prepare_fn else {}
        app.state.sessions = SessionManager(runs_root or os.getenv("API_RUNS_DIR") or BASE_DIR / "runs", **kwargs)
        yield
        # Graph calls never run on the event loop, including shutdown waits.
        from starlette.concurrency import run_in_threadpool
        await run_in_threadpool(app.state.sessions.close)

    app = FastAPI(title="Agent print-tuning API", lifespan=lifespan)
    app.state.upload_limit = upload_limit or int(os.getenv("API_UPLOAD_MAX_BYTES", 50 * 1024 * 1024))
    app.state.printers = printers_fn or discover_printers
    origins = os.getenv("CORS_ORIGINS", "http://localhost:8501,http://127.0.0.1:8501").split(",")
    app.add_middleware(CORSMiddleware, allow_origins=[v.strip() for v in origins],
                       allow_methods=["GET", "POST"], allow_headers=["Content-Type"])

    @app.middleware("http")
    async def upload_bound(request: Request, call_next):
        if request.url.path.endswith("/scans"):
            try:
                if int(request.headers.get("content-length", 0)) > app.state.upload_limit + 1024 * 1024:
                    return JSONResponse({"detail": "Upload too large"}, status_code=413)
            except ValueError:
                return JSONResponse({"detail": "Invalid content length"}, status_code=400)
            chunks, size = [], 0
            async for chunk in request.stream():
                size += len(chunk)
                if size > app.state.upload_limit + 1024 * 1024:
                    return JSONResponse({"detail": "Upload too large"}, status_code=413)
                chunks.append(chunk)
            request._body = b"".join(chunks)
        return await call_next(request)

    @app.exception_handler(SessionError)
    async def session_error(request, error):
        return JSONResponse({"detail": public_value(error.message)}, status_code=error.status)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request, error):
        # Pydantic normally echoes submitted input, including unknown credential fields.
        return JSONResponse({"detail": [{"loc": list(e["loc"]), "msg": e["msg"]} for e in error.errors()]}, status_code=422)

    app.include_router(router)
    return app


app = create_app()
