"""HTTP validation and artifact transport; no agent decisions or print submission."""
import io
import mimetypes
from pathlib import Path
import re
from uuid import uuid4

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse

from app.api.schemas import AnswerRequest, ResumeRequest, SessionRequest
from app.api.session_manager import SessionError, inside
from app.workflow.runner import scan_formats

router = APIRouter(prefix="/api")


def manager(request):
    return request.app.state.sessions


@router.post("/sessions", status_code=201)
def create(request: Request, body: SessionRequest):
    return {"session_id": manager(request).create(body.model_dump())}


@router.get("/sessions")
def sessions(request: Request):
    mgr = manager(request)
    with mgr.lock:
        values = list(mgr.sessions.values())
    rows = []
    for session in values:
        with session.lock:
            public = session.public()
            rows.append({k: public[k] for k in ("session_id", "created_at", "request", "status")})
    return sorted(rows, key=lambda r: r["created_at"], reverse=True)


@router.get("/sessions/{sid}")
def session(request: Request, sid: str):
    current = manager(request).get(sid)
    with current.lock:
        return current.public()


@router.post("/sessions/{sid}/answer")
def answer(request: Request, sid: str, body: AnswerRequest):
    manager(request).answer(sid, body.interrupt_token, body.answer)
    return {"ok": True}


@router.post("/sessions/{sid}/resume")
def resume(request: Request, sid: str, body: ResumeRequest):
    manager(request).resume(sid, body.recover_stopped_tuning, body.report_revision)
    return {"ok": True}


@router.post("/sessions/{sid}/stop")
def stop(request: Request, sid: str):
    """Cooperative pause after the current node; never interrupt hardware I/O halfway."""
    current = manager(request).get(sid)
    with current.lock:
        if not current.busy:
            raise SessionError(409, "Answer the pending interrupt with stop, or leave the session stopped")
        current.stop_requested = True
    return {"ok": True, "message": "Will pause after the active graph node; its current operation may finish"}


def validate_scan(data, suffix):
    try:
        if suffix == ".pdf":
            import pymupdf
            with pymupdf.open(stream=data, filetype="pdf") as document:
                if document.is_encrypted or document.page_count < 1:
                    raise ValueError("PDF must have readable pages")
        else:
            from PIL import Image
            with Image.open(io.BytesIO(data)) as image:
                expected = "PNG" if suffix == ".png" else "JPEG"
                if image.format != expected:
                    raise ValueError("Scan content does not match its extension")
                image.verify()
    except Exception:
        raise SessionError(422, "Unreadable scan or content does not match its file extension") from None


@router.post("/sessions/{sid}/scans")
def scans(request: Request, sid: str, interrupt_token: str = Form(...), files: list[UploadFile] = File(...)):
    mgr = manager(request)
    current = mgr.get(sid)
    with current.lock:
        payload = dict(mgr.pending(current, interrupt_token))
        if payload["kind"] != "tuning_scan":
            raise SessionError(409, "The active interrupt is not requesting scans")
    limit = request.app.state.upload_limit
    total, loaded = 0, []
    if not files or len(files) > 32:
        raise SessionError(422, "Upload between 1 and 32 scan files")
    for upload in files:
        name = re.sub(r"[^A-Za-z0-9._ -]", "_", (upload.filename or "scan").replace("\\", "/").split("/")[-1]).strip(" .")
        suffix = Path(name).suffix.lower()
        if suffix.lstrip(".") not in scan_formats():
            raise SessionError(422, "Unsupported scan format")
        name = Path(name).stem[:80] + suffix
        data = upload.file.read(limit - total + 1)
        total += len(data)
        if total > limit:
            raise SessionError(413, "Scan upload exceeds API_UPLOAD_MAX_BYTES")
        validate_scan(data, suffix)
        loaded.append((name, data))
    # The graph could have advanced while content was validated. Reclaim the exact pause.
    with current.lock:
        active = mgr.pending(current, interrupt_token)
        if active != payload:
            raise SessionError(409, "The scan trial changed during upload")
        folder = inside(current.root, active["suggested_scan_directory"])
        folder.mkdir(parents=True, exist_ok=True)
        saved = []
        for name, data in loaded:
            target = inside(current.root, folder / f"{uuid4().hex}_{name}")
            with target.open("xb") as stream:
                stream.write(data)
            saved.append(str(target))
        current.record.setdefault("uploads", []).extend(saved)
        current.persist()
    return {"paths": saved}


@router.get("/sessions/{sid}/files/{path:path}")
def files(request: Request, sid: str, path: str):
    current = manager(request).get(sid)
    target = inside(current.root, path.replace("\\", "/"))
    relative = target.relative_to(current.root)
    if any(part.startswith(".") for part in relative.parts) or target.suffix.lower() not in (".pdf", ".png", ".jpg", ".jpeg"):
        raise SessionError(403, "Only PDF and image artifacts are available")
    if not target.is_file():
        raise SessionError(404, "Artifact not found")
    return FileResponse(target, media_type=mimetypes.guess_type(target.name)[0],
                        headers={"Cache-Control": "no-store, max-age=0"})


@router.get("/printers")
def printers(request: Request, fake_printer: bool = True):
    try:
        return {"printers": request.app.state.printers(fake_printer)}
    except Exception as error:
        # OS command output may contain arbitrary environment values.
        raise HTTPException(503, f"Printer discovery failed ({type(error).__name__}); check the server printer configuration") from None


def discover_printers(fake):
    # Read-only integration call with isolated subprocess environment (no os.environ mutation).
    import json
    from app.printing.tools import load_printer_tools
    tools = load_printer_tools({"PRINTER_MODE": "fake" if fake else "real"})
    result = next(t for t in tools if t.name == "list_printers").invoke({})
    result = json.loads(result) if isinstance(result, str) else result
    if not result.get("ok", True):
        raise RuntimeError("Printer discovery failed")
    return result["printers"]
