"""Small client for the print-tuning web API (the contract in frontend/README.md).

Every failure (connection refused, timeout, HTTP error, bad JSON) becomes an ApiError whose
message can be shown in the UI as it is. The base URL comes from API_URL
(default http://localhost:8000). Tests pass a FastAPI TestClient as `http`.
"""
import os
from pathlib import PurePosixPath, PureWindowsPath
from urllib.parse import quote

import requests

DEFAULT_API_URL = "http://localhost:8000"
SCAN_FORMATS = ("png", "jpg", "jpeg", "pdf")
PREVIEW_SUFFIXES = (".pdf", ".png", ".jpg", ".jpeg")


class ApiError(Exception):
    """A failed API call; str(error) is a message for the user, status is the HTTP code or None."""

    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


def api_url() -> str:
    return (os.environ.get("API_URL") or DEFAULT_API_URL).rstrip("/")


def _detail(response) -> str:
    """The server's own explanation: FastAPI's "detail" (text or validation list), else the body."""
    try:
        body = response.json()
    except ValueError:
        return (response.text or "").strip()[:500]
    detail = body.get("detail", body) if isinstance(body, dict) else body
    if isinstance(detail, list):        # FastAPI request-validation errors
        return "; ".join(f"{'.'.join(str(p) for p in item.get('loc', [])[1:]) or 'body'}: {item.get('msg')}"
                         if isinstance(item, dict) else str(item) for item in detail)
    return str(detail)


class ApiClient:
    def __init__(self, base_url: str | None = None, http=None, timeout: float = 30.0):
        self.base_url = (base_url or api_url()).rstrip("/")
        self._own_session = http is None
        self.http = http if http is not None else requests.Session()
        self.timeout = timeout

    # ---------- transport ----------

    def _call(self, method: str, path: str, *, raw: bool = False, **kwargs):
        if self._own_session:
            kwargs.setdefault("timeout", self.timeout)
        url = self.base_url + path
        try:
            response = self.http.request(method, url, **kwargs)
        except requests.ConnectionError:
            raise ApiError(f"Cannot reach the backend at {self.base_url}. Is it running?") from None
        except requests.Timeout:
            raise ApiError(f"The backend at {self.base_url} did not answer within {self.timeout:g} s.") from None
        except requests.RequestException as error:
            raise ApiError(f"Request to the backend failed: {error}") from None
        if response.status_code >= 400:
            what = {404: "Not found", 409: "Conflict", 422: "Invalid input", 403: "Refused"}.get(
                response.status_code, f"Backend error {response.status_code}")
            message = _detail(response)
            raise ApiError(f"{what}: {message}" if message else what, response.status_code)
        if raw:
            return response.content
        try:
            return response.json()
        except ValueError:
            raise ApiError(f"The backend sent a response that is not JSON ({method} {path}).",
                           response.status_code) from None

    # ---------- endpoints ----------

    def start_session(self, *, request: str, model: str | None = None, specialist_model: str | None = None,
                      printer: str | None = None, fake_printer: bool = True, paper_color: str | None = None,
                      initial_decisions: int | None = None, extra_decisions: int | None = None) -> str:
        body = {"request": request, "model": model or None, "specialist_model": specialist_model or None,
                "printer": printer or None, "fake_printer": bool(fake_printer), "paper_color": paper_color or None}
        # Omission keeps the backend's defaults; programmatic callers may still override them.
        if initial_decisions is not None:
            body["initial_decisions"] = initial_decisions
        if extra_decisions is not None:
            body["extra_decisions"] = extra_decisions
        return self._call("POST", "/api/sessions", json=body)["session_id"]

    def list_sessions(self) -> list[dict]:
        return self._call("GET", "/api/sessions")

    def get_session(self, session_id: str) -> dict:
        return self._call("GET", f"/api/sessions/{quote(session_id, safe='')}")

    def answer(self, session_id: str, answer, interrupt_token: str | None = None) -> dict:
        """Answer the pending pause; a 409 (nothing pending) or 422 (invalid answer) raises ApiError."""
        if interrupt_token is None:
            interrupt_token = (self.get_session(session_id).get("pending") or {}).get("interrupt_token")
        return self._call("POST", f"/api/sessions/{quote(session_id, safe='')}/answer",
                          json={"answer": answer, "interrupt_token": interrupt_token})

    def upload_scans(self, session_id: str, files: list[tuple[str, bytes, str | None]], interrupt_token: str | None = None) -> list[str]:
        """Upload (filename, content, mime type) scans for the active round; return the saved paths."""
        if not files:
            raise ApiError("Choose at least one scan file to upload.")
        bad = [name for name, _, _ in files if name.rsplit(".", 1)[-1].lower() not in SCAN_FORMATS]
        if bad:
            raise ApiError(f"Not a scan file ({', '.join(SCAN_FORMATS)}): {', '.join(bad)}")
        parts = [("files", (name, content, mime or "application/octet-stream")) for name, content, mime in files]
        if interrupt_token is None:
            interrupt_token = (self.get_session(session_id).get("pending") or {}).get("interrupt_token")
        return self._call("POST", f"/api/sessions/{quote(session_id, safe='')}/scans", files=parts,
                          data={"interrupt_token": interrupt_token or ""})["paths"]

    def resume(self, session_id: str, *, recover_stopped_tuning=False, report_revision=None):
        return self._call("POST", f"/api/sessions/{quote(session_id, safe='')}/resume",
                          json={"recover_stopped_tuning": recover_stopped_tuning, "report_revision": report_revision})

    def download_file(self, session_id: str, path: str) -> bytes:
        """A file of the run, by its path relative to run_dir."""
        return self._call("GET", self.file_path(session_id, path), raw=True)

    def file_url(self, session_id: str, path: str) -> str:
        """Absolute URL of a run file, for opening it in the browser."""
        return self.base_url + self.file_path(session_id, path)

    @staticmethod
    def file_path(session_id: str, path: str) -> str:
        # Each segment is encoded on its own; "." and ".." are encoded too so the client does not
        # silently resolve them and the server sees (and refuses) exactly what was asked for.
        segments = [quote(part, safe="") if part.strip(".") else part.replace(".", "%2E")
                    for part in str(path).replace("\\", "/").split("/")]
        return f"/api/sessions/{quote(session_id, safe='')}/files/" + "/".join(segments)

    def printers(self, real=False) -> list[str]:
        return list(self._call("GET", "/api/printers" + ("?fake_printer=false" if real else "")).get("printers", []))


# ---------- helpers for the UI (pure, no network) ----------

def relative_run_path(path, run_dir) -> str | None:
    """path relative to run_dir in "/" form, or None if it is not inside run_dir (Windows or POSIX paths)."""
    if not path or not run_dir:
        return None
    text, root = str(path), str(run_dir)
    flavour = PureWindowsPath if (":" in root[:3] or "\\" in root) else PurePosixPath
    candidate, base = flavour(text), flavour(root)
    if not candidate.is_absolute():
        candidate = base / candidate
    try:
        relative = candidate.relative_to(base)
    except ValueError:
        return None
    parts = relative.parts
    if not parts or ".." in parts:
        return None
    return "/".join(parts)


def find_preview_files(value, run_dir, limit: int = 12) -> list[str]:
    """Paths (relative to run_dir) of PDFs/PNGs/JPGs mentioned anywhere in a JSON-like value, in order."""
    found: list[str] = []

    def walk(item):
        if len(found) >= limit:
            return
        if isinstance(item, dict):
            for child in item.values():
                walk(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                walk(child)
        elif isinstance(item, str) and item.lower().endswith(PREVIEW_SUFFIXES):
            relative = relative_run_path(item, run_dir)
            if relative and relative not in found:
                found.append(relative)

    walk(value)
    return found
