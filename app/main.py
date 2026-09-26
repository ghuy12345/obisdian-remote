"""FastAPI app: REST API + MCP endpoint + background sync loop, in one process."""
from __future__ import annotations

import contextlib
import hmac
import logging
import threading
from typing import Literal

import anyio
from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .config import get_settings
from .embeddings import EmbeddingError
from .engine import Brain, NoteNotFound, WriteRejected
from .gitrepo import GitError, WriteConflict
from .mcp_tools import build_mcp, set_brain

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("vault-brain")

settings = get_settings()
brain = Brain(settings)
set_brain(brain)
mcp = build_mcp([h.strip() for h in settings.allowed_hosts.split(",") if h.strip()])
mcp_app = mcp.streamable_http_app()  # must be created before session_manager is used
_stop = threading.Event()


def _startup_and_loop() -> None:
    while not _stop.is_set():
        try:
            brain.startup()
            break
        except Exception as e:  # keep retrying (e.g. repo URL not set yet, network blip)
            brain.last_error = f"startup: {e}"
            log.exception("startup failed, retrying in 30s")
            _stop.wait(30)
    while not _stop.wait(settings.sync_interval_seconds):
        try:
            brain.sync()
        except Exception:
            log.exception("sync crashed; will retry")


@contextlib.asynccontextmanager
async def lifespan(_: FastAPI):
    worker = threading.Thread(target=_startup_and_loop, daemon=True, name="sync")
    worker.start()
    async with mcp.session_manager.run():
        yield
    _stop.set()


app = FastAPI(title="Vault Brain", lifespan=lifespan, docs_url="/api/docs", openapi_url="/api/openapi.json")

if settings.mcp_secret:
    app.mount(f"/{settings.mcp_secret}", mcp_app)
else:
    log.warning("MCP_SECRET is not set; the MCP endpoint is disabled")


# ---- errors -------------------------------------------------------------
@app.exception_handler(NoteNotFound)
async def _nf(_, e: NoteNotFound):
    return JSONResponse(status_code=404, content={"error": str(e), "suggestions": e.suggestions})


@app.exception_handler(WriteRejected)
async def _wr(_, e: WriteRejected):
    return JSONResponse(status_code=400, content={"error": str(e)})


@app.exception_handler(WriteConflict)
async def _wc(_, e: WriteConflict):
    return JSONResponse(status_code=409, content={"error": str(e)})


@app.exception_handler(GitError)
async def _ge(_, e: GitError):
    return JSONResponse(status_code=502, content={"error": str(e)})


@app.exception_handler(EmbeddingError)
async def _ee(_, e: EmbeddingError):
    return JSONResponse(status_code=502, content={"error": str(e)})


# ---- auth ---------------------------------------------------------------
def require_token(authorization: str = Header(default="")) -> None:
    if not settings.api_token:
        raise HTTPException(503, "API_TOKEN is not set on the server")
    token = authorization.removeprefix("Bearer ").strip()
    if not hmac.compare_digest(token, settings.api_token):
        raise HTTPException(401, "invalid token")


def require_ready() -> None:
    if not brain.ready:
        raise HTTPException(503, "index is still starting up")


async def run(fn, *a):
    return await anyio.to_thread.run_sync(lambda: fn(*a))


api = [Depends(require_token), Depends(require_ready)]


@app.get("/health")
async def health():
    return {"ok": True, "ready": brain.ready, "last_error": brain.last_error}


@app.get("/api/status", dependencies=[Depends(require_token)])
async def status():
    return await run(brain.status)


@app.post("/api/sync", dependencies=api)
async def sync():
    return await run(brain.sync)


@app.get("/api/context", dependencies=api)
async def context(topic: str, max_chars: int = Query(6000, ge=500, le=40000)):
    return await run(brain.get_context, topic, max_chars)


@app.get("/api/search", dependencies=api)
async def search(q: str, mode: Literal["hybrid", "keyword", "semantic"] = "hybrid",
                 limit: int = Query(10, ge=1, le=50), folder: str | None = None):
    return await run(brain.search, q, mode, limit, folder)


@app.get("/api/notes", dependencies=api)
async def list_notes(folder: str | None = None, tag: str | None = None,
                     sort: Literal["recent", "title", "size"] = "recent", limit: int = Query(50, ge=1, le=500)):
    return await run(brain.list_notes, folder, tag, sort, limit)


@app.get("/api/note", dependencies=api)
async def read_note(name: str, max_chars: int = Query(20000, ge=500, le=100000)):
    return await run(brain.read_note, name, max_chars)


@app.get("/api/backlinks", dependencies=api)
async def backlinks(name: str):
    return await run(brain.backlinks, name)


@app.get("/api/neighbors", dependencies=api)
async def neighbors(name: str, hops: int = Query(2, ge=1, le=4), limit: int = Query(50, ge=1, le=200)):
    return await run(brain.neighbors, name, hops, limit)


@app.get("/api/path", dependencies=api)
async def find_path(a: str, b: str):
    return await run(brain.find_path, a, b)


@app.get("/api/related", dependencies=api)
async def related(name: str, limit: int = Query(10, ge=1, le=50), include_linked: bool = False):
    return await run(brain.related, name, limit, include_linked)


@app.get("/api/hubs", dependencies=api)
async def hubs(limit: int = Query(20, ge=1, le=100), folder: str | None = None):
    return await run(brain.hubs, limit, folder)


@app.get("/api/clusters", dependencies=api)
async def clusters(min_size: int = Query(3, ge=2), limit: int = Query(20, ge=1, le=50)):
    return await run(brain.clusters, min_size, limit)


@app.get("/api/gaps", dependencies=api)
async def gaps(limit: int = Query(30, ge=1, le=200)):
    return await run(brain.gaps, limit)


@app.get("/api/history", dependencies=api)
async def history(name: str, limit: int = Query(20, ge=1, le=100)):
    return await run(brain.history, name, limit)


class CreateNote(BaseModel):
    path: str
    content: str
    frontmatter: dict | None = None


class AppendNote(BaseModel):
    note: str
    content: str
    heading: str | None = Field(default=None)


@app.post("/api/notes", dependencies=api, status_code=201)
async def create_note(body: CreateNote):
    return await run(brain.create_note, body.path, body.content, body.frontmatter)


@app.post("/api/append", dependencies=api)
async def append(body: AppendNote):
    return await run(brain.append_to_note, body.note, body.content, body.heading)
