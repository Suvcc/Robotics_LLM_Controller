"""FastAPI application exposing the safe AlienGo runtime on a private LAN."""

from __future__ import annotations

import asyncio
import os
import queue
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, File, Request, Response, UploadFile, WebSocket
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ..config import AppConfig
from ..runtime import (
    AuthenticationFailed,
    ClientSession,
    ControlRequired,
    RateLimited,
    RobotRuntime,
    RuntimeConflict,
    RuntimeNotFound,
    RuntimeProblem,
)

COOKIE_NAME = "aliengo_session"
STATIC_DIR = Path(__file__).resolve().parent / "static"


class SessionRequest(BaseModel):
    display_name: str = Field(min_length=1, max_length=32)
    passcode: str = Field(min_length=1)


class CommandRequest(BaseModel):
    text: str = Field(min_length=1)


class ConfirmationRequest(BaseModel):
    approved: bool


def _bearer_token(request: Request) -> str | None:
    authorization = request.headers.get("authorization", "")
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() == "bearer" and token:
        return token
    return request.cookies.get(COOKIE_NAME)


def _status_for_problem(problem: RuntimeProblem) -> int:
    if isinstance(problem, AuthenticationFailed):
        return 401
    if isinstance(problem, RateLimited):
        return 429
    if isinstance(problem, ControlRequired):
        return 403
    if isinstance(problem, RuntimeNotFound):
        return 404
    if problem.code == "audio_too_large":
        return 413
    if problem.code in {"invalid_command", "invalid_display_name", "invalid_audio"}:
        return 422
    return 409


def create_app(
    config: AppConfig,
    *,
    runtime: RobotRuntime | None = None,
    passcode: str | None = None,
) -> FastAPI:
    if runtime is None:
        resolved_passcode = passcode or os.environ.get("ALIENGO_SERVER_PASSCODE", "")
        runtime = RobotRuntime(config, resolved_passcode)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        runtime.start()
        yield
        runtime.close()

    app = FastAPI(
        title="AlienGo LAN Control API",
        version="1.0.0",
        docs_url="/api/docs",
        openapi_url="/api/v1/openapi.json",
        lifespan=lifespan,
    )
    app.state.runtime = runtime

    @app.exception_handler(RuntimeProblem)
    async def runtime_problem_handler(request: Request, exc: RuntimeProblem):
        return JSONResponse(
            status_code=_status_for_problem(exc),
            content={"error": {"code": exc.code, "message": exc.message}},
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(request: Request, exc: RequestValidationError):
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "code": "validation_error",
                    "message": "Request validation failed.",
                }
            },
        )

    @app.middleware("http")
    async def security_headers(request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self'; "
            "connect-src 'self' wss:; media-src 'self' blob:; "
            "img-src 'self' data:; frame-ancestors 'none'"
        )
        if request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    def current_session(request: Request) -> ClientSession:
        return runtime.authenticate(_bearer_token(request))

    @app.get("/api/v1/health")
    def health():
        return {"status": "ok"}

    @app.post("/api/v1/sessions", status_code=201)
    def create_session(payload: SessionRequest, request: Request, response: Response):
        source_ip = request.client.host if request.client else "unknown"
        session = runtime.create_session(
            payload.passcode, payload.display_name, source_ip
        )
        response.set_cookie(
            COOKIE_NAME,
            session.token,
            max_age=int(config.server.session_idle_timeout_s),
            httponly=True,
            secure=True,
            samesite="strict",
            path="/",
        )
        return {
            "client_id": session.id,
            "display_name": session.display_name,
            "access_token": session.token,
            "expires_in_s": config.server.session_idle_timeout_s,
        }

    @app.delete("/api/v1/sessions/current", status_code=204)
    def logout(
        response: Response,
        session: ClientSession = Depends(current_session),
    ):
        runtime.logout(session.id)
        response.delete_cookie(COOKIE_NAME, path="/", secure=True, samesite="strict")

    @app.get("/api/v1/state")
    def state(session: ClientSession = Depends(current_session)):
        return runtime.state_snapshot(session.id)

    @app.put("/api/v1/lease")
    def acquire_lease(session: ClientSession = Depends(current_session)):
        return runtime.acquire_lease(session.id)

    @app.delete("/api/v1/lease", status_code=204)
    def release_lease(session: ClientSession = Depends(current_session)):
        runtime.release_lease(session.id)

    @app.post("/api/v1/commands", status_code=202)
    def submit_command(
        payload: CommandRequest,
        session: ClientSession = Depends(current_session),
    ):
        job = runtime.submit_command(session.id, payload.text)
        return {"id": job.id, "status": job.status}

    @app.get("/api/v1/commands/{command_id}")
    def command_status(
        command_id: str,
        session: ClientSession = Depends(current_session),
    ):
        return runtime.get_command(session.id, command_id)

    @app.post("/api/v1/commands/{command_id}/confirmation", status_code=204)
    def resolve_confirmation(
        command_id: str,
        payload: ConfirmationRequest,
        session: ClientSession = Depends(current_session),
    ):
        runtime.resolve_confirmation(session.id, command_id, payload.approved)

    @app.post("/api/v1/estop", status_code=204)
    def activate_estop(session: ClientSession = Depends(current_session)):
        runtime.activate_estop(
            actor_id=session.id,
            reason=f"Activated by {session.display_name}.",
        )

    @app.delete("/api/v1/estop", status_code=204)
    def release_estop(session: ClientSession = Depends(current_session)):
        runtime.release_estop(session.id)

    @app.post("/api/v1/reset", status_code=204)
    def reset(session: ClientSession = Depends(current_session)):
        runtime.reset(session.id)

    @app.post("/api/v1/transcriptions")
    async def transcribe(
        audio: UploadFile = File(...),
        session: ClientSession = Depends(current_session),
    ):
        data = await audio.read(config.server.max_audio_bytes + 1)
        if len(data) > config.server.max_audio_bytes:
            raise RuntimeConflict(
                "audio_too_large",
                f"Audio exceeds {config.server.max_audio_bytes} bytes.",
            )
        try:
            future = runtime.submit_transcription(
                session.id, data, audio.content_type or ""
            )
            return await asyncio.wrap_future(future)
        except ValueError as exc:
            raise RuntimeConflict("invalid_audio", str(exc)) from exc

    @app.websocket("/api/v1/events")
    async def events(websocket: WebSocket):
        token = websocket.cookies.get(COOKIE_NAME)
        requested_protocols = websocket.scope.get("subprotocols", [])
        accepted_protocol = None
        if not token and len(requested_protocols) >= 2 and requested_protocols[0] == "bearer":
            token = requested_protocols[1]
            accepted_protocol = "bearer"
        try:
            session = runtime.authenticate(token)
        except AuthenticationFailed:
            await websocket.close(code=4401)
            return
        await websocket.accept(subprotocol=accepted_protocol)
        subscriber = runtime.subscribe(session.id)
        try:
            await websocket.send_json(
                {"type": "snapshot", "state": runtime.state_snapshot(session.id)}
            )
            while True:
                try:
                    event = await asyncio.to_thread(subscriber.get, True, 1.0)
                except queue.Empty:
                    event = {"type": "heartbeat"}
                await websocket.send_json(event)
        except Exception:
            # Disconnects and network failures are normal for browser clients.
            pass
        finally:
            runtime.unsubscribe(session.id, subscriber)

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(STATIC_DIR / "index.html")

    return app
