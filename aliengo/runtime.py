"""Shared process runtime for CLI-compatible LAN robot control.

The runtime owns the single physical controller and serializes LLM commands,
while preserving one independent AgentLoop history per authenticated client.
Emergency stop deliberately bypasses command serialization.
"""

from __future__ import annotations

import hmac
import queue
import secrets
import threading
import time
import uuid
from collections import defaultdict, deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from .actionlog import ActionLog
from .agent.llm import LLMClient
from .agent.loop import AgentLoop
from .config import AppConfig
from .robot.interface import RobotController
from .robot.mock import MockRobotController
from .safety.estop import EmergencyStopState
from .skills.registry import to_openai_tools

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SYSTEM_PROMPT_PATH = PROJECT_ROOT / "prompts" / "system_prompt.md"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_system_prompt() -> str:
    return SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")


def build_controller(config: AppConfig, log: ActionLog) -> RobotController:
    """Construct the configured backend without importing Jetson libs on PC."""
    if config.robot.backend == "jetracer":
        from .robot.jetracer import JetRacerController

        return JetRacerController()
    return MockRobotController(log=log)


class RuntimeProblem(Exception):
    """Expected runtime/API failure with a stable machine-readable code."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


class AuthenticationFailed(RuntimeProblem):
    def __init__(self, message: str = "Invalid or expired session."):
        super().__init__("authentication_failed", message)


class RateLimited(RuntimeProblem):
    def __init__(self):
        super().__init__("rate_limited", "Too many authentication attempts.")


class ControlRequired(RuntimeProblem):
    def __init__(self, message: str = "An active control lease is required."):
        super().__init__("control_required", message)


class RuntimeConflict(RuntimeProblem):
    pass


class RuntimeNotFound(RuntimeProblem):
    def __init__(self, message: str):
        super().__init__("not_found", message)


@dataclass
class PendingConfirmation:
    skill: str
    params: dict
    reason: str
    created_at: str = field(default_factory=utc_now)
    response: bool | None = None
    resolved_at: str | None = None
    signal: threading.Event = field(default_factory=threading.Event, repr=False)

    def to_dict(self) -> dict:
        return {
            "skill": self.skill,
            "params": self.params,
            "reason": self.reason,
            "created_at": self.created_at,
            "response": self.response,
            "resolved_at": self.resolved_at,
        }


@dataclass
class CommandJob:
    id: str
    owner_id: str
    text: str
    status: str = "accepted"
    created_at: str = field(default_factory=utc_now)
    started_at: str | None = None
    completed_at: str | None = None
    reply: str | None = None
    error: str | None = None
    events: list[dict] = field(default_factory=list)
    confirmation: PendingConfirmation | None = None

    def to_dict(self) -> dict:
        result = {
            "id": self.id,
            "text": self.text,
            "status": self.status,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "reply": self.reply,
            "error": self.error,
            "events": list(self.events),
        }
        if self.confirmation:
            result["confirmation"] = self.confirmation.to_dict()
        return result


@dataclass
class ClientSession:
    id: str
    token: str
    display_name: str
    loop: AgentLoop
    created_at: str = field(default_factory=utc_now)
    last_seen: float = 0.0
    revoked: bool = False
    subscribers: set[queue.Queue] = field(default_factory=set, repr=False)


@dataclass
class ControlLease:
    owner_id: str
    owner_name: str
    acquired_at: str
    last_control_at: float


Transcriber = Callable[[bytes, str, AppConfig], dict]


class RobotRuntime:
    """Thread-safe owner of sessions, control lease, commands, and e-stop."""

    TERMINAL_STATUSES = {"completed", "failed", "cancelled"}

    def __init__(
        self,
        config: AppConfig,
        passcode: str,
        *,
        controller: RobotController | None = None,
        llm=None,
        system_prompt: str | None = None,
        log: ActionLog | None = None,
        clock: Callable[[], float] = time.monotonic,
        transcriber: Transcriber | None = None,
    ):
        if len(passcode) < 12:
            raise ValueError("ALIENGO_SERVER_PASSCODE must be at least 12 characters.")
        self.config = config
        self._passcode = passcode
        self.log = log or ActionLog(config.logging.actions_path)
        self.controller = controller or build_controller(config, self.log)
        self.llm = llm or LLMClient(config.llm)
        self.system_prompt = system_prompt or load_system_prompt()
        self.tools = to_openai_tools()
        self.estop_state = EmergencyStopState()
        self._clock = clock
        self._transcriber = transcriber

        self._lock = threading.RLock()
        self._sessions: dict[str, ClientSession] = {}
        self._tokens: dict[str, str] = {}
        self._commands: dict[str, CommandJob] = {}
        self._lease: ControlLease | None = None
        self._active_job: CommandJob | None = None
        self._auth_attempts: dict[str, deque[float]] = defaultdict(deque)

        self._command_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="aliengo-command"
        )
        self._speech_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="aliengo-speech"
        )
        self._monitor_stop = threading.Event()
        self._monitor_thread: threading.Thread | None = None
        self._closed = False

    # -- lifecycle --------------------------------------------------------

    def start(self) -> None:
        with self._lock:
            if self._monitor_thread and self._monitor_thread.is_alive():
                return
            self._monitor_stop.clear()
            self._monitor_thread = threading.Thread(
                target=self._monitor,
                daemon=True,
                name="aliengo-runtime-monitor",
            )
            self._monitor_thread.start()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._monitor_stop.set()
            pending = self._active_job.confirmation if self._active_job else None
            if pending and pending.response is None:
                pending.response = False
                pending.resolved_at = utc_now()
                pending.signal.set()
        self.estop_state.activate()
        self.controller.emergency_stop()
        self._command_executor.shutdown(wait=False, cancel_futures=True)
        self._speech_executor.shutdown(wait=False, cancel_futures=True)
        self.controller.close()

    def _monitor(self) -> None:
        while not self._monitor_stop.wait(0.5):
            try:
                self.expire_stale_state()
            except Exception as exc:  # keep the safety monitor alive
                self.log.write(type="runtime_monitor_error", error=str(exc))

    def expire_stale_state(self) -> None:
        should_estop = False
        expired_owner: str | None = None
        now = self._clock()
        with self._lock:
            if self._lease and not self._lease_is_pinned_locked():
                elapsed = now - self._lease.last_control_at
                if elapsed >= self.config.server.lease_timeout_s:
                    expired_owner = self._lease.owner_id
                    self._lease = None
                    state = self.controller.get_state()
                    should_estop = state.moving or state.following

            expired_sessions = [
                session_id
                for session_id, session in self._sessions.items()
                if (
                    now - session.last_seen
                    >= self.config.server.session_idle_timeout_s
                    and session_id != (self._active_job.owner_id if self._active_job else None)
                    and session_id != (self._lease.owner_id if self._lease else None)
                )
            ]
            for session_id in expired_sessions:
                self._remove_session_locked(session_id)

        if should_estop:
            self.activate_estop(
                actor_id=None,
                reason="Control lease expired while the robot was active.",
            )
        if expired_owner:
            self.log.write(type="lease_expired", client_id=expired_owner)
            self._publish_global({"type": "lease_changed"})

    # -- authentication ---------------------------------------------------

    def create_session(
        self, passcode: str, display_name: str, source_ip: str
    ) -> ClientSession:
        display_name = display_name.strip()
        if not display_name or len(display_name) > 32:
            raise RuntimeConflict(
                "invalid_display_name", "Display name must be 1 to 32 characters."
            )
        self._check_auth_rate_limit(source_ip)
        if not hmac.compare_digest(passcode, self._passcode):
            raise AuthenticationFailed("Incorrect passcode.")
        with self._lock:
            self._auth_attempts.pop(source_ip, None)
            session_id = str(uuid.uuid4())
            token = secrets.token_urlsafe(32)
            loop = AgentLoop(
                llm=self.llm,
                controller=self.controller,
                config=self.config,
                system_prompt=self.system_prompt,
                tools=self.tools,
                log=self.log,
                estop_state=self.estop_state,
            )
            session = ClientSession(
                id=session_id,
                token=token,
                display_name=display_name,
                loop=loop,
                last_seen=self._clock(),
            )
            self._sessions[session_id] = session
            self._tokens[token] = session_id
        self.log.write(
            type="network_session_created",
            client_id=session.id,
            display_name=display_name,
            source_ip=source_ip,
        )
        return session

    def _check_auth_rate_limit(self, source_ip: str) -> None:
        now = self._clock()
        with self._lock:
            attempts = self._auth_attempts[source_ip]
            while attempts and now - attempts[0] >= 60.0:
                attempts.popleft()
            if len(attempts) >= self.config.server.auth_attempts_per_minute:
                raise RateLimited()
            attempts.append(now)

    def authenticate(self, token: str | None) -> ClientSession:
        if not token:
            raise AuthenticationFailed()
        with self._lock:
            session_id = self._tokens.get(token)
            session = self._sessions.get(session_id or "")
            if not session or session.revoked:
                raise AuthenticationFailed()
            if (
                self._clock() - session.last_seen
                >= self.config.server.session_idle_timeout_s
                and session.id != (self._active_job.owner_id if self._active_job else None)
            ):
                self._remove_session_locked(session.id)
                raise AuthenticationFailed()
            session.last_seen = self._clock()
            return session

    def logout(self, session_id: str) -> None:
        should_estop = False
        with self._lock:
            session = self._require_session_locked(session_id)
            session.revoked = True
            self._tokens.pop(session.token, None)
            if self._active_job and self._active_job.owner_id == session_id:
                should_estop = True
                pending = self._active_job.confirmation
                if pending and pending.response is None:
                    pending.response = False
                    pending.resolved_at = utc_now()
                    pending.signal.set()
            else:
                if self._lease and self._lease.owner_id == session_id:
                    state = self.controller.get_state()
                    should_estop = state.moving or state.following
                    self._lease = None
                self._remove_session_locked(session_id)
        if should_estop:
            self.activate_estop(actor_id=session_id, reason="Controller logged out.")
        self._publish_global({"type": "lease_changed"})

    def _remove_session_locked(self, session_id: str) -> None:
        session = self._sessions.pop(session_id, None)
        if session:
            self._tokens.pop(session.token, None)

    def _require_session_locked(self, session_id: str) -> ClientSession:
        session = self._sessions.get(session_id)
        if not session:
            raise AuthenticationFailed()
        return session

    # -- lease ------------------------------------------------------------

    def acquire_lease(self, session_id: str) -> dict:
        self.expire_stale_state()
        with self._lock:
            session = self._require_session_locked(session_id)
            if self._lease and self._lease.owner_id != session_id:
                raise RuntimeConflict(
                    "lease_held",
                    f"Control is currently held by {self._lease.owner_name}.",
                )
            now = self._clock()
            if not self._lease:
                self._lease = ControlLease(
                    owner_id=session.id,
                    owner_name=session.display_name,
                    acquired_at=utc_now(),
                    last_control_at=now,
                )
            else:
                self._lease.last_control_at = now
            snapshot = self._lease_snapshot_locked(session_id)
        self.log.write(type="lease_acquired", client_id=session_id)
        self._publish_global({"type": "lease_changed"})
        return snapshot

    def release_lease(self, session_id: str) -> None:
        should_estop = False
        with self._lock:
            self._require_lease_owner_locked(session_id)
            if self._lease_is_pinned_locked():
                raise RuntimeConflict(
                    "command_busy", "Control cannot be released during a command."
                )
            state = self.controller.get_state()
            should_estop = state.moving or state.following
            self._lease = None
        if should_estop:
            self.activate_estop(
                actor_id=session_id,
                reason="Control was released while the robot was active.",
            )
        self.log.write(type="lease_released", client_id=session_id)
        self._publish_global({"type": "lease_changed"})

    def _renew_lease_locked(self, session_id: str) -> None:
        self._require_lease_owner_locked(session_id)
        assert self._lease is not None
        self._lease.last_control_at = self._clock()

    def _require_lease_owner_locked(self, session_id: str) -> None:
        if not self._lease or self._lease.owner_id != session_id:
            raise ControlRequired()

    def _lease_is_pinned_locked(self) -> bool:
        return bool(
            self._lease
            and self._active_job
            and self._active_job.owner_id == self._lease.owner_id
            and self._active_job.status not in self.TERMINAL_STATUSES
        )

    def _lease_snapshot_locked(self, viewer_id: str | None = None) -> dict | None:
        if not self._lease:
            return None
        elapsed = max(0.0, self._clock() - self._lease.last_control_at)
        remaining = max(0.0, self.config.server.lease_timeout_s - elapsed)
        if self._lease_is_pinned_locked():
            remaining = self.config.server.lease_timeout_s
        return {
            "owner_id": self._lease.owner_id,
            "owner_name": self._lease.owner_name,
            "acquired_at": self._lease.acquired_at,
            "expires_in_s": round(remaining, 1),
            "held_by_you": viewer_id == self._lease.owner_id,
            "pinned": self._lease_is_pinned_locked(),
        }

    # -- commands ---------------------------------------------------------

    def submit_command(self, session_id: str, text: str) -> CommandJob:
        text = text.strip()
        if not text:
            raise RuntimeConflict("invalid_command", "Command text cannot be empty.")
        if len(text) > self.config.server.max_command_chars:
            raise RuntimeConflict(
                "invalid_command",
                f"Command exceeds {self.config.server.max_command_chars} characters.",
            )
        self.expire_stale_state()
        with self._lock:
            self._require_session_locked(session_id)
            self._renew_lease_locked(session_id)
            if self._active_job and self._active_job.status not in self.TERMINAL_STATUSES:
                raise RuntimeConflict(
                    "command_busy", "Another command is already running."
                )
            job = CommandJob(
                id=str(uuid.uuid4()), owner_id=session_id, text=text
            )
            self._commands[job.id] = job
            self._active_job = job
            self._trim_commands_locked()
            self._command_executor.submit(self._run_job, job)
        self.log.write(
            type="network_command_submitted",
            command_id=job.id,
            client_id=session_id,
            text=text,
        )
        self._publish_global({"type": "command_changed", "command_id": job.id})
        return job

    def _run_job(self, job: CommandJob) -> None:
        with self._lock:
            session = self._sessions.get(job.owner_id)
            if not session:
                job.status = "failed"
                job.error = "The owning session no longer exists."
                job.completed_at = utc_now()
                self._active_job = None
                return
            job.status = "running"
            job.started_at = utc_now()
        self._publish_session(
            job.owner_id,
            {"type": "command_changed", "command_id": job.id, "status": "running"},
        )
        try:
            job.reply = session.loop.run_command(
                job.text,
                confirm=lambda skill, params, reason: self._await_confirmation(
                    job, skill, params, reason
                ),
                on_event=lambda kind, payload: self._record_command_event(
                    job, kind, payload
                ),
            )
            job.status = "completed"
        except Exception as exc:  # infrastructure errors become job failures
            job.status = "failed"
            job.error = str(exc)
            self.log.write(
                type="network_command_failed",
                command_id=job.id,
                client_id=job.owner_id,
                error=str(exc),
            )
        finally:
            with self._lock:
                job.completed_at = utc_now()
                if self._lease and self._lease.owner_id == job.owner_id:
                    self._lease.last_control_at = self._clock()
                if self._active_job is job:
                    self._active_job = None
                if session.revoked:
                    if self._lease and self._lease.owner_id == session.id:
                        self._lease = None
                    self._remove_session_locked(session.id)
            self.log.write(
                type="network_command_complete",
                command_id=job.id,
                client_id=job.owner_id,
                status=job.status,
            )
            self._publish_session(
                job.owner_id,
                {
                    "type": "command_changed",
                    "command_id": job.id,
                    "status": job.status,
                },
            )
            self._publish_global({"type": "state_changed"})

    def _record_command_event(
        self, job: CommandJob, kind: str, payload: dict
    ) -> None:
        event = {"kind": kind, "payload": payload, "ts": utc_now()}
        with self._lock:
            job.events.append(event)
        self._publish_session(
            job.owner_id,
            {
                "type": "command_event",
                "command_id": job.id,
                **event,
            },
        )
        self._publish_global({"type": "state_changed"})

    def _await_confirmation(
        self, job: CommandJob, skill: str, params: dict, reason: str
    ) -> bool:
        pending = PendingConfirmation(skill=skill, params=params, reason=reason)
        with self._lock:
            if self.estop_state.active:
                return False
            job.confirmation = pending
            job.status = "awaiting_confirmation"
        self._publish_session(
            job.owner_id,
            {
                "type": "confirmation_required",
                "command_id": job.id,
                "confirmation": pending.to_dict(),
                "timeout_s": self.config.server.confirmation_timeout_s,
            },
        )
        answered = pending.signal.wait(self.config.server.confirmation_timeout_s)
        with self._lock:
            if not answered or pending.response is None:
                pending.response = False
                pending.resolved_at = utc_now()
            approved = bool(pending.response and not self.estop_state.active)
            if job.status == "awaiting_confirmation":
                job.status = "running"
        self._publish_session(
            job.owner_id,
            {
                "type": "confirmation_resolved",
                "command_id": job.id,
                "approved": approved,
            },
        )
        return approved

    def resolve_confirmation(
        self, session_id: str, command_id: str, approved: bool
    ) -> None:
        with self._lock:
            self._renew_lease_locked(session_id)
            job = self._commands.get(command_id)
            if not job or job.owner_id != session_id:
                raise RuntimeNotFound("Command was not found.")
            pending = job.confirmation
            if job.status != "awaiting_confirmation" or not pending:
                raise RuntimeConflict(
                    "confirmation_unavailable", "No confirmation is pending."
                )
            if pending.response is not None:
                raise RuntimeConflict(
                    "confirmation_resolved", "Confirmation was already resolved."
                )
            if approved and self.estop_state.active:
                raise RuntimeConflict(
                    "estop_active", "Cannot approve while emergency stop is active."
                )
            pending.response = approved
            pending.resolved_at = utc_now()
            pending.signal.set()

    def get_command(self, session_id: str, command_id: str) -> dict:
        with self._lock:
            job = self._commands.get(command_id)
            if not job or job.owner_id != session_id:
                raise RuntimeNotFound("Command was not found.")
            return job.to_dict()

    def _trim_commands_locked(self) -> None:
        if len(self._commands) <= 500:
            return
        terminal = [
            job for job in self._commands.values() if job.status in self.TERMINAL_STATUSES
        ]
        terminal.sort(key=lambda job: job.completed_at or job.created_at)
        for job in terminal[: len(self._commands) - 500]:
            self._commands.pop(job.id, None)

    # -- emergency stop and reset ----------------------------------------

    def activate_estop(self, actor_id: str | None, reason: str) -> None:
        self.estop_state.activate()
        self.controller.emergency_stop()
        with self._lock:
            pending = self._active_job.confirmation if self._active_job else None
            if pending and pending.response is None:
                pending.response = False
                pending.resolved_at = utc_now()
                pending.signal.set()
        self.log.write(
            type="emergency_stop_activated", client_id=actor_id, reason=reason
        )
        self._publish_global(
            {"type": "estop_changed", "active": True, "reason": reason}
        )

    def release_estop(self, session_id: str) -> None:
        with self._lock:
            self._require_lease_owner_locked(session_id)
            if self._active_job and self._active_job.status not in self.TERMINAL_STATUSES:
                raise RuntimeConflict(
                    "command_busy", "Wait for the active command to finish."
                )
            self._renew_lease_locked(session_id)
        self.controller.release_emergency_stop()
        self.estop_state.release()
        self.log.write(type="emergency_stop_released", client_id=session_id)
        self._publish_global({"type": "estop_changed", "active": False})

    def reset(self, session_id: str) -> None:
        with self._lock:
            self._require_lease_owner_locked(session_id)
            if self._active_job and self._active_job.status not in self.TERMINAL_STATUSES:
                raise RuntimeConflict(
                    "command_busy", "Wait for the active command to finish."
                )
            if self.estop_state.active:
                raise RuntimeConflict(
                    "estop_active", "Release emergency stop before resetting."
                )
            self._renew_lease_locked(session_id)
            loops = [session.loop for session in self._sessions.values()]
        self.controller.reset()
        for loop in loops:
            loop.reset_conversation()
        self.log.write(type="network_reset", client_id=session_id)
        self._publish_global({"type": "state_changed"})

    # -- state/events -----------------------------------------------------

    def state_snapshot(self, viewer_id: str | None = None) -> dict:
        self.expire_stale_state()
        with self._lock:
            active = self._active_job
            return {
                "robot": self.controller.get_state().to_dict(),
                "backend": self.config.robot.backend,
                "estop_active": self.estop_state.active,
                "busy": bool(active and active.status not in self.TERMINAL_STATUSES),
                "active_command": (
                    {
                        "id": active.id,
                        "status": active.status,
                        "owned_by_you": active.owner_id == viewer_id,
                    }
                    if active and active.status not in self.TERMINAL_STATUSES
                    else None
                ),
                "lease": self._lease_snapshot_locked(viewer_id),
            }

    def subscribe(self, session_id: str) -> queue.Queue:
        subscriber: queue.Queue = queue.Queue(maxsize=100)
        with self._lock:
            session = self._require_session_locked(session_id)
            session.subscribers.add(subscriber)
        return subscriber

    def unsubscribe(self, session_id: str, subscriber: queue.Queue) -> None:
        with self._lock:
            session = self._sessions.get(session_id)
            if session:
                session.subscribers.discard(subscriber)

    def _publish_session(self, session_id: str, event: dict) -> None:
        event = {**event, "ts": event.get("ts", utc_now())}
        with self._lock:
            session = self._sessions.get(session_id)
            subscribers = list(session.subscribers) if session else []
        for subscriber in subscribers:
            self._queue_event(subscriber, event)

    def _publish_global(self, event: dict) -> None:
        event = {**event, "ts": event.get("ts", utc_now())}
        with self._lock:
            subscribers = [
                subscriber
                for session in self._sessions.values()
                for subscriber in session.subscribers
            ]
        for subscriber in subscribers:
            self._queue_event(subscriber, event)

    @staticmethod
    def _queue_event(subscriber: queue.Queue, event: dict) -> None:
        try:
            subscriber.put_nowait(event)
        except queue.Full:
            try:
                subscriber.get_nowait()
            except queue.Empty:
                pass
            try:
                subscriber.put_nowait(event)
            except queue.Full:
                pass

    # -- speech -----------------------------------------------------------

    def submit_transcription(
        self, session_id: str, data: bytes, content_type: str
    ) -> Future:
        with self._lock:
            self._renew_lease_locked(session_id)
        if len(data) > self.config.server.max_audio_bytes:
            raise RuntimeConflict(
                "audio_too_large",
                f"Audio exceeds {self.config.server.max_audio_bytes} bytes.",
            )

        def run() -> dict:
            if self._transcriber:
                return self._transcriber(data, content_type, self.config)
            from .speech.stt import transcribe_upload

            return transcribe_upload(data, content_type, self.config)

        return self._speech_executor.submit(run)
