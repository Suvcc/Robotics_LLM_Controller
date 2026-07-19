"""Threaded LAN runtime tests; all LLM and robot behavior remains offline."""

import json
import threading
import time
from dataclasses import dataclass

import pytest

from aliengo.config import AppConfig
from aliengo.robot.mock import MockRobotController
from aliengo.runtime import (
    AuthenticationFailed,
    ControlRequired,
    RateLimited,
    RobotRuntime,
    RuntimeConflict,
)


@dataclass
class FakeFunction:
    name: str
    arguments: str


@dataclass
class FakeToolCall:
    id: str
    function: FakeFunction


@dataclass
class FakeMessage:
    content: str | None = None
    tool_calls: list | None = None


def tool_message(name: str, args: dict) -> FakeMessage:
    return FakeMessage(
        tool_calls=[
            FakeToolCall(
                id=f"call_{name}",
                function=FakeFunction(name=name, arguments=json.dumps(args)),
            )
        ]
    )


class TextLLM:
    last_usage = None

    def chat(self, messages, tools):
        return FakeMessage(content="Done safely.")


class ScriptedLLM:
    last_usage = None

    def __init__(self, messages):
        self.messages = list(messages)
        self.index = 0

    def chat(self, messages, tools):
        result = self.messages[min(self.index, len(self.messages) - 1)]
        self.index += 1
        return result


class BlockingLLM:
    last_usage = None

    def __init__(self):
        self.started = threading.Event()
        self.release = threading.Event()

    def chat(self, messages, tools):
        self.started.set()
        self.release.wait(timeout=2)
        return FakeMessage(content="Finished.")


class FakeClock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now


def wait_for(predicate, timeout=2):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("condition was not reached before timeout")


def make_runtime(*, llm=None, clock=time.monotonic, config=None):
    config = config or AppConfig()
    config.logging.actions_path = None
    return RobotRuntime(
        config,
        "correct-horse-battery",
        controller=MockRobotController(),
        llm=llm or TextLLM(),
        system_prompt="test prompt",
        clock=clock,
    )


def test_sessions_have_isolated_histories_and_share_robot():
    runtime = make_runtime()
    try:
        alice = runtime.create_session("correct-horse-battery", "Alice", "a")
        bob = runtime.create_session("correct-horse-battery", "Bob", "b")
        runtime.acquire_lease(alice.id)
        alice_job = runtime.submit_command(alice.id, "alice command")
        wait_for(lambda: alice_job.status == "completed")
        runtime.release_lease(alice.id)
        runtime.acquire_lease(bob.id)
        bob_job = runtime.submit_command(bob.id, "bob command")
        wait_for(lambda: bob_job.status == "completed")

        alice_users = [m["content"] for m in alice.loop.history if m["role"] == "user"]
        bob_users = [m["content"] for m in bob.loop.history if m["role"] == "user"]
        assert alice_users[0].endswith("alice command")
        assert bob_users[0].endswith("bob command")
        assert all("bob command" not in message for message in alice_users)
    finally:
        runtime.close()


def test_runtime_rejects_backlog_while_command_is_running():
    llm = BlockingLLM()
    runtime = make_runtime(llm=llm)
    try:
        session = runtime.create_session("correct-horse-battery", "Alice", "a")
        runtime.acquire_lease(session.id)
        runtime.submit_command(session.id, "first")
        assert llm.started.wait(timeout=1)
        with pytest.raises(RuntimeConflict) as exc:
            runtime.submit_command(session.id, "second")
        assert exc.value.code == "command_busy"
    finally:
        llm.release.set()
        runtime.close()


def test_confirmation_owner_can_approve_and_other_session_cannot():
    llm = ScriptedLLM(
        [
            tool_message("stand_up", {}),
            tool_message("follow_person", {}),
            FakeMessage(content="Following."),
        ]
    )
    runtime = make_runtime(llm=llm)
    try:
        alice = runtime.create_session("correct-horse-battery", "Alice", "a")
        bob = runtime.create_session("correct-horse-battery", "Bob", "b")
        runtime.acquire_lease(alice.id)
        job = runtime.submit_command(alice.id, "follow that person")
        wait_for(lambda: job.status == "awaiting_confirmation")
        with pytest.raises(ControlRequired):
            runtime.resolve_confirmation(bob.id, job.id, True)
        runtime.resolve_confirmation(alice.id, job.id, True)
        wait_for(lambda: job.status == "completed")
        assert runtime.controller.get_state().following is True
    finally:
        runtime.close()


def test_estop_declines_pending_confirmation_and_blocks_execution():
    llm = ScriptedLLM(
        [
            tool_message("stand_up", {}),
            tool_message("follow_person", {}),
            FakeMessage(content="Stopped."),
        ]
    )
    runtime = make_runtime(llm=llm)
    try:
        alice = runtime.create_session("correct-horse-battery", "Alice", "a")
        observer = runtime.create_session("correct-horse-battery", "Observer", "b")
        runtime.acquire_lease(alice.id)
        job = runtime.submit_command(alice.id, "follow that person")
        wait_for(lambda: job.status == "awaiting_confirmation")
        runtime.activate_estop(observer.id, "Observer pressed e-stop.")
        wait_for(lambda: job.status == "completed")
        assert job.confirmation.response is False
        assert runtime.estop_state.active
        assert not runtime.controller.get_state().following
    finally:
        runtime.close()


def test_expired_lease_estops_a_background_follow():
    clock = FakeClock()
    config = AppConfig()
    config.server.lease_timeout_s = 5
    runtime = make_runtime(clock=clock, config=config)
    try:
        session = runtime.create_session("correct-horse-battery", "Alice", "a")
        runtime.acquire_lease(session.id)
        runtime.controller.stand_up()
        runtime.controller.follow_person()
        clock.now += 6
        runtime.expire_stale_state()
        snapshot = runtime.state_snapshot(session.id)
        assert snapshot["lease"] is None
        assert snapshot["estop_active"] is True
        assert snapshot["robot"]["following"] is False
    finally:
        runtime.close()


def test_confirmation_times_out_as_declined():
    config = AppConfig()
    config.server.confirmation_timeout_s = 0.05
    llm = ScriptedLLM(
        [
            tool_message("stand_up", {}),
            tool_message("follow_person", {}),
            FakeMessage(content="Not approved."),
        ]
    )
    runtime = make_runtime(llm=llm, config=config)
    try:
        session = runtime.create_session("correct-horse-battery", "Alice", "a")
        runtime.acquire_lease(session.id)
        job = runtime.submit_command(session.id, "follow that person")
        wait_for(lambda: job.status == "completed")
        assert job.confirmation.response is False
        assert runtime.controller.get_state().following is False
    finally:
        runtime.close()


def test_authentication_attempts_are_rate_limited_per_source():
    config = AppConfig()
    config.server.auth_attempts_per_minute = 2
    runtime = make_runtime(config=config)
    try:
        for _ in range(2):
            with pytest.raises(AuthenticationFailed):
                runtime.create_session("wrong-passcode", "Alice", "same-ip")
        with pytest.raises(RateLimited):
            runtime.create_session("correct-horse-battery", "Alice", "same-ip")
    finally:
        runtime.close()
