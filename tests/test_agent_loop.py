"""Agent-loop tests with a scripted fake LLM — no GPU or Ollama needed."""
import json
from dataclasses import dataclass, field

import pytest

from aliengo.agent.loop import AgentLoop
from aliengo.config import AppConfig
from aliengo.robot.mock import MockRobotController
from aliengo.skills.definitions import Posture
from aliengo.skills.registry import to_openai_tools


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


def tool_msg(*calls: tuple[str, dict | str]) -> FakeMessage:
    tool_calls = []
    for i, (name, args) in enumerate(calls):
        arguments = args if isinstance(args, str) else json.dumps(args)
        tool_calls.append(FakeToolCall(id=f"call_{name}_{i}", function=FakeFunction(name, arguments)))
    return FakeMessage(tool_calls=tool_calls)


def text_msg(text: str) -> FakeMessage:
    return FakeMessage(content=text)


@dataclass
class FakeLLM:
    script: list[FakeMessage]
    seen: list[list[dict]] = field(default_factory=list)
    _i: int = 0

    def chat(self, messages, tools):
        self.seen.append([dict(m) for m in messages])
        message = self.script[min(self._i, len(self.script) - 1)]
        self._i += 1
        return message


def make_loop(script, *, config=None, confirm=None):
    controller = MockRobotController()
    loop = AgentLoop(
        llm=FakeLLM(script),
        controller=controller,
        config=config or AppConfig(),
        system_prompt="test prompt",
        tools=to_openai_tools(),
        confirm=confirm,
    )
    return loop, controller


def last_tool_payloads(loop) -> list[dict]:
    return [
        json.loads(m["content"]) for m in loop.history if m.get("role") == "tool"
    ]


def test_multi_step_command():
    loop, controller = make_loop([
        tool_msg(("stand_up", {})),
        tool_msg(("move_forward", {"distance": 2})),
        text_msg("Done, I moved forward two meters."),
    ])
    reply = loop.run_command("stand up and move forward two meters")
    assert reply == "Done, I moved forward two meters."
    state = controller.get_state()
    assert state.posture is Posture.STANDING
    assert state.x == pytest.approx(2.0)
    assert loop.last_command_stats["llm_calls"] == 3
    assert loop.last_command_stats["malformed_calls"] == 0


def test_safety_block_is_fed_back_to_llm():
    loop, controller = make_loop([
        tool_msg(("move_forward", {"distance": 100})),
        text_msg("That distance exceeds my limit."),
    ])
    reply = loop.run_command("move forward 100 meters")
    assert "limit" in reply
    payloads = last_tool_payloads(loop)
    assert payloads[0]["blocked"] is True
    assert "distance" in payloads[0]["error"]
    assert controller.get_state().x == 0.0


def test_skill_failure_is_reported_honestly():
    loop, controller = make_loop([
        tool_msg(("stand_up", {})),
        tool_msg(("move_forward", {"distance": 1})),
        text_msg("The move failed: motor stall."),
    ])
    controller.inject_failure("move_forward", "motor stall")
    reply = loop.run_command("go forward")
    assert "motor stall" in reply
    payloads = last_tool_payloads(loop)
    assert payloads[-1]["success"] is False
    assert payloads[-1]["error"] == "motor stall"


def test_batched_calls_after_block_are_skipped():
    # Model batches move (blocked: sitting) + stand_up in one message.
    loop, controller = make_loop([
        tool_msg(("move_forward", {"distance": 1}), ("stand_up", {})),
        text_msg("I need to stand up first."),
    ])
    loop.run_command("go forward")
    payloads = last_tool_payloads(loop)
    assert len(payloads) == 2  # every tool_call got a response
    assert payloads[0]["blocked"] is True
    assert payloads[1]["error"].startswith("Not executed")
    assert controller.get_state().posture is Posture.SITTING


def test_malformed_arguments_abort_after_retries():
    loop, _ = make_loop([
        tool_msg(("move_forward", "{not json")),
        tool_msg(("move_forward", "{not json")),
        tool_msg(("move_forward", "{not json")),
        text_msg("should never get here"),
    ])
    reply = loop.run_command("go forward")
    assert "invalid tool calls" in reply
    assert loop.last_command_stats["malformed_calls"] == 3


def test_iteration_cap():
    config = AppConfig()
    config.llm.max_iterations = 3
    loop, _ = make_loop([tool_msg(("stop", {}))], config=config)  # loops forever
    reply = loop.run_command("stop")
    assert reply.startswith("Stopped: reached the maximum")


def test_confirmation_declined():
    loop, controller = make_loop(
        [
            tool_msg(("stand_up", {})),
            tool_msg(("follow_person", {})),
            text_msg("Okay, I won't follow anyone."),
        ],
        confirm=lambda skill, params, reason: False,
    )
    loop.run_command("follow that person")
    payloads = last_tool_payloads(loop)
    assert payloads[-1]["error"] == "User declined confirmation."
    assert controller.get_state().following is False


def test_confirmation_accepted():
    loop, controller = make_loop(
        [
            tool_msg(("stand_up", {})),
            tool_msg(("follow_person", {})),
            text_msg("Following now."),
        ],
        confirm=lambda skill, params, reason: True,
    )
    loop.run_command("follow that person")
    assert controller.get_state().following is True


def test_history_trimmed_at_command_boundaries():
    config = AppConfig()
    config.llm.max_history_commands = 2
    loop, _ = make_loop(
        [tool_msg(("stand_up", {})), text_msg("ok")] * 10, config=config
    )
    for i in range(5):
        loop.run_command(f"command {i}")

    user_messages = [m["content"] for m in loop.history if m["role"] == "user"]
    # Content carries the injected state prefix, so match on the command tail.
    assert [m.split("\n")[-1] for m in user_messages] == ["command 3", "command 4"]
    assert loop.history[0]["role"] == "system"
    # No orphaned tool message right after the system prompt.
    assert loop.history[1]["role"] == "user"


def test_state_prefix_injected_into_command():
    loop, _ = make_loop([text_msg("ok")])
    loop.run_command("stand up")
    user_msg = next(m["content"] for m in loop.history if m["role"] == "user")
    assert "Robot state:" in user_msg
    assert user_msg.endswith("stand up")


def test_state_injection_can_be_disabled():
    config = AppConfig()
    config.llm.inject_state = False
    loop, _ = make_loop([text_msg("ok")], config=config)
    loop.run_command("stand up")
    user_msg = next(m["content"] for m in loop.history if m["role"] == "user")
    assert user_msg == "stand up"


def test_format_state_prefix_detail():
    from aliengo.agent.loop import format_state_prefix
    from aliengo.robot.interface import RobotState
    from aliengo.skills.definitions import Posture

    sitting = format_state_prefix(RobotState())
    assert "sitting" in sitting and "stationary" in sitting and "100%" in sitting

    following = format_state_prefix(
        RobotState(posture=Posture.STANDING, following=True, heading_deg=90)
    )
    assert "standing" in following and "following a person" in following
    assert "facing 90°" in following


def test_redundant_stand_up_is_noop_success_not_block():
    # Regression for stale-plan carryover: a redundant stand_up mid-task must
    # come back as a success the LLM can continue past, never a block that
    # derails the rest of the command.
    loop, controller = make_loop([
        tool_msg(("stand_up", {}), ("stand_up", {})),
        text_msg("Already standing."),
    ])
    loop.run_command("stand up")
    payloads = last_tool_payloads(loop)
    assert all(p["success"] for p in payloads)
    assert "Continue with the task" in payloads[1]["data"]["note"]
    assert controller.get_state().posture is Posture.STANDING


def test_estop_blocks_but_stop_runs():
    loop, controller = make_loop([
        tool_msg(("stand_up", {})),
        tool_msg(("stop", {})),
        text_msg("Everything is stopped."),
    ])
    loop.estop_active = True
    loop.run_command("stand up then stop")
    payloads = last_tool_payloads(loop)
    assert payloads[0]["blocked"] is True  # stand_up blocked under e-stop
    assert payloads[1]["success"] is True  # stop still ran
    assert controller.get_state().posture is Posture.SITTING
