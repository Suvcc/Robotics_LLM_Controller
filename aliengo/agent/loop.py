"""The tool-calling agent loop.

Pipeline per tool call: LLM -> safety.validate -> controller. The controller
is never called except through an ALLOW (or operator-confirmed CONFIRM)
decision; BLOCK reasons and skill errors are fed back to the LLM as tool
results so it can recover.
"""
import json
from typing import Callable

from ..actionlog import ActionLog
from ..config import AppConfig
from ..robot.interface import RobotController, execute_skill
from ..safety.validator import Decision, SafetySession, validate

MAX_MALFORMED_RETRIES = 2

# confirm(skill_name, params, reason) -> bool
ConfirmFn = Callable[[str, dict, str], bool]
# on_event(kind, payload) — kinds: tool_call, safety, result, info
EventFn = Callable[[str, dict], None]


class AgentLoop:
    def __init__(
        self,
        llm,
        controller: RobotController,
        config: AppConfig,
        system_prompt: str,
        tools: list[dict],
        confirm: ConfirmFn | None = None,
        on_event: EventFn | None = None,
        log: ActionLog | None = None,
    ):
        self.llm = llm
        self.controller = controller
        self.config = config
        self.tools = tools
        self.confirm = confirm
        self.on_event = on_event or (lambda kind, payload: None)
        self.log = log or ActionLog(None)
        self.estop_active = False
        self.history: list[dict] = [{"role": "system", "content": system_prompt}]

    def reset_conversation(self, system_prompt: str | None = None) -> None:
        system = system_prompt or self.history[0]["content"]
        self.history = [{"role": "system", "content": system}]

    def run_command(self, user_text: str) -> str:
        self.history.append({"role": "user", "content": user_text})
        self.log.write(type="user_command", text=user_text)
        session = SafetySession(estop_active=self.estop_active)
        malformed_count = 0

        for _ in range(self.config.llm.max_iterations):
            message = self.llm.chat(self.history, self.tools)
            tool_calls = list(message.tool_calls or [])
            assistant: dict = {"role": "assistant", "content": message.content or ""}
            if tool_calls:
                assistant["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in tool_calls
                ]
            self.history.append(assistant)

            if not tool_calls:
                final = message.content or ""
                self.log.write(type="llm_response", text=final)
                return final

            # Robot actions are strictly sequential; after a block/failure the
            # rest of the batch is skipped (with an honest tool result) so the
            # LLM can re-plan with full knowledge.
            skip_reason: str | None = None
            for tc in tool_calls:
                name = tc.function.name
                if skip_reason:
                    self._tool_result(tc, {"success": False, "error": f"Not executed: {skip_reason}."})
                    continue

                try:
                    raw_args = json.loads(tc.function.arguments or "{}")
                    if not isinstance(raw_args, dict):
                        raise ValueError("arguments must be a JSON object")
                except (json.JSONDecodeError, ValueError):
                    malformed_count += 1
                    self.on_event("info", {"text": f"Malformed arguments for {name}."})
                    self._tool_result(
                        tc,
                        {"success": False, "error": "Arguments were not a valid JSON object. Retry with correct JSON."},
                    )
                    continue

                self.on_event("tool_call", {"skill": name, "args": raw_args})
                session.estop_active = self.estop_active
                decision = validate(
                    name, raw_args, self.controller.get_state(), session, self.config.safety
                )
                self.log.write(
                    type="safety", skill=name, args=raw_args,
                    decision=decision.decision.value, reason=decision.reason,
                )
                self.on_event(
                    "safety",
                    {"skill": name, "decision": decision.decision.value, "reason": decision.reason},
                )

                if decision.decision is Decision.BLOCK:
                    self._tool_result(tc, {"success": False, "blocked": True, "error": decision.reason})
                    skip_reason = f"'{name}' was blocked by the safety layer"
                    continue

                if decision.decision is Decision.CONFIRM:
                    approved = bool(self.confirm and self.confirm(name, decision.params, decision.reason))
                    if not approved:
                        self._tool_result(tc, {"success": False, "error": "User declined confirmation."})
                        skip_reason = f"the user declined '{name}'"
                        continue

                result = execute_skill(self.controller, name, decision.params)
                session.tool_calls_made += 1
                if result.success:
                    session.distance_moved_m += decision.params.get("distance", 0.0)
                self.on_event("result", result.to_dict())
                self._tool_result(tc, result.to_dict())
                if not result.success:
                    skip_reason = f"'{name}' failed"

            if malformed_count > MAX_MALFORMED_RETRIES:
                return (
                    "I kept producing invalid tool calls and stopped for safety. "
                    "Please rephrase the command."
                )

        return "Stopped: reached the maximum number of steps for one command."

    def _tool_result(self, tool_call, payload: dict) -> None:
        self.history.append(
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(payload, ensure_ascii=False),
            }
        )
