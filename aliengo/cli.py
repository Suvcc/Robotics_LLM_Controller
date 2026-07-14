"""Terminal chat REPL: type commands, watch tool calls, safety verdicts, and
the mock robot execute them."""
import argparse
import json
from pathlib import Path

from dotenv import load_dotenv
from openai import APIConnectionError, AuthenticationError
from rich.console import Console
from rich.prompt import Confirm

from .actionlog import ActionLog
from .agent.llm import LLMClient
from .agent.loop import AgentLoop
from .config import load_config
from .robot.mock import MockRobotController
from .skills.registry import to_openai_tools

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROMPT_PATH = PROJECT_ROOT / "prompts" / "system_prompt.md"

FALLBACK_SYSTEM_PROMPT = (
    "You are the control brain of AlienGo, a quadruped robot dog. Act only "
    "through the provided tools. Distances are meters, angles degrees. Stand "
    "up before moving. Ask when a command is ambiguous; refuse when no skill "
    "matches. Act only on the latest message; never resume earlier tasks. "
    "Report failures honestly and keep replies short."
)

HELP = """\
Commands: anything in plain English, or:
  /state   show robot state
  /log [n] show last n log entries (default 5)
  /estop   toggle the emergency stop
  /reset   reset robot and conversation
  /help    this help
  /quit    exit"""

console = Console()


def _load_system_prompt() -> str:
    if PROMPT_PATH.exists():
        return PROMPT_PATH.read_text(encoding="utf-8")
    return FALLBACK_SYSTEM_PROMPT


def _print_state(controller: MockRobotController, estop: bool) -> None:
    s = controller.get_state().to_dict()
    estop_str = " [bold red]E-STOP[/]" if estop else ""
    console.print(
        f"[dim]state:[/] {s['posture']} | pos ({s['x']}, {s['y']}) "
        f"heading {s['heading_deg']}° | battery {s['battery_pct']}%{estop_str}"
    )


def _print_log(config, n: int) -> None:
    path = Path(config.logging.actions_path or "")
    if not path.exists():
        console.print("[dim]no log yet[/]")
        return
    lines = path.read_text(encoding="utf-8").strip().splitlines()[-n:]
    for line in lines:
        entry = json.loads(line)
        console.print(f"[dim]{entry.get('ts', '')[:19]}[/] {entry}")


def _on_event(kind: str, payload: dict) -> None:
    if kind == "tool_call":
        args = ", ".join(f"{k}={v}" for k, v in payload["args"].items())
        console.print(f"  [cyan]→ {payload['skill']}({args})[/]")
    elif kind == "safety":
        decision = payload["decision"]
        if decision == "allow":
            console.print("    [green]safety: allow[/]")
        elif decision == "block":
            console.print(f"    [red]safety: BLOCK — {payload['reason']}[/]")
        else:
            console.print(f"    [yellow]safety: confirm — {payload['reason']}[/]")
    elif kind == "result":
        if payload["success"]:
            extra = f" {payload['data']}" if payload.get("data") else ""
            console.print(f"    [green]✓ {payload['action']} ok{extra}[/]")
        else:
            console.print(f"    [red]✗ {payload['action']}: {payload.get('error')}[/]")
    elif kind == "info":
        console.print(f"    [yellow]{payload['text']}[/]")


def _confirm(skill: str, params: dict, reason: str) -> bool:
    return Confirm.ask(f"[yellow]{reason}[/] Run [bold]{skill}[/]?", default=False)


def main() -> None:
    # Pick up OPENAI_API_KEY from a project .env file; a value already set in
    # the shell wins over the file.
    load_dotenv(PROJECT_ROOT / ".env")
    parser = argparse.ArgumentParser(description="AlienGo mock control REPL")
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "config.yaml"),
        help="Config file to use, e.g. config.openai.yaml",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    log = ActionLog(config.logging.actions_path)
    controller = MockRobotController(log=log)
    system_prompt = _load_system_prompt()
    loop = AgentLoop(
        llm=LLMClient(config.llm),
        controller=controller,
        config=config,
        system_prompt=system_prompt,
        tools=to_openai_tools(),
        confirm=_confirm,
        on_event=_on_event,
        log=log,
    )

    console.print(f"[bold]AlienGo mock control[/] — model {config.llm.model}")
    console.print(HELP)
    _print_state(controller, loop.estop_active)

    while True:
        try:
            text = console.input("[bold blue]you>[/] ").strip()
        except (KeyboardInterrupt, EOFError):
            break
        if not text:
            continue

        if text.startswith("/"):
            cmd, _, arg = text.partition(" ")
            if cmd == "/quit":
                break
            elif cmd == "/help":
                console.print(HELP)
            elif cmd == "/state":
                _print_state(controller, loop.estop_active)
            elif cmd == "/log":
                _print_log(config, int(arg) if arg.isdigit() else 5)
            elif cmd == "/estop":
                loop.estop_active = not loop.estop_active
                state = "ACTIVE" if loop.estop_active else "released"
                console.print(f"[bold red]emergency stop {state}[/]")
            elif cmd == "/reset":
                controller.__init__(log=log)
                loop.reset_conversation()
                loop.estop_active = False
                console.print("[dim]robot and conversation reset[/]")
            else:
                console.print(f"unknown command {cmd} — try /help")
            continue

        try:
            reply = loop.run_command(text)
        except APIConnectionError:
            console.print(
                f"[red]Cannot reach the LLM at {config.llm.base_url}. "
                "If using Ollama, start it with 'ollama serve'.[/]"
            )
            continue
        except AuthenticationError:
            console.print(
                "[red]Authentication failed. Set your key in this shell first: "
                '$env:OPENAI_API_KEY = "sk-..."[/]'
            )
            continue
        console.print(f"[bold magenta]dog>[/] {reply}")
        _print_state(controller, loop.estop_active)


if __name__ == "__main__":
    main()
