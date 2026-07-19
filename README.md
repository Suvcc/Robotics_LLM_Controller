# AlienGo Local LLM System

Mock-first intelligence and control stack for an AlienGo robot dog. A local or
OpenAI-compatible LLM selects approved skills, every tool call passes through a
deterministic safety gate, and an interchangeable robot controller executes it.

The project supports two operator surfaces:

- `aliengo` — terminal chat on the host PC.
- `aliengo-server` — authenticated HTTPS control from browsers and API clients
  on the same private network.

Ollama stays on `localhost`; network clients can reach only the application API,
so they cannot bypass skill validation or robot safety.

## Setup

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```powershell
uv sync
uv run pytest

# Local terminal control (Ollama + mock robot by default)
uv run aliengo

# OpenAI-compatible preset; put OPENAI_API_KEY in .env first
uv run aliengo --config config.openai.yaml
```

The default model is `qwen3.5:9b` at Ollama's OpenAI-compatible endpoint. Change
`llm.model` in `config.yaml` to use another installed model.

## Private-LAN server

The web server uses trusted HTTPS because phone and desktop browsers allow
microphone capture only in a secure context. The following setup uses
[mkcert](https://github.com/FiloSottile/mkcert) to create a private certificate
authority for your LAN devices.

### 1. Reserve an address

Give the host PC a stable DHCP reservation in the router. Choose a stable name
such as `aliengo.local` if it resolves on every intended client, and note the
host's private IP from `ipconfig`.

### 2. Create and trust a certificate

Install mkcert on the host, then run from the project root, replacing the sample
IP with the host's reserved address:

```powershell
mkcert -install
New-Item -ItemType Directory -Force .certs
mkcert -cert-file .certs/aliengo.pem -key-file .certs/aliengo-key.pem aliengo.local 192.168.1.50 localhost 127.0.0.1
mkcert -CAROOT
```

`mkcert -CAROOT` prints the directory containing `rootCA.pem`. Install and trust
that CA certificate on each phone or computer that will open the control page.
Never share `rootCA-key.pem`. The entire `.certs/` project directory is
gitignored.

### 3. Configure the passcode

Create `.env` from `.env.example` and set a unique passcode of at least 12
characters:

```dotenv
ALIENGO_SERVER_PASSCODE=replace-this-with-a-long-random-secret
```

### 4. Start the host

```powershell
uv run aliengo-server
```

Allow TCP port `8443` only on the Windows **Private** firewall profile. Other
devices can then open `https://aliengo.local:8443` or the certified private IP.
Do not expose this service through router port forwarding or guest Wi-Fi.

The generated API documentation is available at `/api/docs`. External clients
create a session through `POST /api/v1/sessions`, then send the returned token
as `Authorization: Bearer <token>`.

## Network safety model

- One authenticated client holds a five-minute control lease.
- There is at most one active command and no command backlog.
- Every authenticated client can activate emergency stop immediately.
- Only the lease holder can release e-stop, reset, or approve risky actions.
- Each client has private LLM conversation history; robot and e-stop state are
  global.
- Lease expiry or release while following activates the latched e-stop.
- Browser recordings are limited to 15 seconds and 10 MB, transcribed locally,
  and placed into the editor for review; transcription never executes a command.

Hardware mode remains opt-in through `robot.backend: jetracer`. Calibrate every
`PLACEHOLDER` in the JetRacer and detector modules before driving. A separate
hardware watchdog is still required to protect against total PC, process, or
power failure.

## Architecture

```text
terminal / HTTPS browser / API
              |
              v
       per-client AgentLoop
              |
              v
LLM -> tool call -> safety.validate -> RobotController -> SkillResult
                                      |
                              mock or JetRacer
```

- `aliengo/skills/definitions.py` — skill schemas and parameter bounds.
- `aliengo/safety/validator.py` — deterministic six-stage safety gate.
- `aliengo/runtime.py` — sessions, lease, command serialization, and global e-stop.
- `aliengo/web/` — versioned API, WebSocket events, and browser client.
- `aliengo/robot/interface.py` — backend protocol and future ROS/Unitree seam.
- `scripts/evaluate.py` — live-model benchmark over `tests/eval_commands.json`.

Action and server audit events are appended to `logs/actions.jsonl`. Tests use a
scripted fake LLM and run offline without Ollama, a GPU, microphone, or robot.
