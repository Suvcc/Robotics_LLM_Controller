"""FastAPI integration tests with two authenticated LAN clients."""

import time

from fastapi.testclient import TestClient

from aliengo.config import AppConfig
from aliengo.robot.mock import MockRobotController
from aliengo.runtime import RobotRuntime
from aliengo.web.app import create_app


class TextMessage:
    content = "Command complete."
    tool_calls = None


class TextLLM:
    last_usage = None

    def chat(self, messages, tools):
        return TextMessage()


def wait_for_command(client, command_id, headers):
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        response = client.get(f"/api/v1/commands/{command_id}", headers=headers)
        payload = response.json()
        if payload["status"] in {"completed", "failed", "cancelled"}:
            return payload
        time.sleep(0.01)
    raise AssertionError("command did not finish")


def login(client, name):
    response = client.post(
        "/api/v1/sessions",
        json={"display_name": name, "passcode": "correct-horse-battery"},
    )
    assert response.status_code == 201
    token = response.json()["access_token"]
    return response, {"Authorization": f"Bearer {token}"}


def test_auth_lease_two_clients_estop_and_command_flow():
    config = AppConfig()
    config.logging.actions_path = None
    runtime = RobotRuntime(
        config,
        "correct-horse-battery",
        controller=MockRobotController(),
        llm=TextLLM(),
        system_prompt="test prompt",
    )
    app = create_app(config, runtime=runtime)

    with TestClient(app, base_url="https://testserver") as client:
        bad = client.post(
            "/api/v1/sessions",
            json={"display_name": "Intruder", "passcode": "wrong"},
        )
        assert bad.status_code == 401
        assert bad.json()["error"]["code"] == "authentication_failed"
        invalid = client.post("/api/v1/sessions", json={})
        assert invalid.status_code == 422
        assert invalid.json()["error"]["code"] == "validation_error"

        alice_response, alice_headers = login(client, "Alice")
        _, bob_headers = login(client, "Bob")
        cookie = alice_response.headers["set-cookie"].lower()
        assert "secure" in cookie and "httponly" in cookie and "samesite=strict" in cookie

        assert client.put("/api/v1/lease", headers=alice_headers).status_code == 200
        denied = client.post(
            "/api/v1/commands", json={"text": "hello"}, headers=bob_headers
        )
        assert denied.status_code == 403

        assert client.post("/api/v1/estop", headers=bob_headers).status_code == 204
        state = client.get("/api/v1/state", headers=alice_headers).json()
        assert state["estop_active"] is True
        assert client.delete("/api/v1/estop", headers=alice_headers).status_code == 204

        submitted = client.post(
            "/api/v1/commands",
            json={"text": "report your state"},
            headers=alice_headers,
        )
        assert submitted.status_code == 202
        command = wait_for_command(client, submitted.json()["id"], alice_headers)
        assert command["status"] == "completed"
        assert command["reply"] == "Command complete."

        with client.websocket_connect(
            "/api/v1/events",
            subprotocols=["bearer", alice_headers["Authorization"].split(" ", 1)[1]],
        ) as websocket:
            assert websocket.receive_json()["type"] == "snapshot"


def test_transcription_requires_lease_and_never_executes_command():
    config = AppConfig()
    config.logging.actions_path = None
    controller = MockRobotController()
    runtime = RobotRuntime(
        config,
        "correct-horse-battery",
        controller=controller,
        llm=TextLLM(),
        system_prompt="test prompt",
        transcriber=lambda data, content_type, cfg: {
            "text": "stand up",
            "duration_s": 1.25,
        },
    )
    app = create_app(config, runtime=runtime)

    with TestClient(app, base_url="https://testserver") as client:
        _, headers = login(client, "Alice")
        denied = client.post(
            "/api/v1/transcriptions",
            files={"audio": ("clip.webm", b"fake", "audio/webm")},
            headers=headers,
        )
        assert denied.status_code == 403
        client.put("/api/v1/lease", headers=headers)
        response = client.post(
            "/api/v1/transcriptions",
            files={"audio": ("clip.webm", b"fake", "audio/webm")},
            headers=headers,
        )
        assert response.status_code == 200
        assert response.json()["text"] == "stand up"
        assert controller.get_state().posture.value == "sitting"


def test_static_browser_client_and_health_are_available():
    config = AppConfig()
    config.logging.actions_path = None
    runtime = RobotRuntime(
        config,
        "correct-horse-battery",
        controller=MockRobotController(),
        llm=TextLLM(),
        system_prompt="test prompt",
    )
    app = create_app(config, runtime=runtime)
    with TestClient(app, base_url="https://testserver") as client:
        assert client.get("/api/v1/health").json() == {"status": "ok"}
        page = client.get("/")
        assert page.status_code == 200
        assert "AlienGo Network Control" in page.text
        assert client.get("/static/app.css").status_code == 200
