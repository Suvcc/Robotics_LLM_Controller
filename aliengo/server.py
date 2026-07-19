"""HTTPS entrypoint for the private-LAN AlienGo server."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

from .config import load_config
from .runtime import PROJECT_ROOT
from .web.app import create_app


def _resolve_path(path: str, config_path: Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return (config_path.parent / candidate).resolve()


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    parser = argparse.ArgumentParser(description="AlienGo private-LAN HTTPS server")
    parser.add_argument(
        "--config", default=str(PROJECT_ROOT / "config.yaml"), help="YAML config path"
    )
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = load_config(config_path)

    passcode = os.environ.get("ALIENGO_SERVER_PASSCODE", "")
    if len(passcode) < 12:
        raise SystemExit(
            "Set ALIENGO_SERVER_PASSCODE to a secret of at least 12 characters."
        )
    certfile = _resolve_path(config.server.tls_certfile, config_path)
    keyfile = _resolve_path(config.server.tls_keyfile, config_path)
    if not certfile.is_file() or not keyfile.is_file():
        raise SystemExit(
            "Trusted TLS certificate files are required. See README.md for mkcert setup.\n"
            f"Expected certificate: {certfile}\nExpected key: {keyfile}"
        )

    import uvicorn

    uvicorn.run(
        create_app(config, passcode=passcode),
        host=config.server.host,
        port=config.server.port,
        ssl_certfile=str(certfile),
        ssl_keyfile=str(keyfile),
        workers=1,
    )


if __name__ == "__main__":
    main()
