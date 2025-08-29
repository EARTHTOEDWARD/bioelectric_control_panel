from __future__ import annotations

"""
BCP WebSocket Bridge — broadcasts NDJSON BCP-Events to connected clients.

Usage:
  bcp-ws-bridge --events /path/to/events.ndjson --host 127.0.0.1 --port 8765 --mode replay --speed 1.0 --static
  bcp-ws-bridge --events /path/to/events.ndjson --mode tail --static

Endpoints:
  - WebSocket: ws://<host>:<port>/ws
  - Optional static demo: http://<host>:<port+1>/client.html (when --static)

Requires: websockets (install via extras: pip install -e .[ws])
"""

import argparse
import asyncio
import json
import os
import sys
import time
import threading
from pathlib import Path
from typing import Iterable

try:
    import websockets  # type: ignore
except Exception as e:  # pragma: no cover - import error surfaced at runtime
    websockets = None


CLIENTS: set = set()


async def _broadcast(line: str) -> None:
    dead = set()
    for ws in CLIENTS.copy():
        try:
            await ws.send(line)
        except Exception:
            dead.add(ws)
    for ws in dead:
        CLIENTS.discard(ws)


async def _handler(ws, path):
    if path != "/ws":
        await ws.close()
        return
    CLIENTS.add(ws)
    try:
        async for _ in ws:
            pass
    finally:
        CLIENTS.discard(ws)


def _tail_lines(path: Path) -> Iterable[str]:
    with path.open("r", encoding="utf-8") as f:
        f.seek(0, os.SEEK_END)
        while True:
            pos = f.tell()
            line = f.readline()
            if not line:
                time.sleep(0.1)
                f.seek(pos)
                continue
            yield line.rstrip("\n")


def _replay_lines(path: Path, speed: float = 1.0) -> Iterable[str]:
    last_t = None
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                t = float(obj.get("t_window", 0.0))
            except Exception:
                t = None
            if last_t is None or t is None:
                last_t = t
                yield line
                continue
            dt = max(0.0, (t - last_t)) / max(1e-6, speed)
            if dt > 2.0:
                dt = 2.0 / max(1e-6, speed)
            time.sleep(dt)
            last_t = t
            yield line


def _serve_static(static_dir: Path, host: str, port: int) -> None:
    import http.server
    import socketserver

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(static_dir), **kwargs)

        def log_message(self, fmt, *args):  # noqa: N802 - quiet logs
            pass

    with socketserver.TCPServer((host, port), Handler) as httpd:
        httpd.serve_forever()


def _static_dir() -> Path:
    # Locate bcp/tools/public next to this file
    here = Path(__file__).resolve().parent
    return here / "public"


def main() -> None:  # pragma: no cover - CLI
    ap = argparse.ArgumentParser(description="BCP WebSocket bridge for NDJSON events")
    ap.add_argument("--events", required=True, help="Path to NDJSON events file")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--mode", choices=["replay", "tail"], default="replay")
    ap.add_argument("--speed", type=float, default=1.0, help="Replay speed multiplier")
    ap.add_argument("--static", action="store_true", help="Serve demo client (public/) on port+1")
    args = ap.parse_args()

    if websockets is None:
        print("[ERR] websockets not installed. pip install -e .[ws]", file=sys.stderr)
        sys.exit(2)

    events_path = Path(args.events)
    if not events_path.exists():
        print(f"[ERR] events file not found: {events_path}", file=sys.stderr)
        sys.exit(1)

    if args.static:
        static_dir = _static_dir()
        port_static = int(args.port) + 1
        th = threading.Thread(
            target=_serve_static, args=(static_dir, str(args.host), int(port_static)), daemon=True
        )
        th.start()
        print(f"[i] Demo client: http://{args.host}:{port_static}/client.html (serving {static_dir})")

    print(f"[i] WebSocket: ws://{args.host}:{args.port}/ws")

    async def _run() -> None:
        ws_server = await websockets.serve(_handler, str(args.host), int(args.port))
        try:
            src = _tail_lines(events_path) if args.mode == "tail" else _replay_lines(events_path, speed=float(args.speed))
            for line in src:
                try:
                    _ = json.loads(line)
                except Exception:
                    continue
                await _broadcast(line)
        except KeyboardInterrupt:
            pass
        finally:
            ws_server.close()
            await ws_server.wait_closed()

    try:
        asyncio.run(_run())
    except RuntimeError:
        loop = asyncio.get_event_loop()
        loop.run_until_complete(_run())


if __name__ == "__main__":
    main()

