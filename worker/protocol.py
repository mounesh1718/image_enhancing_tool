"""JSON-lines protocol between the C# app and this worker.

Every message is one UTF-8 JSON object on one line.

  worker -> app at startup:  {"type": "ready", "protocol": 1, "pid": 1234}
  app -> worker request:     {"id": "7", "cmd": "ping", "args": {...}}
  worker -> app result:      {"id": "7", "type": "result", "ok": true, "result": {...}}
  worker -> app failure:     {"id": "7", "type": "result", "ok": false,
                              "error": {"code": "unknown_command", "message": "..."}}
  worker -> app progress:    {"id": "7", "type": "progress", "data": {...}}

Image data never travels over the pipe; requests and results carry file paths only.
stdout is reserved for protocol messages; diagnostics go to stderr.
"""

from __future__ import annotations

import json
import os
import sys
import traceback
from typing import Any, Callable, TextIO

PROTOCOL_VERSION = 1

Handler = Callable[[dict[str, Any], "Context"], Any]

_COMMANDS: dict[str, Handler] = {}


class WorkerError(Exception):
    """An expected failure reported to the app with a stable error code."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


class ShutdownRequested(Exception):
    pass


class Context:
    """Per-request helpers passed to command handlers."""

    def __init__(self, request_id: str | None, out: TextIO):
        self.request_id = request_id
        self._out = out

    def progress(self, **data: Any) -> None:
        write_message(self._out, {"id": self.request_id, "type": "progress", "data": data})


def command(name: str) -> Callable[[Handler], Handler]:
    def register(fn: Handler) -> Handler:
        if name in _COMMANDS:
            raise ValueError(f"duplicate command: {name}")
        _COMMANDS[name] = fn
        return fn

    return register


def write_message(out: TextIO, message: dict[str, Any]) -> None:
    out.write(json.dumps(message, separators=(",", ":"), ensure_ascii=False))
    out.write("\n")
    out.flush()


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _error(request_id: str | None, code: str, message: str) -> dict[str, Any]:
    return {"id": request_id, "type": "result", "ok": False, "error": {"code": code, "message": message}}


def handle_line(line: str, out: TextIO) -> None:
    """Parse one request line, run its command and write exactly one result message.

    Raises ShutdownRequested after replying to a shutdown command.
    """
    request_id = None
    try:
        try:
            request = json.loads(line)
        except json.JSONDecodeError as exc:
            raise WorkerError("bad_request", f"invalid JSON: {exc.msg}") from None
        if not isinstance(request, dict):
            raise WorkerError("bad_request", "request must be a JSON object")

        request_id = request.get("id")
        if request_id is not None:
            request_id = str(request_id)
        name = request.get("cmd")
        args = request.get("args") or {}
        if not isinstance(name, str):
            raise WorkerError("bad_request", "missing 'cmd'")
        if not isinstance(args, dict):
            raise WorkerError("bad_request", "'args' must be an object")

        handler = _COMMANDS.get(name)
        if handler is None:
            raise WorkerError("unknown_command", f"unknown command: {name}")

        result = handler(args, Context(request_id, out))
        write_message(out, {"id": request_id, "type": "result", "ok": True, "result": result})
    except ShutdownRequested:
        write_message(out, {"id": request_id, "type": "result", "ok": True, "result": None})
        raise
    except WorkerError as exc:
        write_message(out, _error(request_id, exc.code, exc.message))
    except MemoryError:
        write_message(out, _error(request_id, "out_of_memory", "not enough memory for this operation"))
    except Exception as exc:  # noqa: BLE001 - one bad request must not kill the worker
        log(traceback.format_exc())
        write_message(out, _error(request_id, "internal_error", f"{type(exc).__name__}: {exc}"))


def serve(stdin: TextIO | None = None, stdout: TextIO | None = None) -> int:
    stdin = stdin or sys.stdin
    stdout = stdout or sys.stdout
    write_message(stdout, {"type": "ready", "protocol": PROTOCOL_VERSION, "pid": os.getpid()})
    for line in stdin:
        line = line.strip()
        if not line:
            continue
        try:
            handle_line(line, stdout)
        except ShutdownRequested:
            return 0
    return 0  # stdin closed: the app went away
