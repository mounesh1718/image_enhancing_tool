import io
import json
import subprocess
import sys
import unittest
from pathlib import Path

WORKER_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKER_DIR))

import main  # noqa: E402,F401 - registers the commands
from protocol import ShutdownRequested, handle_line  # noqa: E402


def run(line: str) -> dict:
    out = io.StringIO()
    handle_line(line, out)
    lines = out.getvalue().splitlines()
    assert len(lines) == 1, lines
    return json.loads(lines[0])


class HandleLineTests(unittest.TestCase):
    def test_ping_returns_result_with_same_id(self):
        msg = run('{"id": "1", "cmd": "ping", "args": {"echo": "hé"}}')
        self.assertEqual(msg["id"], "1")
        self.assertTrue(msg["ok"])
        self.assertEqual(msg["result"]["echo"], "hé")

    def test_invalid_json_is_bad_request(self):
        msg = run("{not json")
        self.assertFalse(msg["ok"])
        self.assertEqual(msg["error"]["code"], "bad_request")

    def test_unknown_command(self):
        msg = run('{"id": 5, "cmd": "nope"}')
        self.assertEqual(msg["id"], "5")
        self.assertEqual(msg["error"]["code"], "unknown_command")

    def test_args_must_be_object(self):
        msg = run('{"id": "2", "cmd": "ping", "args": [1]}')
        self.assertEqual(msg["error"]["code"], "bad_request")

    def test_info_reports_versions(self):
        result = run('{"id": "3", "cmd": "info"}')["result"]
        self.assertIn("python", result)
        self.assertIn("opencv", result)

    def test_shutdown_replies_then_raises(self):
        out = io.StringIO()
        with self.assertRaises(ShutdownRequested):
            handle_line('{"id": "9", "cmd": "shutdown"}', out)
        self.assertTrue(json.loads(out.getvalue())["ok"])


class ProcessTests(unittest.TestCase):
    def test_full_session_over_pipes(self):
        requests = "\n".join([
            '{"id": "1", "cmd": "ping"}',
            "garbage",
            '{"id": "2", "cmd": "shutdown"}',
            '{"id": "3", "cmd": "ping"}',  # must not be processed after shutdown
        ]) + "\n"
        proc = subprocess.run(
            [sys.executable, str(WORKER_DIR / "main.py")],
            input=requests.encode("utf-8"),
            capture_output=True,
            timeout=30,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr.decode(errors="replace"))
        messages = [json.loads(l) for l in proc.stdout.decode("utf-8").splitlines()]
        self.assertEqual(messages[0]["type"], "ready")
        self.assertEqual([m.get("id") for m in messages[1:]], ["1", None, "2"])
        self.assertFalse(messages[2]["ok"])


if __name__ == "__main__":
    unittest.main()
