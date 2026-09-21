"""Exercise monitoring outcomes without a phone, installation, or real sleeps."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "mcp-deploy-verify.sh"


class DeployVerificationTests(unittest.TestCase):
    def run_watch(self, states):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mock = root / "xcrun"
            mock.write_text("""#!/usr/bin/env python3
import os, pathlib, sys
if 'processes' not in sys.argv:
    sys.exit(0)
counter = pathlib.Path(os.environ['PROBE_COUNTER'])
n = int(counter.read_text()) if counter.exists() else 0
counter.write_text(str(n + 1))
states = os.environ['PROBE_STATES'].split(',')
state = states[min(n, len(states) - 1)]
if state == 'unavailable':
    print('Device connection interrupted', file=sys.stderr)
    sys.exit(1)
print('PID Executable Path')
if state == 'alive':
    print('1193 /private/var/containers/MCPZimChat.app/MCPZimChat')
""")
            mock.chmod(0o755)
            sleep = root / "sleep"
            sleep.write_text("#!/bin/sh\nexit 0\n")
            sleep.chmod(0o755)
            env = dict(os.environ, PATH=f"{root}:{os.environ['PATH']}",
                       MCPZIM_WATCH_SECS="10", PROBE_COUNTER=str(root / "count"),
                       PROBE_STATES=','.join(states))
            return subprocess.run(["bash", str(SCRIPT), "watch"], env=env,
                                  capture_output=True, text=True, timeout=10)

    def test_alive(self):
        result = self.run_watch(["alive", "alive"])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("RESULT: OK", result.stdout)

    def test_transient_connection_failure_is_not_an_app_exit(self):
        result = self.run_watch(["unavailable", "alive"])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("app state unknown", result.stdout)
        self.assertNotIn("app died", result.stdout)

    def test_unavailable_final_query_is_inconclusive(self):
        result = self.run_watch(["alive", "unavailable"])
        self.assertEqual(result.returncode, 3, result.stderr)
        self.assertIn("RESULT: INCONCLUSIVE", result.stdout)
        self.assertNotIn("RESULT: OK", result.stdout)

    def test_successful_query_without_app_is_a_failure(self):
        result = self.run_watch(["missing"])
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn("app NOT RUNNING", result.stdout)


if __name__ == "__main__":
    unittest.main()
