"""
API configuration test tool — validates api_config.json connectivity.

Usage:
    python tests/api_test.py
"""

import json
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import api_config


class APITester:
    """Validates API connectivity for configured providers."""

    def __init__(self):
        self.results: list[tuple[str, bool, str]] = []

    def _header(self, title: str):
        print("\n" + "=" * 60)
        print(f"  {title}")
        print("=" * 60)

    def _result(self, name: str, ok: bool, msg: str = ""):
        tag = "[PASS]" if ok else "[FAIL]"
        print(f"{tag} {name}")
        if msg:
            print(f"      {msg}")
        self.results.append((name, ok, msg))

    # ── tests ───────────────────────────────────────────────────

    def test_config_loaded(self):
        self._header("Test 1: Config Loaded")
        checks = [
            ("default_provider", api_config.active_provider),
            ("api_key", api_config.key[:20] + "..." if len(api_config.key) > 20 else api_config.key),
            ("api_url", api_config.url),
            ("talk model", api_config.model("talk")),
            ("vl model", api_config.model("vl")),
            ("experiment model", api_config.model("experiment")),
        ]
        all_ok = True
        for name, val in checks:
            ok = bool(val)
            self._result(name, ok, val)
            if not ok:
                all_ok = False
        return all_ok

    def test_api_connection(self):
        self._header("Test 2: API Connection")
        if not api_config.key:
            self._result("API connection", False, "No API key configured")
            return False
        if not api_config.url:
            self._result("API connection", False, "No API URL configured")
            return False

        try:
            url = api_config.url
            resp = requests.post(
                url,
                headers={
                    "Authorization": f"Bearer {api_config.key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": api_config.model("talk"),
                    "messages": [{"role": "user", "content": "Hi"}],
                    "max_tokens": 10,
                },
                timeout=15,
            )
            if resp.status_code == 200:
                self._result("API connection", True, f"HTTP {resp.status_code}")
                return True
            else:
                self._result("API connection", False,
                             f"HTTP {resp.status_code}: {resp.text[:150]}")
                return False
        except requests.Timeout:
            self._result("API connection", False, "Timeout (>15s)")
            return False
        except requests.ConnectionError as e:
            self._result("API connection", False, str(e)[:120])
            return False

    def test_talk_model(self):
        self._header("Test 3: Talk Model (Chat)")
        if not api_config.key:
            self._result("Talk model", False, "No API key — skip")
            return False

        try:
            resp = requests.post(
                api_config.url,
                headers={
                    "Authorization": f"Bearer {api_config.key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": api_config.model("talk"),
                    "messages": [{"role": "user", "content": "Say hello in one sentence."}],
                    "max_tokens": 50,
                },
                timeout=30,
            )
            if resp.status_code == 200:
                data = resp.json()
                text = data["choices"][0]["message"]["content"]
                self._result("Talk model", True, text[:120])
                return True
            else:
                self._result("Talk model", False, f"HTTP {resp.status_code}: {resp.text[:150]}")
                return False
        except Exception as e:
            self._result("Talk model", False, str(e)[:120])
            return False

    def test_streaming(self):
        self._header("Test 4: Streaming Response")
        if not api_config.key:
            self._result("Streaming", False, "No API key — skip")
            return False

        try:
            resp = requests.post(
                api_config.url,
                headers={
                    "Authorization": f"Bearer {api_config.key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": api_config.model("talk"),
                    "messages": [{"role": "user", "content": "Count to 3."}],
                    "stream": True,
                    "max_tokens": 30,
                },
                stream=True,
                timeout=30,
            )
            if resp.status_code != 200:
                self._result("Streaming", False, f"HTTP {resp.status_code}")
                return False

            chunks = 0
            content = ""
            for line in resp.iter_lines():
                if not line:
                    continue
                s = line.decode("utf-8")
                if s.startswith("data: "):
                    s = s[6:]
                if s == "[DONE]":
                    break
                try:
                    delta = json.loads(s)["choices"][0].get("delta", {}).get("content", "")
                    if delta:
                        content += delta
                        chunks += 1
                except (json.JSONDecodeError, KeyError):
                    pass

            self._result("Streaming", chunks > 0,
                         f"{chunks} chunks: {content[:100]}")
            return chunks > 0
        except Exception as e:
            self._result("Streaming", False, str(e)[:120])
            return False

    def test_prompt_optimizer_e2e(self):
        """End-to-end: optimize a real prompt via the Pipeline."""
        self._header("Test 5: Prompt Optimizer E2E")
        from src.prompt_optimizer import optimize_prompt

        raw = "a cat sitting on a table"
        opt = optimize_prompt(raw)

        self._result("Optimizer E2E", isinstance(opt, str) and len(opt) > 0,
                     f"'{raw[:60]}' -> '{opt[:120]}'")

    # ── summary ─────────────────────────────────────────────────

    def print_summary(self):
        self._header("Summary")
        total = len(self.results)
        passed = sum(1 for _, ok, _ in self.results if ok)
        failed = total - passed

        print(f"\nTotal: {total}  |  Passed: {passed}  |  Failed: {failed}")

        if failed:
            print("\nFailed tests:")
            for name, ok, msg in self.results:
                if not ok:
                    print(f"  - {name}: {msg}")

        print("\n" + "=" * 60)
        if failed == 0:
            print("  All tests passed — API config is valid.")
        else:
            print("  Some tests failed — check api_config.json")
        print("=" * 60 + "\n")
        return failed == 0

    def run_all(self):
        print(f"\nAPI Config Test — provider: {api_config.active_provider}")
        print(f"URL: {api_config.url}")
        print(f"Models: talk={api_config.model('talk')}, "
              f"vl={api_config.model('vl')}, "
              f"experiment={api_config.model('experiment')}")

        self.test_config_loaded(); time.sleep(0.3)
        self.test_api_connection(); time.sleep(0.3)
        self.test_talk_model(); time.sleep(0.3)
        self.test_streaming(); time.sleep(0.3)
        self.test_prompt_optimizer_e2e()

        return self.print_summary()


if __name__ == "__main__":
    tester = APITester()
    ok = tester.run_all()
    sys.exit(0 if ok else 1)
