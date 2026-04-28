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
    """Validates talk and figure API connectivity."""

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

    # ── talk provider tests ──────────────────────────────────────

    def test_talk_config(self):
        self._header("Test 1: Talk Provider Config")
        checks = [
            ("talk_key", api_config.talk_key[:20] + "..." if len(api_config.talk_key) > 20 else api_config.talk_key),
            ("talk_url", api_config.talk_url),
            ("talk model (talk)", api_config.talk_model("talk")),
            ("talk model (vl)", api_config.talk_model("vl")),
            ("talk model (experiment)", api_config.talk_model("experiment")),
        ]
        for name, val in checks:
            self._result(name, bool(val), str(val))

    def test_talk_connection(self):
        self._header("Test 2: Talk API Connection")
        if not api_config.talk_key:
            self._result("Talk connection", False, "No talk API key")
            return
        try:
            resp = requests.post(
                api_config.talk_url,
                headers={"Authorization": f"Bearer {api_config.talk_key}",
                         "Content-Type": "application/json"},
                json={"model": api_config.talk_model("talk"),
                      "messages": [{"role": "user", "content": "Hi"}],
                      "max_tokens": 10},
                timeout=15,
            )
            ok = resp.status_code == 200
            self._result("Talk connection", ok,
                         f"HTTP {resp.status_code}" if ok else f"HTTP {resp.status_code}: {resp.text[:120]}")
        except Exception as e:
            self._result("Talk connection", False, str(e)[:120])

    def test_talk_chat(self):
        self._header("Test 3: Talk Chat")
        if not api_config.talk_key:
            self._result("Talk chat", False, "No talk API key")
            return
        try:
            resp = requests.post(
                api_config.talk_url,
                headers={"Authorization": f"Bearer {api_config.talk_key}",
                         "Content-Type": "application/json"},
                json={"model": api_config.talk_model("talk"),
                      "messages": [{"role": "user", "content": "Say hello in one sentence."}],
                      "max_tokens": 50},
                timeout=30,
            )
            if resp.status_code == 200:
                text = resp.json()["choices"][0]["message"]["content"]
                self._result("Talk chat", True, text[:120])
            else:
                self._result("Talk chat", False, f"HTTP {resp.status_code}: {resp.text[:120]}")
        except Exception as e:
            self._result("Talk chat", False, str(e)[:120])

    # ── figure provider tests ────────────────────────────────────

    def test_figure_config(self):
        self._header("Test 4: Figure Provider Config")
        checks = [
            ("figure_key", api_config.figure_key[:20] + "..." if len(api_config.figure_key) > 20 else "(empty)"),
            ("figure_url", api_config.figure_url),
            ("figure model (image)", api_config.figure_model("image")),
        ]
        for name, val in checks:
            self._result(name, bool(val), str(val))

    def test_figure_connection(self):
        self._header("Test 5: Figure API Connection")
        if not api_config.figure_key:
            self._result("Figure connection", False, "No figure API key")
            return
        try:
            resp = requests.post(
                api_config.figure_url,
                headers={"Authorization": f"Bearer {api_config.figure_key}",
                         "Content-Type": "application/json"},
                json={"model": api_config.figure_model("image"),
                      "prompt": "test",
                      "size": "256x256",
                      "response_format": "b64_json",
                      "n": 1},
                timeout=120,
            )
            ok = resp.status_code == 200
            self._result("Figure connection", ok,
                         f"HTTP {resp.status_code}" if ok else f"HTTP {resp.status_code}: {resp.text[:120]}")
        except Exception as e:
            self._result("Figure connection", False, str(e)[:120])

    # ── summary ──────────────────────────────────────────────────

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
            print("  All tests passed.")
        else:
            print("  Some tests failed — check api_config.json")
        print("=" * 60 + "\n")
        return failed == 0

    def run_all(self):
        print(f"\nAPI Config Test")
        print(f"  talk_provider:   {api_config.talk_url}")
        print(f"  figure_provider: {api_config.figure_url}")

        self.test_talk_config(); time.sleep(0.3)
        self.test_talk_connection(); time.sleep(0.3)
        self.test_talk_chat(); time.sleep(0.3)
        self.test_figure_config(); time.sleep(0.3)
        self.test_figure_connection()

        return self.print_summary()


if __name__ == "__main__":
    tester = APITester()
    ok = tester.run_all()
    sys.exit(0 if ok else 1)
