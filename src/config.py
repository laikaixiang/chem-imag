"""全局配置 — Settings + ApiConfig"""

import json
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent


class Settings:
    OUTPUT_DIR: Path = Path(os.getenv("OUTPUT_DIR", ROOT / "outputs"))
    TEMP_DIR: Path = Path(os.getenv("TEMP_DIR", ROOT / "outputs" / "temp"))

    AI_IMAGE_PROVIDER: str = os.getenv("AI_IMAGE_PROVIDER", "sd_webui")
    SD_WEBUI_URL: str = os.getenv("SD_WEBUI_URL", "http://127.0.0.1:7860")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    REPLICATE_API_TOKEN: str = os.getenv("REPLICATE_API_TOKEN", "")

    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "claude")
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")

    DEVICE: str = os.getenv("DEVICE", "cpu")

    def ensure_dirs(self):
        for d in [self.OUTPUT_DIR, self.TEMP_DIR,
                  self.OUTPUT_DIR / "structures",
                  self.OUTPUT_DIR / "scenes",
                  self.OUTPUT_DIR / "mindmaps",
                  self.OUTPUT_DIR / "final"]:
            d.mkdir(parents=True, exist_ok=True)
        return self

    def get_device(self) -> str:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"


settings = Settings().ensure_dirs()


# ── API config (api_config.json) ─────────────────────────────────

class ApiConfig:
    """Load api_config.json and expose per-provider keys/URLs/models.

    Usage:
        api = ApiConfig()
        key = api.key                  # active provider's api_key
        url = api.url                  # full endpoint with /chat/completions
        base = api.base_url            # stripped of /chat/completions
        model = api.model("talk")      # model name by purpose
    """

    def __init__(self, path: Path | None = None):
        path = path or ROOT / "api_config.json"
        try:
            with open(path, encoding="utf-8") as f:
                self._data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            self._data = {"default_provider": "", "providers": {}}
        self._active = self._data.get("default_provider", "")

    # ── active provider helpers ─────────────────────────────────

    @property
    def active_provider(self) -> str:
        return self._active

    @property
    def provider(self) -> dict:
        """Return the active provider's full config dict."""
        return self._data.get("providers", {}).get(self._active, {})

    @property
    def key(self) -> str:
        """API key for the active provider."""
        return self.provider.get("api_key", "")

    @property
    def url(self) -> str:
        """Full API endpoint (includes /chat/completions if applicable)."""
        return self.provider.get("api_url", "")

    @property
    def base_url(self) -> str:
        """Base URL — strips /chat/completions suffix for SDKs like PydanticAI."""
        u = self.url
        return u.rsplit("/chat/completions", 1)[0] if "/chat/completions" in u else u

    def model(self, purpose: str = "talk") -> str:
        """Get model name by purpose (talk, vl, experiment, claude, haiku, gpt4o)."""
        return self.provider.get("models", {}).get(purpose, "")

    # ── cross-provider access ───────────────────────────────────

    def provider_config(self, name: str) -> dict:
        return self._data.get("providers", {}).get(name, {})

    def switch_provider(self, name: str):
        """Temporarily switch to a different provider."""
        if name in self._data.get("providers", {}):
            self._active = name

    @property
    def has_key(self) -> bool:
        return bool(self.key)


api_config = ApiConfig()

