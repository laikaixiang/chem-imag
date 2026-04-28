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
    """Load api_config.json with named talk and figure providers.

    Structure:
        {
            "talk_provider": {
                "siliconflow": { api_key, api_url, models: {talk, vl, experiment} }
            },
            "active_talk_provider": "siliconflow",
            "figure_provider": {
                "packyapi": { api_key, api_url, api_type, models: {image} }
            },
            "active_figure_provider": "packyapi"
        }

    Usage:
        api = ApiConfig()

        # Talk (LLM) provider — compound extraction, prompt optimization
        key   = api.talk_key           # active talk provider's api_key
        url   = api.talk_url           # active talk provider's api_url
        model = api.talk_model("talk") # active talk provider's model

        # Figure (image gen) provider — scene generation
        key   = api.figure_key
        url   = api.figure_url
        model = api.figure_model("image")

        # Provider lists for UI dropdowns
        api.available_talk_providers    → ["siliconflow"]
        api.available_figure_providers  → ["packyapi", "mock"]

        # Backward-compat shortcuts → active talk provider
        api.key / api.url / api.model("talk")
    """

    def __init__(self, path: Path | None = None):
        path = path or ROOT / "api_config.json"
        try:
            with open(path, encoding="utf-8") as f:
                self._data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            self._data = {}

    # ── helpers ──────────────────────────────────────────────────

    def _resolve_active(self, category: str) -> dict:
        """Return the active provider dict for a category (talk / figure)."""
        providers = self._data.get(category, {})
        if not providers:
            return {}
        active_key = f"active_{category.rsplit('_', 1)[0]}_provider"
        active_name = self._data.get(active_key, "")
        if active_name and active_name in providers:
            entry = providers[active_name]
            if isinstance(entry, dict):
                return entry
        # No active set — pick the first named provider
        for name, entry in providers.items():
            if isinstance(entry, dict) and "api_key" in entry:
                return entry
        return {}

    def _list_providers(self, category: str) -> list[str]:
        """List all named providers under a category."""
        providers = self._data.get(category, {})
        return [k for k, v in providers.items() if isinstance(v, dict) and "api_key" in v]

    # ── talk provider ────────────────────────────────────────────

    @property
    def talk_key(self) -> str:
        return self._resolve_active("talk_provider").get("api_key", "")

    @property
    def talk_url(self) -> str:
        return self._resolve_active("talk_provider").get("api_url", "")

    @property
    def talk_base_url(self) -> str:
        u = self.talk_url
        return u.rsplit("/chat/completions", 1)[0] if "/chat/completions" in u else u

    def talk_model(self, purpose: str = "talk") -> str:
        return self._resolve_active("talk_provider").get("models", {}).get(purpose, "")

    @property
    def available_talk_providers(self) -> list[str]:
        return self._list_providers("talk_provider")

    # ── figure provider ──────────────────────────────────────────

    @property
    def figure_key(self) -> str:
        return self._resolve_active("figure_provider").get("api_key", "")

    @property
    def figure_url(self) -> str:
        return self._resolve_active("figure_provider").get("api_url", "")

    def figure_model(self, purpose: str = "image") -> str:
        return self._resolve_active("figure_provider").get("models", {}).get(purpose, "")

    @property
    def figure_provider_type(self) -> str:
        return self._resolve_active("figure_provider").get("api_type", "mock")

    @property
    def available_figure_providers(self) -> list[str]:
        """Return available figure providers. Always includes mock as fallback."""
        providers = self._list_providers("figure_provider")
        return providers + ["mock"] if providers else ["mock"]

    # ── backward-compat (→ active talk provider) ─────────────────

    @property
    def key(self) -> str:
        return self.talk_key

    @property
    def url(self) -> str:
        return self.talk_url

    @property
    def base_url(self) -> str:
        return self.talk_base_url

    def model(self, purpose: str = "talk") -> str:
        return self.talk_model(purpose)

    @property
    def has_key(self) -> bool:
        return bool(self.talk_key)

    @property
    def active_provider(self) -> str:
        """Return "anthropic" or "openai_compatible" based on talk provider config."""
        url = self.talk_url.lower()
        if "anthropic" in url:
            return "anthropic"
        return "openai_compatible"


api_config = ApiConfig()

