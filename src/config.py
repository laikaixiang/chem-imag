"""全局配置"""
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
