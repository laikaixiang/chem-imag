"""AI scene generation interface with SD WebUI + ControlNet support.

Provides text-to-image generation, ControlNet-conditioned generation,
and mindmap skeleton enhancement for chem-mindmap.
"""

import base64
import logging
from io import BytesIO
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image

from src.config import api_config, settings

logger = logging.getLogger(__name__)

ENHANCE_PROMPTS = {
    "academic": (
        "clean academic mind map for organic chemistry journal paper, "
        "white background with subtle gradient, professional diagram style, "
        "clear hierarchical layout showing chemical compounds and reactions, "
        "sharp text, high resolution, suitable for publication, "
        "molecular structures are clearly visible and scientifically accurate"
    ),
    "modern": (
        "modern scientific infographic style mind map, "
        "vibrant but professional colors, gradient background, "
        "organic chemistry concepts organized hierarchically, "
        "3D effects on molecular structures, "
        "clean typography, presentation quality"
    ),
    "minimal": (
        "minimalist black and white scientific diagram, "
        "clean lines, no colors except structural formulas, "
        "chemistry mind map for academic paper, "
        "high contrast, sharp edges, publication ready"
    ),
}

GUIDANCE_MODIFIERS = {
    "canny": "straight-on angle, front-facing view, flat layout with no perspective distortion",
    "depth": "gentle depth of field, subject in focus, clean separation from background",
    "scribble": "loose sketch style converted to clean vector-like appearance, crisp outlines",
}


class SceneGenerator:
    """AI scene generator with ControlNet conditioning.

    Supports multiple backends: SD WebUI, OpenAI, Replicate, and a local mock mode.

    Usage:
        gen = SceneGenerator(provider="sd_webui")
        img = gen.generate("a chemistry lab bench, clean white background")
        enhanced = gen.enhance_mindmap(mindmap_bgr, style_prompt="academic")

    TODO: 为 SceneGenerator 添加本地 GPU 支持
      1. _estimate_depth() 替换 Sobel placeholder，集成 MiDaS 深度估计模型，
         通过 settings.DEVICE 自动选择 cuda/cpu
      2. _extract_canny_edges() 添加 OpenCV CUDA 路径
         (cv2.cuda.createCannyEdgeDetector)，在 CUDA 可用时走 GPU
      3. 构造函数增加 use_gpu 参数，允许显式启用/禁用本地 GPU
      4. 补充 GPU 路径的测试（mock 模式下跳过实际推理，仅验证设备选择逻辑）
    """

    def __init__(self, provider: str = "sd_webui", config: Optional[dict] = None):
        self.provider = provider
        self._cfg = config or {}
        self._sd_url = self._cfg.get("sd_webui_url", settings.SD_WEBUI_URL)

    # ── public API ──────────────────────────────────────────────

    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        width: int = 1024,
        height: int = 768,
        guidance_scale: float = 7.5,
        num_steps: int = 30,
        control_image: Optional[np.ndarray] = None,
        control_type: str = "canny",
    ) -> Image.Image:
        """Generate an AI image, optionally conditioned on a control image.

        Args:
            prompt: Text description of the desired image.
            negative_prompt: Things to avoid in the output.
            width, height: Output dimensions in pixels.
            guidance_scale: CFG scale (higher = tighter prompt adherence).
            num_steps: Denoising steps.
            control_image: BGR or grayscale image for ControlNet conditioning.
            control_type: One of "canny", "depth", "scribble".

        Returns:
            PIL Image in RGB mode.
        """
        prompt = self._build_guided_prompt(prompt, control_type if control_image is not None else None)

        if self.provider == "sd_webui":
            if control_image is not None:
                return self._generate_with_controlnet(
                    prompt, negative_prompt, control_image, control_type,
                    width, height, guidance_scale, num_steps,
                )
            return self._generate_sd_webui(
                prompt, negative_prompt, width, height, guidance_scale, num_steps,
            )
        elif self.provider == "packyapi":
            return self._generate_openai_image(
                prompt, width, height,
            )
        elif self.provider == "mock":
            return self._generate_mock(width, height)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def enhance_mindmap(
        self,
        mindmap_image: np.ndarray,
        style_prompt: str = "academic",
        width: int = 1024,
        height: int = 768,
    ) -> Image.Image:
        """Enhance a mindmap skeleton image into a polished scene.

        Pipeline:
        1. Build an enhancement prompt from the configured style.
        2. Extract Canny edges from the skeleton.
        3. Generate via ControlNet conditioned on those edges.

        Args:
            mindmap_image: BGR or grayscale mindmap skeleton image.
            style_prompt: Key into ENHANCE_PROMPTS ("academic", "modern", "minimal")
                          or a free-form style string.
            width, height: Output dimensions.

        Returns:
            PIL Image in RGB mode.
        """
        base = ENHANCE_PROMPTS.get(style_prompt, style_prompt)
        prompt = self._build_guided_prompt(base, "canny")

        negative = (
            "low quality, blurry, distorted text, messy layout, "
            "overlapping elements, unreadable labels, cluttered composition"
        )

        edges = self._extract_canny_edges(mindmap_image)

        if self.provider == "mock":
            return self._generate_mock(width, height)

        return self._generate_with_controlnet(
            prompt, negative, edges, "canny",
            width, height, guidance_scale=9.0, num_steps=30,
        )

    def generate_style_prompt(self, mindmap_json: dict, style: str = "academic") -> str:
        """Build a prompt from mindmap content metadata and a named style.

        Args:
            mindmap_json: Dict with keys like 'title', 'topics', 'compounds'.
            style: Key into ENHANCE_PROMPTS.

        Returns:
            A composed prompt string.
        """
        base = ENHANCE_PROMPTS.get(style, ENHANCE_PROMPTS["academic"])
        title = mindmap_json.get("title", "")
        topics = mindmap_json.get("topics", [])
        topic_str = ", ".join(topics[:5]) if topics else "organic chemistry"

        if title:
            return f"{base}, mind map about {title}, covering {topic_str}"
        return f"{base}, mind map covering {topic_str}"

    # ── prompt helpers ──────────────────────────────────────────

    def _build_guided_prompt(self, base_prompt: str, control_type: Optional[str]) -> str:
        """Append viewpoint / quality guidance to the prompt.

        Adds modifiers like "straight-on angle" for Canny-conditioned
        generation to ensure the layout stays flat and readable.
        """
        guidance = ""
        if control_type and control_type in GUIDANCE_MODIFIERS:
            guidance = GUIDANCE_MODIFIERS[control_type]
        else:
            guidance = "high quality, sharp details, professional scientific illustration"

        extra = self._cfg.get("prompt_suffix", "")
        parts = [base_prompt, guidance, extra]
        return ", ".join(p for p in parts if p)

    # ── SD WebUI backend ────────────────────────────────────────

    def _generate_sd_webui(
        self,
        prompt: str,
        negative_prompt: str,
        width: int,
        height: int,
        cfg_scale: float,
        steps: int,
    ) -> Image.Image:
        """Call SD WebUI txt2img API."""
        import requests

        payload = {
            "prompt": prompt,
            "negative_prompt": negative_prompt or "",
            "width": width,
            "height": height,
            "steps": steps,
            "cfg_scale": cfg_scale,
            "sampler_name": "DPM++ 2M Karras",
        }

        url = f"{self._sd_url}/sdapi/v1/txt2img"
        logger.info("SD WebUI txt2img → %s", url)

        try:
            resp = requests.post(url, json=payload, timeout=300)
            resp.raise_for_status()
        except requests.ConnectionError:
            logger.warning("SD WebUI not reachable at %s, falling back to mock", self._sd_url)
            return self._generate_mock(width, height)

        data = resp.json()
        img_data = base64.b64decode(data["images"][0])
        return Image.open(BytesIO(img_data)).convert("RGB")

    def _generate_openai_image(
        self,
        prompt: str,
        width: int = 1024,
        height: int = 768,
    ) -> Image.Image:
        """Call an OpenAI-compatible image generation API (packyapi, DALL-E, etc.).

        Endpoint: POST /v1/images/generations
        Response format: {"data": [{"b64_json": "..."}]} or {"data": [{"url": "..."}]}
        """
        import requests

        key = api_config.figure_key
        url = api_config.figure_url or self._cfg.get("image_api_url", "")
        model = api_config.figure_model("image") or "gpt-image-2"

        if not key:
            logger.warning("No figure API key configured — falling back to mock")
            return self._generate_mock(width, height)

        payload = {
            "model": model,
            "prompt": prompt,
            "size": f"{width}x{height}",
            "quality": "high",
            "output_format": "png",
            "response_format": "b64_json",
            "n": 1,
        }

        logger.info("OpenAI image gen → %s (model=%s)", url, model)

        try:
            resp = requests.post(
                url,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {key}",
                },
                json=payload,
                timeout=300,
            )
            resp.raise_for_status()
        except requests.ConnectionError:
            logger.warning("Image API not reachable at %s, falling back to mock", url)
            return self._generate_mock(width, height)

        data = resp.json()
        if "data" in data and len(data["data"]) > 0:
            entry = data["data"][0]
            if "b64_json" in entry:
                img_data = base64.b64decode(entry["b64_json"])
                img = Image.open(BytesIO(img_data)).convert("RGB")
            elif "url" in entry:
                img_resp = requests.get(entry["url"], timeout=60)
                img_resp.raise_for_status()
                img = Image.open(BytesIO(img_resp.content)).convert("RGB")
            else:
                logger.warning("Unexpected image API response: %s", str(data)[:200])
                return self._generate_mock(width, height)

            # API may return a different size than requested; resize to match
            if (img.width, img.height) != (width, height):
                logger.info("Resizing API output %dx%d → %dx%d", img.width, img.height, width, height)
                img = img.resize((width, height), Image.LANCZOS)
            return img

        logger.warning("Unexpected image API response: %s", str(data)[:200])
        return self._generate_mock(width, height)

    def _generate_with_controlnet(
        self,
        prompt: str,
        negative_prompt: str,
        control_image: np.ndarray,
        control_type: str,
        width: int,
        height: int,
        cfg_scale: float,
        steps: int,
    ) -> Image.Image:
        """Call SD WebUI txt2img API with ControlNet conditioning."""
        import requests

        processed = self._prepare_control_image(control_image, control_type)

        _, buf = cv2.imencode(".png", processed)
        control_b64 = base64.b64encode(buf).decode("utf-8")

        controlnet_models = {
            "canny": "control_v11p_sd15_canny [d14c2d8b]",
            "depth": "control_v11f1p_sd15_depth [cfd03158]",
            "scribble": "control_v11p_sd15_scribble [d4ba51ff]",
        }

        payload = {
            "prompt": prompt,
            "negative_prompt": negative_prompt or "low quality, blurry, distorted text",
            "width": width,
            "height": height,
            "steps": steps,
            "cfg_scale": cfg_scale,
            "sampler_name": "DPM++ 2M Karras",
            "controlnet_units": [
                {
                    "input_image": control_b64,
                    "module": control_type,
                    "model": controlnet_models.get(control_type, controlnet_models["canny"]),
                    "weight": 0.85,
                    "guidance_start": 0.0,
                    "guidance_end": 1.0,
                }
            ],
        }

        url = f"{self._sd_url}/sdapi/v1/txt2img"
        logger.info("SD WebUI ControlNet (%s) → %s", control_type, url)

        try:
            resp = requests.post(url, json=payload, timeout=300)
            resp.raise_for_status()
        except requests.ConnectionError:
            logger.warning("SD WebUI not reachable, falling back to mock")
            return self._generate_mock(width, height)

        data = resp.json()
        img_data = base64.b64decode(data["images"][0])
        return Image.open(BytesIO(img_data)).convert("RGB")

    # ── control image preprocessing ─────────────────────────────

    def _prepare_control_image(self, image: np.ndarray, control_type: str) -> np.ndarray:
        """Convert a raw image into the format expected by a ControlNet module."""
        if control_type == "canny":
            return self._extract_canny_edges(image)
        elif control_type == "depth":
            return self._estimate_depth(image)
        elif control_type == "scribble":
            return self._to_scribble(image)
        else:
            raise ValueError(f"Unknown control_type: {control_type}")

    def _extract_canny_edges(self, image: np.ndarray, low: int = 100, high: int = 200) -> np.ndarray:
        """Extract Canny edges from a BGR or grayscale image."""
        if len(image.shape) == 3 and image.shape[2] == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif len(image.shape) == 3 and image.shape[2] == 4:
            gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        else:
            gray = image
        return cv2.Canny(gray, low, high)

    def _estimate_depth(self, image: np.ndarray) -> np.ndarray:
        """Placeholder depth estimation via Sobel gradient magnitude.

        For production use, replace with MiDaS or another DNN depth estimator.
        """
        if len(image.shape) == 3 and image.shape[2] >= 3:
            gray = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        depth = np.sqrt(grad_x ** 2 + grad_y ** 2)
        depth = np.clip(depth / max(depth.max(), 1) * 255, 0, 255).astype(np.uint8)
        return depth

    def _to_scribble(self, image: np.ndarray) -> np.ndarray:
        """Convert an image into a scribble-like representation via adaptive threshold."""
        if len(image.shape) == 3 and image.shape[2] == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif len(image.shape) == 3 and image.shape[2] == 4:
            gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        else:
            gray = image
        return cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2,
        )

    # ── mock mode ───────────────────────────────────────────────

    def _generate_mock(self, width: int, height: int, color: tuple = (245, 248, 252)) -> Image.Image:
        """Return a solid-color placeholder image for testing."""
        img = Image.new("RGB", (width, height), color)
        logger.info("Mock image generated: %dx%d", width, height)
        return img
