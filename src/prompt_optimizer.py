"""Prompt optimizer — call LLM to expand user input into a richer image prompt.

Reuses api_config.json for talk-model credentials.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a professional image prompt optimizer. "
    "Expand the user's simple description into a detailed, English prompt "
    "suitable for image generation models (Stable Diffusion, DALL-E, etc.). "
    "Preserve the original meaning. Add artistic style, lighting, composition, "
    "camera angle, and color palette details. "
    "Output ONLY the optimized prompt text — no explanations, no markdown, no JSON."
)

_MAX_RETRIES = 2
_TIMEOUT = 60


def optimize_prompt(
    raw_prompt: str,
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
    model: Optional[str] = None,
    system_prompt: Optional[str] = None,
) -> str:
    """Call the talk LLM to optimize a raw user prompt.

    Args:
        raw_prompt: The user's original input (Chinese or English).
        api_key: Override API key (default: from api_config).
        api_url: Override API URL (default: from api_config).
        model: Override model name (default: api_config.model("talk")).
        system_prompt: Override system prompt (default: built-in).

    Returns:
        Optimized English prompt. Falls back to raw_prompt on any error.
    """
    from src.config import api_config

    key = api_key or api_config.key
    url = api_url or api_config.url
    model_name = model or api_config.model("talk") or "gpt-4o"
    sys_prompt = system_prompt or _SYSTEM_PROMPT

    if not key:
        logger.warning("No API key configured — skipping prompt optimization")
        return raw_prompt

    if not url:
        logger.warning("No API URL configured — skipping prompt optimization")
        return raw_prompt

    logger.info("Calling talk model (%s) to optimize prompt...", model_name)

    import requests

    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": f"Optimize this prompt for image generation:\n{raw_prompt}"},
    ]

    for attempt in range(1 + _MAX_RETRIES):
        try:
            resp = requests.post(
                url,
                headers={
                    "Authorization": f"Bearer {key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model_name,
                    "messages": messages,
                    "max_tokens": 256,
                    "temperature": 0.7,
                },
                timeout=_TIMEOUT,
            )

            if resp.status_code == 200:
                data = resp.json()
                optimized = data["choices"][0]["message"]["content"].strip()
                logger.info("Prompt optimized: %s -> %s", raw_prompt[:60], optimized[:120])
                return optimized

            # Retryable error
            if resp.status_code >= 500:
                logger.warning("API server error %d (attempt %d)", resp.status_code, attempt + 1)
                continue

            # Non-retryable error
            logger.warning("API returned %d: %s", resp.status_code, resp.text[:200])
            return raw_prompt

        except requests.Timeout:
            logger.warning("Request timeout (attempt %d/%d)", attempt + 1, 1 + _MAX_RETRIES)
        except requests.ConnectionError as e:
            logger.warning("Connection error: %s", e)
            return raw_prompt
        except Exception as e:
            logger.warning("LLM call failed: %s", e)
            return raw_prompt

    logger.warning("All retries exhausted — returning raw prompt")
    return raw_prompt
