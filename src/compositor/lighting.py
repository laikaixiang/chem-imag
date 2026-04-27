"""Lighting matching and texture blending for chem-mindmap.

Provides color transfer (Reinhard), texture extraction/application,
shadow generation, and a one-stop match_and_blend interface.
"""

import cv2
import numpy as np
from typing import Tuple, Optional


def color_transfer(
    src: np.ndarray,
    target: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Reinhard color transfer in Lab space.

    Transfers the color distribution (mean, std) of target onto src.

    Args:
        src: Source image (HxWx3 RGB)
        target: Target image (HxWx3 RGB)
        mask: Optional binary mask; only masked pixels contribute to stats

    Returns:
        Color-transferred image (HxWx3 RGB, uint8)
    """
    src_lab = cv2.cvtColor(src, cv2.COLOR_RGB2Lab).astype(np.float32)
    target_lab = cv2.cvtColor(target, cv2.COLOR_RGB2Lab).astype(np.float32)

    result = src_lab.copy()

    for c in range(3):
        if mask is not None and mask.sum() > 0:
            src_vals = src_lab[:, :, c][mask > 0]
            tgt_vals = target_lab[:, :, c][mask > 0]
        else:
            src_vals = src_lab[:, :, c].ravel()
            tgt_vals = target_lab[:, :, c].ravel()

        src_mean, src_std = src_vals.mean(), src_vals.std()
        tgt_mean, tgt_std = tgt_vals.mean(), tgt_vals.std()

        if src_std == 0:
            src_std = 1.0
        if tgt_std == 0:
            tgt_std = 1.0

        result[:, :, c] = (src_lab[:, :, c] - src_mean) * (tgt_std / src_std) + tgt_mean

    result_lab = np.clip(result, 0, 255).astype(np.uint8)
    return cv2.cvtColor(result_lab, cv2.COLOR_Lab2RGB)


def extract_texture(image: np.ndarray, sigma: float = 8.0) -> np.ndarray:
    """Extract high-frequency surface texture from image.

    Args:
        image: Input image (HxWx3 RGB, uint8)
        sigma: Gaussian blur sigma

    Returns:
        Texture as float32 array in [-30, 30]
    """
    image_f = image.astype(np.float32)
    blurred = cv2.GaussianBlur(image_f, (0, 0), sigma)
    texture = image_f - blurred
    texture = np.clip(texture, -30, 30)
    return texture


def apply_texture(
    struct: np.ndarray, texture: np.ndarray, strength: float = 0.1
) -> np.ndarray:
    """Overlay surface texture onto a structure image.

    Args:
        struct: Structure image (HxWx3 RGB, uint8)
        texture: Texture array (HxWx3, float32)
        strength: Blend strength (0 = none, 1 = full)

    Returns:
        Textured image (HxWx3 RGB, uint8)
    """
    result = struct.astype(np.float32) + texture * strength
    return np.clip(result, 0, 255).astype(np.uint8)


def generate_shadow(
    alpha: np.ndarray,
    offset: Tuple[int, int] = (3, 3),
    blur: int = 5,
    opacity: float = 0.25,
) -> np.ndarray:
    """Generate a shadow from an alpha mask.

    Args:
        alpha: Alpha mask (HxW single channel, uint8 or float)
        offset: (dx, dy) shadow offset in pixels
        blur: Gaussian blur kernel size
        opacity: Shadow opacity 0-1

    Returns:
        Shadow as float32 single-channel array
    """
    h, w = alpha.shape[:2]
    dx, dy = offset

    canvas_h = h + abs(dy) + blur * 2
    canvas_w = w + abs(dx) + blur * 2

    canvas = np.zeros((canvas_h, canvas_w), dtype=np.float32)

    sx = blur + max(dx, 0)
    sy = blur + max(dy, 0)
    canvas[sy : sy + h, sx : sx + w] = alpha.astype(np.float32) / 255.0

    if blur > 0:
        ksize = blur * 2 + 1
        canvas = cv2.GaussianBlur(canvas, (ksize, ksize), blur / 2.0)

    canvas *= opacity
    return canvas


def apply_shadow(
    background: np.ndarray,
    shadow: np.ndarray,
    position: Tuple[int, int],
) -> np.ndarray:
    """Overlay a shadow onto a background by darkening.

    Args:
        background: Background image (HxWx3 BGR or RGB, uint8)
        shadow: Shadow array (HxW single channel, float)
        position: (x, y) top-left placement on background

    Returns:
        Background with shadow applied (uint8, modified in place)
    """
    sh, sw = shadow.shape[:2]
    bh, bw = background.shape[:2]

    px, py = position
    x1 = max(0, px)
    y1 = max(0, py)
    x2 = min(bw, px + sw)
    y2 = min(bh, py + sh)

    if x1 >= x2 or y1 >= y2:
        return background

    sx1 = x1 - px
    sy1 = y1 - py
    sx2 = sx1 + (x2 - x1)
    sy2 = sy1 + (y2 - y1)

    shadow_crop = shadow[sy1:sy2, sx1:sx2]
    bg_crop = background[y1:y2, x1:x2].astype(np.float32)

    shadow_3c = np.dstack([shadow_crop] * 3)
    darkened = bg_crop * (1.0 - shadow_3c)

    background[y1:y2, x1:x2] = np.clip(darkened, 0, 255).astype(np.uint8)
    return background


def match_and_blend(
    background: np.ndarray,
    foreground: np.ndarray,
    position: Tuple[int, int],
    color_match: bool = True,
    texture_blend: float = 0.06,
    shadow: bool = True,
    feather: int = 3,
) -> np.ndarray:
    """One-stop interface: composite foreground onto background with lighting match.

    Full pipeline: optional shadow -> alpha composite with optional color transfer
    and texture blend.

    Args:
        background: Background image (HxWx3 BGR, uint8)
        foreground: Foreground image (HxWx4 BGRA, uint8)
        position: (x, y) top-left placement on background
        color_match: Whether to apply Reinhard color transfer
        texture_blend: Texture blend strength (0 = off)
        shadow: Whether to generate and apply ambient shadow
        feather: Alpha feather radius in pixels

    Returns:
        Composited background (HxWx3 BGR, uint8)
    """
    from .basic import alpha_composite, feather_alpha

    x, y = position
    fg_h, fg_w = foreground.shape[:2]

    result = background.copy()

    if feather > 0:
        fg = foreground.copy()
        fg[:, :, 3] = feather_alpha(fg[:, :, 3], radius=feather)
    else:
        fg = foreground

    if color_match:
        bg_roi_rgb = cv2.cvtColor(
            background[y : y + fg_h, x : x + fg_w], cv2.COLOR_BGR2RGB
        )
        fg_mask = fg[:, :, 3]

        fg_rgb = cv2.cvtColor(fg[:, :, :3], cv2.COLOR_BGR2RGB)

        matched_rgb = color_transfer(fg_rgb, bg_roi_rgb, mask=fg_mask)
        fg[:, :, :3] = cv2.cvtColor(matched_rgb, cv2.COLOR_RGB2BGR)

    if texture_blend > 0:
        bg_roi_bgr = background[y : y + fg_h, x : x + fg_w]
        bg_roi_rgb = cv2.cvtColor(bg_roi_bgr, cv2.COLOR_BGR2RGB)
        fg_rgb = cv2.cvtColor(fg[:, :, :3], cv2.COLOR_BGR2RGB)

        tex = extract_texture(bg_roi_rgb)
        textured_rgb = apply_texture(fg_rgb, tex, strength=texture_blend)
        fg[:, :, :3] = cv2.cvtColor(textured_rgb, cv2.COLOR_RGB2BGR)

    if shadow:
        shadow_map = generate_shadow(fg[:, :, 3])
        sx = x - 3
        sy = y - 3
        result = apply_shadow(result, shadow_map, (sx, sy))

    result = alpha_composite(result, fg, x, y)
    return result
