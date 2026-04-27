"""Basic image composition functions for chem-mindmap.

Provides core functionality for alpha blending, perspective transforms,
and image manipulation operations.
"""

import cv2
import numpy as np
from typing import Tuple, Optional


def load_image(path: str, mode: str = 'auto') -> np.ndarray:
    """Load image with specified color mode.

    Args:
        path: Path to image file
        mode: Color mode - 'auto', 'rgb', 'bgr', or 'rgba'

    Returns:
        Loaded image as numpy array
    """
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {path}")

    has_alpha = img.shape[2] == 4 if len(img.shape) == 3 else False

    if mode == 'auto':
        return img if has_alpha else img
    elif mode == 'rgb':
        if has_alpha:
            return cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif mode == 'bgr':
        if has_alpha:
            return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return img
    elif mode == 'rgba':
        if has_alpha:
            return cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
        bgr = img if len(img.shape) == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        alpha = np.ones((bgr.shape[0], bgr.shape[1], 1), dtype=bgr.dtype) * 255
        return cv2.cvtColor(np.concatenate([bgr, alpha], axis=2), cv2.COLOR_BGRA2RGBA)
    else:
        raise ValueError(f"Invalid mode: {mode}")


def alpha_composite(bg: np.ndarray, overlay: np.ndarray, x: int, y: int) -> np.ndarray:
    """Composite RGBA overlay onto BGR background using alpha blending.

    Args:
        bg: Background image (H×W×3 BGR)
        overlay: Overlay image (H×W×4 BGRA)
        x: X position to place overlay
        y: Y position to place overlay

    Returns:
        Composited image (H×W×3 BGR)
    """
    result = bg.copy()

    overlay_h, overlay_w = overlay.shape[:2]
    bg_h, bg_w = bg.shape[:2]

    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(bg_w, x + overlay_w)
    y2 = min(bg_h, y + overlay_h)

    if x1 >= x2 or y1 >= y2:
        return result

    overlay_x1 = x1 - x
    overlay_y1 = y1 - y
    overlay_x2 = overlay_x1 + (x2 - x1)
    overlay_y2 = overlay_y1 + (y2 - y1)

    overlay_crop = overlay[overlay_y1:overlay_y2, overlay_x1:overlay_x2]
    bg_crop = bg[y1:y2, x1:x2]

    alpha = overlay_crop[:, :, 3:4].astype(np.float32) / 255.0
    overlay_rgb = overlay_crop[:, :, :3].astype(np.float32)
    bg_rgb = bg_crop.astype(np.float32)

    blended = overlay_rgb * alpha + bg_rgb * (1 - alpha)
    result[y1:y2, x1:x2] = blended.astype(np.uint8)

    return result


def resize_with_alpha(img: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    """Resize RGBA image with high-quality interpolation.

    Args:
        img: RGBA image (H×W×4)
        target_w: Target width
        target_h: Target height

    Returns:
        Resized RGBA image
    """
    return cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)


def perspective_transform(
    img: np.ndarray,
    src_points: np.ndarray,
    dst_points: np.ndarray,
    output_size: Tuple[int, int]
) -> np.ndarray:
    """Apply perspective transform to RGBA image.

    Args:
        img: RGBA image (H×W×4)
        src_points: Source quadrilateral corners (4×2)
        dst_points: Destination quadrilateral corners (4×2)
        output_size: Output image size (width, height)

    Returns:
        Transformed RGBA image
    """
    matrix = cv2.getPerspectiveTransform(
        src_points.astype(np.float32),
        dst_points.astype(np.float32)
    )

    return cv2.warpPerspective(
        img,
        matrix,
        output_size,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0)
    )


def feather_alpha(alpha: np.ndarray, radius: int = 3) -> np.ndarray:
    """Apply feathering (soft edge) to alpha channel.

    Args:
        alpha: Alpha channel (H×W single channel)
        radius: Feather radius in pixels

    Returns:
        Feathered alpha channel
    """
    if radius <= 0:
        return alpha

    kernel_size = radius * 2 + 1
    return cv2.GaussianBlur(alpha, (kernel_size, kernel_size), radius / 2.0)


def add_drop_shadow(
    bg: np.ndarray,
    alpha_mask: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
    offset: Tuple[int, int] = (3, 3),
    blur: int = 5,
    opacity: float = 0.3
) -> np.ndarray:
    """Add drop shadow directly to background image.

    Args:
        bg: Background image (H×W×3 BGR)
        alpha_mask: Alpha mask of the overlay (will be resized to w×h)
        x: Top-left X position of overlay on background
        y: Top-left Y position of overlay on background
        w: Target width for alpha mask
        h: Target height for alpha mask
        offset: Shadow offset (dx, dy) in pixels
        blur: Gaussian blur radius for shadow
        opacity: Shadow opacity (0-1)

    Returns:
        Background image with shadow (H×W×3 BGR)
    """
    result = bg.copy()
    bg_h, bg_w = bg.shape[:2]

    if alpha_mask.shape[0] != h or alpha_mask.shape[1] != w:
        alpha_mask = cv2.resize(alpha_mask, (w, h), interpolation=cv2.INTER_LANCZOS4)

    shadow_canvas = np.zeros((bg_h, bg_w), dtype=np.float32)

    sx = x + offset[0]
    sy = y + offset[1]

    shadow_x1 = max(0, sx)
    shadow_y1 = max(0, sy)
    shadow_x2 = min(bg_w, sx + w)
    shadow_y2 = min(bg_h, sy + h)

    if shadow_x1 < shadow_x2 and shadow_y1 < shadow_y2:
        mask_x1 = shadow_x1 - sx
        mask_y1 = shadow_y1 - sy
        mask_x2 = mask_x1 + (shadow_x2 - shadow_x1)
        mask_y2 = mask_y1 + (shadow_y2 - shadow_y1)
        shadow_canvas[shadow_y1:shadow_y2, shadow_x1:shadow_x2] = \
            alpha_mask[mask_y1:mask_y2, mask_x1:mask_x2].astype(np.float32)

    if blur > 0:
        kernel_size = blur * 2 + 1
        shadow_canvas = cv2.GaussianBlur(shadow_canvas, (kernel_size, kernel_size), blur / 2.0)

    shadow_canvas = shadow_canvas / 255.0 * opacity

    for c in range(3):
        result[:, :, c] = (result[:, :, c].astype(np.float32) * (1 - shadow_canvas)).astype(np.uint8)

    return result
