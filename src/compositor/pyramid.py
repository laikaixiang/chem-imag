"""Laplacian pyramid multi-scale fusion for seamless image compositing.

Phase 2c: pyramid blending eliminates hard seams by blending at
multiple spatial frequencies (coarse to fine).
"""

import cv2
import numpy as np
from typing import List


def gaussian_pyramid(image: np.ndarray, levels: int = 4) -> List[np.ndarray]:
    """Build gaussian pyramid via repeated pyrDown.

    Args:
        image: Input image (HxWxC)
        levels: Number of pyramid levels

    Returns:
        List of images from original size [0] to coarsest [levels-1]
    """
    gp = [image.copy()]
    for _ in range(levels - 1):
        gp.append(cv2.pyrDown(gp[-1]))
    return gp


def laplacian_pyramid(gp: List[np.ndarray]) -> List[np.ndarray]:
    """Build laplacian pyramid from gaussian pyramid.

    lp[i] = gp[i] - pyrUp(gp[i+1])
    lp[-1] = gp[-1]

    Args:
        gp: Gaussian pyramid from gaussian_pyramid()

    Returns:
        Laplacian pyramid (list of float32 arrays)
    """
    lp = []
    for i in range(len(gp) - 1):
        h, w = gp[i].shape[:2]
        up = cv2.pyrUp(gp[i + 1], dstsize=(w, h))
        lp.append(gp[i].astype(np.float32) - up.astype(np.float32))
    lp.append(gp[-1].astype(np.float32))
    return lp


def pyramid_blend(
    bg: np.ndarray,
    fg: np.ndarray,
    mask: np.ndarray,
    levels: int = 4
) -> np.ndarray:
    """Blend foreground and background using laplacian pyramid fusion.

    Core algorithm:
      1. Build gaussian pyramids for bg, fg, mask
      2. Build laplacian pyramids for bg, fg
      3. Blend each level: blended[i] = lp_fg[i]*mask[i] + lp_bg[i]*(1-mask[i])
      4. Reconstruct from coarsest to finest: pyrUp + add

    Args:
        bg: Background image (HxWx3) uint8
        fg: Foreground image (HxWx3) uint8
        mask: Blend mask (HxW) uint8, 255 = full foreground
        levels: Number of pyramid levels

    Returns:
        Blended image (HxWx3) uint8
    """
    gp_bg = gaussian_pyramid(bg, levels)
    gp_fg = gaussian_pyramid(fg, levels)
    gp_mask = gaussian_pyramid(mask, levels)

    lp_bg = laplacian_pyramid(gp_bg)
    lp_fg = laplacian_pyramid(gp_fg)

    blended = []
    for i in range(levels):
        m = gp_mask[i].astype(np.float32) / 255.0
        if m.ndim == 2:
            m = np.dstack([m] * 3)
        blended.append(lp_fg[i] * m + lp_bg[i] * (1 - m))

    result = blended[-1]
    for i in range(levels - 2, -1, -1):
        result = cv2.pyrUp(result, dstsize=(blended[i].shape[1], blended[i].shape[0]))
        result = result + blended[i]

    return np.clip(result, 0, 255).astype(np.uint8)


def seamless_composite(
    bg: np.ndarray,
    overlay: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
    levels: int = 3
) -> np.ndarray:
    """Seamlessly composite overlay onto background using pyramid blending.

    1. Resize overlay to (w, h)
    2. Extract alpha as mask
    3. Crop overlapping ROI from bg and fg
    4. Pyramid blend the ROI
    5. Write blended ROI back to result

    Args:
        bg: Background image (HxWx3 BGR)
        overlay: Overlay image (HxWx4 BGRA) with alpha channel
        x, y: Top-left position on background
        w, h: Target width and height for overlay
        levels: Pyramid levels for blending (more = smoother transition)

    Returns:
        Composited image (HxWx3 BGR)
    """
    from .basic import resize_with_alpha

    result = bg.copy()
    bg_h, bg_w = bg.shape[:2]

    overlay_resized = resize_with_alpha(overlay, w, h)
    fg_rgb = overlay_resized[:, :, :3]
    alpha = overlay_resized[:, :, 3]

    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(bg_w, x + w)
    y2 = min(bg_h, y + h)

    if x1 >= x2 or y1 >= y2:
        return result

    fg_x1 = x1 - x
    fg_y1 = y1 - y
    fg_x2 = fg_x1 + (x2 - x1)
    fg_y2 = fg_y1 + (y2 - y1)

    bg_roi = bg[y1:y2, x1:x2]
    fg_crop = fg_rgb[fg_y1:fg_y2, fg_x1:fg_x2]
    mask_crop = alpha[fg_y1:fg_y2, fg_x1:fg_x2]

    blended_roi = pyramid_blend(bg_roi, fg_crop, mask_crop, levels)
    result[y1:y2, x1:x2] = blended_roi

    return result
