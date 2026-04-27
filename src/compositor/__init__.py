"""Compositor module for image composition and blending."""

from .basic import (
    load_image,
    alpha_composite,
    resize_with_alpha,
    perspective_transform,
    feather_alpha,
    add_drop_shadow,
)

from .lighting import (
    color_transfer,
    extract_texture,
    apply_texture,
    generate_shadow,
    apply_shadow,
    match_and_blend,
)

from .pyramid import (
    gaussian_pyramid,
    laplacian_pyramid,
    pyramid_blend,
    seamless_composite,
)

__all__ = [
    'load_image',
    'alpha_composite',
    'resize_with_alpha',
    'perspective_transform',
    'feather_alpha',
    'add_drop_shadow',
    'color_transfer',
    'extract_texture',
    'apply_texture',
    'generate_shadow',
    'apply_shadow',
    'match_and_blend',
    'gaussian_pyramid',
    'laplacian_pyramid',
    'pyramid_blend',
    'seamless_composite',
]
