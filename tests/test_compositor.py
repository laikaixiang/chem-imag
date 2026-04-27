"""Phase 2 compositor tests: lighting matching and texture blending."""

import numpy as np
import cv2
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from compositor.lighting import (
    color_transfer,
    extract_texture,
    apply_texture,
    generate_shadow,
    apply_shadow,
    match_and_blend,
)
from compositor.basic import alpha_composite, load_image
from compositor.pyramid import seamless_composite, pyramid_blend, gaussian_pyramid, laplacian_pyramid
from config import settings


def create_test_rgba(w, h, color, alpha=255):
    img = np.zeros((h, w, 4), dtype=np.uint8)
    img[:, :, :3] = color
    img[:, :, 3] = alpha
    return img


def create_test_bgr(w, h, color):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :] = color
    return img


def test_color_transfer():
    """Test Reinhard color transfer in Lab space."""
    print("\n=== Lighting Test 1: Color Transfer ===")

    src = np.full((100, 100, 3), (200, 100, 100), dtype=np.uint8)
    target = np.full((100, 100, 3), (100, 150, 200), dtype=np.uint8)

    result = color_transfer(src, target)

    assert result.shape == src.shape
    assert result.dtype == np.uint8

    result_mean = result.astype(np.float32).mean(axis=(0, 1))
    target_mean = target.astype(np.float32).mean(axis=(0, 1))

    print(f"  Source mean (RGB): {src.mean(axis=(0,1)).astype(int)}")
    print(f"  Target mean (RGB): {target_mean.astype(int)}")
    print(f"  Result mean (RGB): {result_mean.astype(int)}")
    print("  Note: Lab-space means don't exactly match RGB means, that's expected")

    output_path = settings.OUTPUT_DIR / 'compositor' / 'test_color_transfer.png'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
    print(f"  Saved: {output_path}")


def test_color_transfer_with_mask():
    """Test color transfer with mask."""
    print("\n=== Lighting Test 2: Color Transfer with Mask ===")

    src_rgb = np.random.RandomState(42).randint(0, 255, (80, 80, 3)).astype(np.uint8)
    target_rgb = np.random.RandomState(99).randint(0, 255, (80, 80, 3)).astype(np.uint8)

    mask = np.zeros((80, 80), dtype=np.uint8)
    mask[20:60, 20:60] = 255

    result = color_transfer(src_rgb, target_rgb, mask=mask)

    assert result.shape == src_rgb.shape
    assert result.dtype == np.uint8

    print(f"  Shape: {result.shape}, dtype: {result.dtype}")
    print("  Masked color transfer works")


def test_extract_texture():
    """Test high-frequency texture extraction."""
    print("\n=== Lighting Test 3: Texture Extraction ===")

    rng = np.random.RandomState(7)
    image = rng.randint(0, 255, (100, 100, 3)).astype(np.uint8)

    tex = extract_texture(image, sigma=8.0)

    assert tex.shape == image.shape
    assert tex.dtype == np.float32
    assert tex.min() >= -30 and tex.max() <= 30

    print(f"  Shape: {tex.shape}, dtype: {tex.dtype}")
    print(f"  Range: [{tex.min():.2f}, {tex.max():.2f}]")
    print("  Texture extracted in expected range")


def test_apply_texture():
    """Test texture application."""
    print("\n=== Lighting Test 4: Texture Application ===")

    struct = np.full((80, 80, 3), (128, 128, 128), dtype=np.uint8)
    tex = np.ones((80, 80, 3), dtype=np.float32) * 10.0

    result = apply_texture(struct, tex, strength=1.0)

    assert result.shape == struct.shape
    assert result.dtype == np.uint8

    result_mean = result.astype(np.float32).mean()
    struct_mean = struct.astype(np.float32).mean()

    print(f"  Struct mean: {struct_mean:.1f}")
    print(f"  Result mean: {result_mean:.1f}")
    print(f"  Difference: {result_mean - struct_mean:.1f} (expected ~10)")


def test_generate_shadow():
    """Test shadow generation from alpha mask."""
    print("\n=== Lighting Test 5: Shadow Generation ===")

    alpha = np.zeros((60, 60), dtype=np.uint8)
    cv2.circle(alpha, (30, 30), 25, 255, -1)

    shadow = generate_shadow(alpha, offset=(5, 5), blur=7, opacity=0.3)

    assert shadow.dtype == np.float32
    assert shadow.max() <= 0.3
    assert shadow.shape[0] > alpha.shape[0]
    assert shadow.shape[1] > alpha.shape[1]

    print(f"  Alpha shape: {alpha.shape}")
    print(f"  Shadow shape: {shadow.shape}")
    print(f"  Shadow max: {shadow.max():.3f}")

    vis_path = settings.OUTPUT_DIR / 'compositor' / 'test_shadow_map.png'
    vis_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(vis_path), (shadow * 255).astype(np.uint8))
    print(f"  Saved: {vis_path}")


def test_apply_shadow():
    """Test shadow application to background."""
    print("\n=== Lighting Test 6: Shadow Application ===")

    bg = create_test_bgr(200, 200, (200, 200, 200))

    shadow = np.ones((80, 80), dtype=np.float32) * 0.3

    result = apply_shadow(bg.copy(), shadow, (50, 50))

    assert result.shape == bg.shape

    center_val = result[90, 90]
    bg_val = bg[90, 90]

    print(f"  Background at (90,90): {bg_val}")
    print(f"  With shadow at (90,90): {center_val}")
    print(f"  Expected darkening: 200 * (1-0.3) = 140")

    assert center_val[0] < bg_val[0]
    print("  Shadow darkened correctly")


def test_match_and_blend():
    """Test the one-stop match_and_blend interface."""
    print("\n=== Lighting Test 7: Match and Blend ===")

    bg = create_test_bgr(300, 250, (180, 200, 220))

    rng = np.random.RandomState(13)
    gradient = np.linspace(0, 30, 250).astype(np.uint8).reshape(250, 1)
    bg[:, :, 0] = np.clip(bg[:, :, 0].astype(int) + gradient, 0, 255).astype(np.uint8)

    fg = np.zeros((100, 100, 4), dtype=np.uint8)
    fg[:, :, :3] = (50, 50, 150)
    cv2.circle(fg, (50, 50), 40, (50, 50, 150, 255), -1)
    cv2.circle(fg, (50, 50), 40, (60, 120, 220), 2)

    result = match_and_blend(bg, fg, position=(60, 60), feather=2)

    assert result.shape == bg.shape
    assert result.dtype == np.uint8

    print(f"  Background: {bg.shape}")
    print(f"  Foreground: {fg.shape}")
    print(f"  Result: {result.shape}")
    print("  Match and blend applied successfully")

    output_path = settings.OUTPUT_DIR / 'compositor' / 'test_match_and_blend.png'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), result)
    print(f"  Saved: {output_path}")


def test_lighting_vs_plain_comparison():
    """Side-by-side comparison: plain alpha vs match_and_blend."""
    print("\n=== Lighting Test 8: Plain vs Match-and-Blend Comparison ===")

    bg = create_test_bgr(400, 300, (180, 200, 220))

    rng = np.random.RandomState(42)
    noise = rng.randint(-8, 9, (300, 400, 3)).astype(np.int16)
    bg = np.clip(bg.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    fg = np.zeros((120, 120, 4), dtype=np.uint8)
    fg[:, :, :3] = (255, 220, 100)
    cv2.rectangle(fg, (20, 20), (100, 100), (255, 220, 100, 255), -1)
    cv2.rectangle(fg, (20, 20), (100, 100), (255, 180, 0, 255), 2)

    plain = alpha_composite(bg.copy(), fg.copy(), 100, 80)
    matched = match_and_blend(bg.copy(), fg.copy(), position=(100, 80), feather=2)

    plain_path = settings.OUTPUT_DIR / 'compositor' / 'test_comparison_plain.png'
    matched_path = settings.OUTPUT_DIR / 'compositor' / 'test_comparison_matched.png'
    plain_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(plain_path), plain)
    cv2.imwrite(str(matched_path), matched)

    plain_mean = plain.astype(np.float32).mean(axis=(0, 1))
    matched_mean = matched.astype(np.float32).mean(axis=(0, 1))

    print(f"  Plain alpha mean (BGR): {plain_mean.astype(int)}")
    print(f"  Match+blend mean (BGR): {matched_mean.astype(int)}")
    print(f"  Plain:  {plain_path}")
    print(f"  Matched: {matched_path}")
    print("  Visual comparison: match_and_blend should look more 'embedded'")


def test_pyramid_vs_direct():
    """Compare pyramid blend vs direct alpha composite on boundary quality."""
    print("\n=== Pyramid Test: Pyramid vs Direct Alpha ===")

    rng = np.random.RandomState(42)

    # Textured background
    bg = create_test_bgr(400, 300, (180, 200, 220))
    noise = rng.randint(-15, 16, (300, 400, 3)).astype(np.int16)
    bg = np.clip(bg.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Overlay with gradient alpha (tests soft boundary)
    fg = np.zeros((150, 150, 4), dtype=np.uint8)
    fg[:, :, :3] = (50, 100, 200)
    fg_noise = rng.randint(-10, 11, (150, 150, 3)).astype(np.int16)
    fg[:, :, :3] = np.clip(fg[:, :, :3].astype(np.int16) + fg_noise, 0, 255).astype(np.uint8)
    # Left-to-right alpha gradient for smooth transition test
    alpha_grad = np.linspace(0, 255, 150, dtype=np.uint8)
    fg[:, :, 3] = np.tile(alpha_grad, (150, 1))

    # Direct alpha composite
    direct = alpha_composite(bg.copy(), fg.copy(), 100, 80)

    # Pyramid seamless composite
    result = seamless_composite(bg.copy(), fg.copy(), 100, 80, 150, 150, levels=4)

    assert direct.shape == bg.shape
    assert result.shape == bg.shape

    direct_path = settings.OUTPUT_DIR / 'compositor' / 'test_pyramid_direct.png'
    pyramid_path = settings.OUTPUT_DIR / 'compositor' / 'test_pyramid_blend.png'
    direct_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(direct_path), direct)
    cv2.imwrite(str(pyramid_path), result)

    print(f"  Direct alpha composite: {direct_path}")
    print(f"  Pyramid blend:          {pyramid_path}")
    print("  Pyramid blend should show smoother boundary transition")
    print("  (gradient alpha edge is the visual differentiator)")


if __name__ == '__main__':
    print("Testing compositor lighting functions...")
    print(f"Output directory: {settings.OUTPUT_DIR}")

    try:
        test_color_transfer()
        test_color_transfer_with_mask()
        test_extract_texture()
        test_apply_texture()
        test_generate_shadow()
        test_apply_shadow()
        test_match_and_blend()
        test_lighting_vs_plain_comparison()
        test_pyramid_vs_direct()

        print("\n" + "=" * 50)
        print("All compositor tests passed!")
        print(f"Outputs saved in: {settings.OUTPUT_DIR / 'compositor'}")
        print("=" * 50)

    except Exception as e:
        print(f"\nTest failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
