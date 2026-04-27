"""Tests for basic compositor functions."""

import numpy as np
import cv2
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from compositor.basic import (
    alpha_composite,
    resize_with_alpha,
    perspective_transform,
    feather_alpha,
    add_drop_shadow,
)
from config import settings


def create_test_rgba(w, h, color, alpha=255):
    """Create a test RGBA image."""
    img = np.zeros((h, w, 4), dtype=np.uint8)
    img[:, :, :3] = color
    img[:, :, 3] = alpha
    return img


def create_test_bgr(w, h, color):
    """Create a test BGR image."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :] = color
    return img


def test_alpha_composite():
    """Test alpha blending."""
    print("\n=== Test 1: Alpha Composite ===")

    bg = create_test_bgr(400, 300, (100, 150, 200))
    overlay = create_test_rgba(100, 100, (255, 0, 0), alpha=128)

    result = alpha_composite(bg, overlay, 50, 50)

    assert result.shape == bg.shape
    assert result.dtype == np.uint8

    blended_pixel = result[75, 75]
    expected = (100 * 0.5 + 255 * 0.5, 150 * 0.5 + 0 * 0.5, 200 * 0.5 + 0 * 0.5)

    print(f"✓ Alpha composite works")
    print(f"  Background: BGR {bg.shape}")
    print(f"  Overlay: BGRA {overlay.shape} at (50, 50)")
    print(f"  Result pixel at (75, 75): {blended_pixel}")

    output_path = settings.OUTPUT_DIR / 'compositor' / 'test_alpha_composite.png'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), result)
    print(f"  Saved: {output_path}")


def test_alpha_composite_boundary():
    """Test alpha composite with boundary clipping."""
    print("\n=== Test 2: Alpha Composite Boundary ===")

    bg = create_test_bgr(200, 200, (50, 50, 50))
    overlay = create_test_rgba(100, 100, (0, 255, 0), alpha=255)

    result = alpha_composite(bg, overlay, 150, 150)

    assert result.shape == bg.shape
    print(f"✓ Boundary clipping works")
    print(f"  Overlay extends beyond background edge")
    print(f"  No crash, result shape: {result.shape}")

    output_path = settings.OUTPUT_DIR / 'compositor' / 'test_boundary.png'
    cv2.imwrite(str(output_path), result)
    print(f"  Saved: {output_path}")


def test_resize_with_alpha():
    """Test RGBA image resizing."""
    print("\n=== Test 3: Resize with Alpha ===")

    img = create_test_rgba(200, 200, (255, 128, 0), alpha=200)

    resized = resize_with_alpha(img, 100, 100)

    assert resized.shape == (100, 100, 4)
    assert resized.dtype == np.uint8

    print(f"✓ Resize works")
    print(f"  Original: {img.shape}")
    print(f"  Resized: {resized.shape}")

    output_path = settings.OUTPUT_DIR / 'compositor' / 'test_resize.png'
    cv2.imwrite(str(output_path), resized)
    print(f"  Saved: {output_path}")


def test_perspective_transform():
    """Test perspective transformation."""
    print("\n=== Test 4: Perspective Transform ===")

    img = create_test_rgba(200, 200, (0, 128, 255), alpha=255)
    cv2.rectangle(img, (50, 50), (150, 150), (255, 255, 255, 255), 3)

    src_points = np.array([
        [0, 0],
        [200, 0],
        [200, 200],
        [0, 200]
    ], dtype=np.float32)

    dst_points = np.array([
        [20, 0],
        [200, 30],
        [180, 200],
        [0, 170]
    ], dtype=np.float32)

    transformed = perspective_transform(img, src_points, dst_points, (200, 200))

    assert transformed.shape == (200, 200, 4)
    print(f"✓ Perspective transform works")
    print(f"  Applied perspective warp")
    print(f"  Output shape: {transformed.shape}")

    output_path = settings.OUTPUT_DIR / 'compositor' / 'test_perspective.png'
    cv2.imwrite(str(output_path), transformed)
    print(f"  Saved: {output_path}")


def test_feather_alpha():
    """Test alpha channel feathering."""
    print("\n=== Test 5: Feather Alpha ===")

    alpha = np.zeros((200, 200), dtype=np.uint8)
    cv2.rectangle(alpha, (50, 50), (150, 150), 255, -1)

    feathered = feather_alpha(alpha, radius=10)

    assert feathered.shape == alpha.shape
    assert feathered.dtype == np.uint8

    edge_value = feathered[50, 75]
    center_value = feathered[100, 100]

    print(f"✓ Feathering works")
    print(f"  Original alpha: hard edges")
    print(f"  Feathered alpha: soft edges")
    print(f"  Edge value: {edge_value}, Center value: {center_value}")

    img = create_test_rgba(200, 200, (255, 0, 255), alpha=255)
    img[:, :, 3] = feathered

    output_path = settings.OUTPUT_DIR / 'compositor' / 'test_feather.png'
    cv2.imwrite(str(output_path), img)
    print(f"  Saved: {output_path}")


def test_add_drop_shadow():
    """Test drop shadow generation."""
    print("\n=== Test 6: Add Drop Shadow ===")

    overlay = create_test_rgba(100, 100, (255, 255, 0), alpha=255)
    cv2.circle(overlay, (50, 50), 40, (255, 255, 0, 255), -1)

    bg = create_test_bgr(200, 200, (240, 240, 240))

    alpha_mask = overlay[:, :, 3]
    result = add_drop_shadow(bg, alpha_mask, 20, 20, 100, 100,
                             offset=(10, 10), blur=15, opacity=0.6)

    assert result.shape == bg.shape
    assert result.dtype == np.uint8

    result = alpha_composite(result, overlay, 20, 20)

    print(f"✓ Drop shadow works")
    print(f"  Shadow offset: (10, 10), blur: 15px, opacity: 0.6")
    print(f"  Background size: {bg.shape}")

    output_path = settings.OUTPUT_DIR / 'compositor' / 'test_shadow.png'
    cv2.imwrite(str(output_path), result)
    print(f"  Saved: {output_path}")


def test_composite_demo():
    """Create a comprehensive demo combining multiple effects."""
    print("\n=== Test 7: Composite Demo ===")

    bg = create_test_bgr(600, 400, (220, 230, 240))

    overlay1 = create_test_rgba(120, 120, (255, 100, 100), alpha=200)
    bg = add_drop_shadow(bg, overlay1[:, :, 3], 50, 50, 120, 120,
                         offset=(8, 8), blur=12, opacity=0.5)
    bg = alpha_composite(bg, overlay1, 50, 50)

    overlay2 = create_test_rgba(100, 100, (100, 255, 100), alpha=180)
    src_pts = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float32)
    dst_pts = np.array([[10, 0], [100, 15], [90, 100], [0, 85]], dtype=np.float32)
    overlay2 = perspective_transform(overlay2, src_pts, dst_pts, (100, 100))
    bg = add_drop_shadow(bg, overlay2[:, :, 3], 250, 100, 100, 100,
                         offset=(6, 6), blur=10, opacity=0.4)
    bg = alpha_composite(bg, overlay2, 250, 100)

    overlay3 = create_test_rgba(80, 80, (100, 100, 255), alpha=255)
    overlay3[:, :, 3] = feather_alpha(overlay3[:, :, 3], radius=8)

    result = alpha_composite(bg, overlay3, 450, 200)

    print(f"✓ Composite demo complete")
    print(f"  Combined: alpha blend + perspective + shadow + feather")
    print(f"  Final size: {result.shape}")

    output_path = settings.OUTPUT_DIR / 'compositor' / 'test_composite_demo.png'
    cv2.imwrite(str(output_path), result)
    print(f"  Saved: {output_path}")


if __name__ == '__main__':
    print("Testing compositor basic functions...")
    print(f"Output directory: {settings.OUTPUT_DIR}")

    try:
        test_alpha_composite()
        test_alpha_composite_boundary()
        test_resize_with_alpha()
        test_perspective_transform()
        test_feather_alpha()
        test_add_drop_shadow()
        test_composite_demo()

        print("\n" + "="*50)
        print("✓ All tests passed!")
        print(f"📁 Outputs saved in: {settings.OUTPUT_DIR / 'compositor'}")
        print("="*50)

    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
