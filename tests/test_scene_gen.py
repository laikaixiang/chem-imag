"""Tests for AI scene generation (Phase 4)."""

import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.scene_gen.generator import SceneGenerator, ENHANCE_PROMPTS, GUIDANCE_MODIFIERS
from src.config import settings


def create_test_mindmap(w: int = 512, h: int = 512) -> np.ndarray:
    """Create a simple mindmap-like skeleton image for testing."""
    img = np.ones((h, w, 3), dtype=np.uint8) * 255
    cv2.rectangle(img, (50, 50), (w - 50, h - 50), (0, 0, 0), 3)
    cv2.line(img, (w // 2, 60), (w // 2, h - 60), (0, 0, 0), 2)
    for i in range(3):
        y = 130 + i * 100
        cv2.line(img, (w // 2, y), (w // 2 - 120, y - 30), (0, 0, 0), 2)
        cv2.line(img, (w // 2, y), (w // 2 + 120, y - 30), (0, 0, 0), 2)
    cv2.putText(img, "Organic Chemistry", (w // 2 - 100, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    cv2.putText(img, "Alcohols", (w // 2 - 190, 140),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.putText(img, "Ketones", (w // 2 + 80, 140),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.putText(img, "Reactions", (w // 2 - 190, 240),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.putText(img, "Properties", (w // 2 + 80, 240),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    return img


def test_mock_generate():
    """Test mock mode image generation."""
    print("\n=== Test 1: Mock Generate ===")

    gen = SceneGenerator(provider="mock")
    img = gen.generate("test prompt", width=512, height=512)

    assert isinstance(img, Image.Image), f"Expected PIL Image, got {type(img)}"
    assert img.size == (512, 512), f"Expected (512, 512), got {img.size}"
    assert img.mode == "RGB"

    print(f"  ✓ Mock generate returns {img.size} {img.mode} image")

    output_path = settings.OUTPUT_DIR / "scenes" / "test_mock_generate.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(output_path))
    print(f"  Saved: {output_path}")


def test_enhance_mindmap_mock():
    """Test mindmap enhancement with mock provider."""
    print("\n=== Test 2: Enhance Mindmap (Mock) ===")

    mindmap = create_test_mindmap()
    gen = SceneGenerator(provider="mock")
    img = gen.enhance_mindmap(mindmap, style_prompt="academic", width=512, height=512)

    assert isinstance(img, Image.Image)
    assert img.size == (512, 512)
    assert img.mode == "RGB"

    print(f"  ✓ Enhance mindmap (mock) returns {img.size} {img.mode} image")

    output_path = settings.OUTPUT_DIR / "scenes" / "test_enhance_mock.png"
    img.save(str(output_path))
    print(f"  Saved: {output_path}")


def test_canny_edge_extraction():
    """Test Canny edge extraction from a mindmap image."""
    print("\n=== Test 3: Canny Edge Extraction ===")

    mindmap = create_test_mindmap()
    gen = SceneGenerator(provider="mock")
    edges = gen._extract_canny_edges(mindmap)

    assert isinstance(edges, np.ndarray)
    assert edges.ndim == 2, f"Expected 2D, got {edges.ndim}D"
    assert edges.dtype == np.uint8
    assert edges.shape == (512, 512)

    edge_ratio = np.count_nonzero(edges) / edges.size
    print(f"  ✓ Canny edges extracted: {edges.shape}, edge pixel ratio: {edge_ratio:.3f}")

    assert edge_ratio > 0.0, "Expected some edge pixels"

    output_path = settings.OUTPUT_DIR / "scenes" / "test_canny_edges.png"
    cv2.imwrite(str(output_path), edges)
    print(f"  Saved: {output_path}")


def test_canny_edge_extraction_bgr():
    """Test Canny edge extraction from BGR input."""
    print("\n=== Test 4: Canny Edge Extraction (BGR) ===")

    bgr = create_test_mindmap()
    gen = SceneGenerator(provider="mock")
    edges = gen._extract_canny_edges(bgr)

    assert edges.shape == (512, 512)
    print(f"  ✓ Canny from BGR: {edges.shape}")


def test_canny_edge_extraction_rgba():
    """Test Canny edge extraction from RGBA input."""
    print("\n=== Test 5: Canny Edge Extraction (RGBA) ===")

    rgba = np.zeros((256, 256, 4), dtype=np.uint8)
    rgba[:, :, :3] = 255
    rgba[:, :, 3] = 255
    cv2.rectangle(rgba, (50, 50), (200, 200), (0, 0, 0, 255), 2)

    gen = SceneGenerator(provider="mock")
    edges = gen._extract_canny_edges(rgba)

    assert edges.shape == (256, 256)
    print(f"  ✓ Canny from RGBA: {edges.shape}")


def test_generate_style_prompt():
    """Test automatic style prompt generation from mindmap metadata."""
    print("\n=== Test 6: Generate Style Prompt ===")

    gen = SceneGenerator(provider="mock")

    mindmap_json = {
        "title": "Alcohol Reactions",
        "topics": ["Oxidation", "Esterification", "Dehydration", "Substitution"],
    }

    prompt = gen.generate_style_prompt(mindmap_json, style="academic")
    assert "Alcohol Reactions" in prompt
    assert "Oxidation" in prompt
    assert "academic mind map" in prompt
    print(f"  ✓ Academic prompt generated")
    print(f"    Prompt: {prompt[:120]}...")

    prompt_modern = gen.generate_style_prompt(mindmap_json, style="modern")
    assert "infographic" in prompt_modern
    print(f"  ✓ Modern prompt generated")
    print(f"    Prompt: {prompt_modern[:120]}...")

    prompt_minimal = gen.generate_style_prompt(mindmap_json, style="minimal")
    assert "black and white" in prompt_minimal
    print(f"  ✓ Minimal prompt generated")
    print(f"    Prompt: {prompt_minimal[:120]}...")

    no_title = gen.generate_style_prompt({"topics": ["Phenols"]}, style="academic")
    assert "Phenols" in no_title
    print(f"  ✓ Prompt without title generated")


def test_build_guided_prompt():
    """Test _build_guided_prompt adds guidance modifiers."""
    print("\n=== Test 7: Build Guided Prompt ===")

    gen = SceneGenerator(provider="mock")

    p = gen._build_guided_prompt("test prompt", "canny")
    assert "straight-on angle" in p
    print(f"  ✓ Canny guidance added: {p}")

    p = gen._build_guided_prompt("test prompt", None)
    assert "high quality" in p
    print(f"  ✓ Default quality guidance added: {p}")

    p = gen._build_guided_prompt("test prompt", "depth")
    assert "depth of field" in p
    print(f"  ✓ Depth guidance added: {p}")


def test_generate_invalid_width_height():
    """Test generate with non-standard dimensions."""
    print("\n=== Test 8: Generate with Custom Dimensions ===")

    gen = SceneGenerator(provider="mock")
    img = gen.generate("test", width=768, height=768)
    assert img.size == (768, 768)
    print(f"  ✓ Custom dimensions: {img.size}")

    img = gen.generate("test", width=256, height=1024)
    assert img.size == (256, 1024)
    print(f"  ✓ Tall aspect ratio: {img.size}")


def test_prepare_control_image():
    """Test _prepare_control_image for all control types."""
    print("\n=== Test 9: Prepare Control Image ===")

    mindmap = create_test_mindmap()
    gen = SceneGenerator(provider="mock")

    for ct in ("canny", "depth", "scribble"):
        result = gen._prepare_control_image(mindmap, ct)
        assert isinstance(result, np.ndarray)
        assert result.ndim == 2, f"{ct}: expected 2D, got {result.ndim}D"
        print(f"  ✓ {ct}: {result.shape} {result.dtype}")

    try:
        gen._prepare_control_image(mindmap, "unknown_type")
        assert False, "Should have raised"
    except ValueError:
        print(f"  ✓ unknown control_type raises ValueError")


def test_enhance_prompts_coverage():
    """Test all named enhance prompts are valid."""
    print("\n=== Test 10: ENHANCE_PROMPTS Coverage ===")

    for key in ("academic", "modern", "minimal"):
        assert key in ENHANCE_PROMPTS, f"Missing prompt key: {key}"
        assert len(ENHANCE_PROMPTS[key]) > 50, f"Prompt too short: {key}"
    print(f"  ✓ All {len(ENHANCE_PROMPTS)} enhance prompts present")


def test_guidance_modifiers_coverage():
    """Test all control types have guidance modifiers."""
    print("\n=== Test 11: GUIDANCE_MODIFIERS Coverage ===")

    for ct in ("canny", "depth", "scribble"):
        assert ct in GUIDANCE_MODIFIERS, f"Missing guidance: {ct}"
    print(f"  ✓ All {len(GUIDANCE_MODIFIERS)} guidance modifiers present")


def test_enhance_mindmap_custom_style():
    """Test enhance_mindmap with free-form style string."""
    print("\n=== Test 12: Enhance Mindmap with Custom Style ===")

    mindmap = create_test_mindmap()
    gen = SceneGenerator(provider="mock")
    img = gen.enhance_mindmap(mindmap, style_prompt="vibrant colorful hand-drawn style")

    assert isinstance(img, Image.Image)
    print(f"  ✓ Custom style string works: {img.size}")


if __name__ == "__main__":
    print("Testing scene_gen (Phase 4)...")
    print(f"Output directory: {settings.OUTPUT_DIR}")

    try:
        test_mock_generate()
        test_enhance_mindmap_mock()
        test_canny_edge_extraction()
        test_canny_edge_extraction_bgr()
        test_canny_edge_extraction_rgba()
        test_generate_style_prompt()
        test_build_guided_prompt()
        test_generate_invalid_width_height()
        test_prepare_control_image()
        test_enhance_prompts_coverage()
        test_guidance_modifiers_coverage()
        test_enhance_mindmap_custom_style()

        print("\n" + "=" * 50)
        print("✓ All Phase 4 tests passed!")
        print(f"Outputs saved in: {settings.OUTPUT_DIR / 'scenes'}")
        print("=" * 50)

    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
