"""Tests for Phase 8: Prompt Optimizer."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.prompt_optimizer import optimize_prompt
from src.config import api_config

try:
    from rdkit import Chem  # noqa: F401
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False


def test_optimizer_no_api_key():
    """Optimizer returns raw prompt when no API key."""
    print("\n=== Test 1: No API Key Fallback ===")
    result = optimize_prompt("test prompt", api_key="", api_url="http://localhost/v1")
    assert result == "test prompt"
    print("  ✓ Returns raw prompt when no key")


def test_optimizer_no_url():
    """Optimizer returns raw prompt when no URL."""
    print("\n=== Test 2: No URL Fallback ===")
    result = optimize_prompt("test prompt", api_key="sk-fake", api_url="")
    assert result == "test prompt"
    print("  ✓ Returns raw prompt when no URL")


def test_optimizer_with_real_api():
    """Test with actual API config (if configured)."""
    print("\n=== Test 3: Real API Call ===")

    if not api_config.key or not api_config.url:
        print("  ⚠ Skipped — no API key configured")
        return

    raw = "a chemistry lab with glassware on a bench"
    result = optimize_prompt(raw)

    assert isinstance(result, str)
    assert len(result) > 0
    print(f"  Raw:       {raw}")
    print(f"  Optimized: {result}")

    # The optimized result should contain the original key concepts (case-insensitive)
    assert "lab" in result.lower() or "chemistry" in result.lower()
    print("  ✓ Optimization succeeded")


def test_optimizer_pipeline_integration():
    """Pipeline with use_optimizer=True doesn't crash."""
    print("\n=== Test 4: Pipeline Integration ===")
    if not HAS_RDKIT:
        print("  ⚠ Skipped — RDKit not installed")
        return

    from src.pipeline import ChemicalImagePipeline

    pipe = ChemicalImagePipeline()
    result = pipe.generate(
        "generate a mindmap about alcohol reactions",
        width=800, height=600,
        use_optimizer=True,
    )

    assert "final_image" in result
    assert "steps" in result
    print(f"  ✓ Pipeline completed with steps: {result['steps']}")


def test_optimizer_pipeline_without_optimizer():
    """Pipeline without optimizer also works."""
    print("\n=== Test 5: Pipeline Without Optimizer ===")
    if not HAS_RDKIT:
        print("  ⚠ Skipped — RDKit not installed")
        return

    from src.pipeline import ChemicalImagePipeline

    pipe = ChemicalImagePipeline()
    result = pipe.generate(
        "generate a mindmap about ketones",
        width=800, height=600,
        use_optimizer=False,
    )

    assert "final_image" in result
    assert "optimize" not in result["steps"]
    print("  ✓ Pipeline without optimizer — 'optimize' not in steps")


if __name__ == "__main__":
    print("Testing Phase 8: Prompt Optimizer")

    try:
        test_optimizer_no_api_key()
        test_optimizer_no_url()
        test_optimizer_with_real_api()
        test_optimizer_pipeline_integration()
        test_optimizer_pipeline_without_optimizer()

        print("\n" + "=" * 50)
        print("✓ All Phase 8 tests passed!")
        print("=" * 50)

    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
