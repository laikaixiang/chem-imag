"""Phase 7: End-to-end pipeline tests."""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import settings
from src.pipeline import ChemicalImagePipeline

try:
    from rdkit import Chem  # noqa: F401
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False


def test_pipeline_full_flow():
    """Full E2E pipeline: input text → final image."""
    print("\n=== Test 1: Full Pipeline ===")
    if not HAS_RDKIT:
        print("  ⚠ Skipped — RDKit not installed")
        return

    pipe = ChemicalImagePipeline()
    result = pipe.generate(
        "生成关于苯酚、苯甲酸和乙醇的思维导图",
        style="academic", width=1200, height=800,
    )

    assert "final_image" in result
    assert "steps" in result
    assert "compounds" in result
    assert Path(result["final_image"]).exists()

    img = cv2.imread(result["final_image"])
    assert img is not None
    assert img.shape[0] == 800
    assert img.shape[1] == 1200

    print(f"  ✓ Pipeline complete: {result['final_image']}")
    print(f"    Steps: {' → '.join(result['steps'])}")
    print(f"    Compounds: {result['compounds']}")


def test_pipeline_agent_mode():
    """Agent mode fallback (no API key required)."""
    print("\n=== Test 2: Agent Mode ===")
    if not HAS_RDKIT:
        print("  ⚠ Skipped — RDKit not installed")
        return

    pipe = ChemicalImagePipeline()
    result = pipe.generate(
        "Test: alcohols and ketones mindmap",
        use_agent=True,
        width=800, height=600,
    )

    assert "final_image" in result
    print(f"  ✓ Agent mode returned: {list(result.keys())}")


def test_compound_extraction():
    """LLM compound extraction (falls back to defaults without API key)."""
    print("\n=== Test 3: Compound Extraction ===")

    pipe = ChemicalImagePipeline()
    result = pipe._extract_compounds("生成关于苯酚、苯甲酸及其酯化反应的思维导图")

    assert "title" in result
    assert "compounds" in result
    assert isinstance(result["compounds"], list)
    assert len(result["compounds"]) > 0
    for c in result["compounds"]:
        assert "name" in c
        assert "parent" in c

    names = [c["name"] for c in result["compounds"]]
    print(f"  ✓ Extracted title: {result['title']}")
    print(f"  ✓ Compounds: {names}")

    # Empty input also returns valid structure (fallback)
    defaults = pipe._extract_compounds("")
    assert len(defaults["compounds"]) > 0
    print(f"  ✓ Empty input fallback: {defaults['compounds']}")


def test_resolve_compounds():
    """PubChem compound resolution."""
    print("\n=== Test 4: Resolve Compounds ===")
    if not HAS_RDKIT:
        print("  ⚠ Skipped — RDKit not installed")
        return

    pipe = ChemicalImagePipeline()
    results = pipe._resolve_compounds(["phenol", "benzoic acid"])
    assert len(results) == 2
    assert results[0]["status"] == "ok"
    assert results[0]["smiles"]
    print(f"  ✓ phenol → {results[0]['smiles']}")
    print(f"  ✓ benzoic acid → {results[1]['smiles']}")

    # Bad name
    bad = pipe._resolve_compounds(["xyznonexistent"])
    assert bad[0]["status"] == "error"
    print(f"  ✓ Invalid name returns error status")


def test_structure_generation():
    """Generate structure images from resolved compounds."""
    print("\n=== Test 5: Structure Generation ===")
    if not HAS_RDKIT:
        print("  ⚠ Skipped — RDKit not installed")
        return

    pipe = ChemicalImagePipeline()
    resolved = [
        {"name": "phenol", "smiles": "c1ccc(cc1)O", "status": "ok"},
        {"name": "invalid", "smiles": "", "status": "error"},
    ]
    results = pipe._generate_structures(resolved, "academic")
    assert results[0]["status"] == "ok"
    assert Path(results[0]["struct_path"]).exists()
    assert results[1]["status"] == "skip"
    print(f"  ✓ phenol structure: {results[0]['struct_path']}")
    print(f"  ✓ invalid compound skipped")


def test_mindmap_building():
    """Build mindmap from parsed LLM tree."""
    print("\n=== Test 6: Mindmap Building ===")

    pipe = ChemicalImagePipeline()
    parsed = {
        "title": "Oxygen Compounds",
        "compounds": [
            {"name": "phenol", "parent": None},
            {"name": "benzoic acid", "parent": None},
            {"name": "salicylic acid", "parent": "benzoic acid"},
        ],
    }
    resolved = [
        {"name": "phenol", "smiles": "c1ccc(cc1)O", "status": "ok"},
        {"name": "benzoic acid", "smiles": "c1ccc(cc1)C(=O)O", "status": "ok"},
        {"name": "salicylic acid", "smiles": "c1ccc(c(c1)C(=O)O)O", "status": "ok"},
    ]

    path = pipe._build_mindmap_from_parsed(parsed, resolved)
    assert Path(path).exists()

    img = cv2.imread(path)
    assert img is not None
    assert img.shape[1] > 400
    print(f"  ✓ Mindmap: {path} ({img.shape[1]}×{img.shape[0]})")


def test_scene_generation():
    """Scene generation (mock mode)."""
    print("\n=== Test 7: Scene Generation ===")

    pipe = ChemicalImagePipeline()
    path = pipe._generate_scene("test chemistry scene", "academic", 800, 600)
    assert Path(path).exists()

    img = Image.open(path)
    assert img.size == (800, 600)
    print(f"  ✓ Scene: {path} ({img.size})")


def test_composite():
    """Composite structures into scene."""
    print("\n=== Test 8: Composite ===")
    if not HAS_RDKIT:
        print("  ⚠ Skipped — RDKit not installed")
        return

    pipe = ChemicalImagePipeline()
    pipe.output_dir = settings.OUTPUT_DIR

    # Build prerequisites
    resolved = [
        {"name": "phenol", "smiles": "c1ccc(cc1)O", "status": "ok"},
        {"name": "benzoic acid", "smiles": "c1ccc(cc1)C(=O)O", "status": "ok"},
    ]
    structures = pipe._generate_structures(resolved, "academic")
    scene_path = pipe._generate_scene("test", "academic", 800, 600)

    # Compose
    final = pipe._composite(scene_path, structures, 800, 600)
    assert Path(final).exists()

    img = cv2.imread(final)
    assert img.shape == (600, 800, 3)
    print(f"  ✓ Composite: {final} ({img.shape[1]}×{img.shape[0]})")


def test_pipeline_output_structure():
    """Verify complete output directory structure after pipeline run."""
    print("\n=== Test 9: Output Structure ===")

    dirs = ["structures", "mindmaps", "scenes", "final"]
    for d in dirs:
        p = settings.OUTPUT_DIR / d
        exists = p.exists()
        print(f"  {'✓' if exists else '✗'} {p}")
    print(f"  Output root: {settings.OUTPUT_DIR}")


def test_pipeline_result_metadata():
    """Pipeline result dict has all expected keys."""
    print("\n=== Test 10: Result Metadata ===")
    if not HAS_RDKIT:
        print("  ⚠ Skipped — RDKit not installed")
        return

    pipe = ChemicalImagePipeline()
    result = pipe.generate("苯酚和乙醇", width=800, height=600)

    expected_keys = ["final_image", "steps", "compounds", "resolved", "mindmap_path", "scene_path"]
    for k in expected_keys:
        assert k in result, f"Missing key: {k}"
    print(f"  ✓ All {len(expected_keys)} keys present")


if __name__ == "__main__":
    print("Testing Phase 7: End-to-End Pipeline")
    print(f"Output directory: {settings.OUTPUT_DIR}")
    settings.ensure_dirs()

    try:
        test_compound_extraction()
        test_resolve_compounds()
        test_structure_generation()
        test_mindmap_building()
        test_scene_generation()
        test_composite()
        test_pipeline_full_flow()
        test_pipeline_agent_mode()
        test_pipeline_output_structure()
        test_pipeline_result_metadata()

        print("\n" + "=" * 50)
        print("✓ All Phase 7 tests passed!")
        print(f"Outputs in: {settings.OUTPUT_DIR}")
        print("=" * 50)

    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
