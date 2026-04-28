"""Tests for Phase 5: Agent orchestration system."""

import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agent.tools import Tool, ToolRegistry, register_all_tools
from src.agent.orchestrator import AgentOrchestrator
from src.agent.prompts import SYSTEM_PROMPT, build_user_prompt, build_tools_description

# RDKit requires conda install; gracefully skip tests that depend on it
try:
    from rdkit import Chem  # noqa: F401
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False


# ── Tool definition tests ──────────────────────────────────────

def test_tool_dataclass():
    """Test Tool dataclass creation and format conversion."""
    print("\n=== Test 1: Tool Dataclass ===")

    def dummy_func(x: int) -> dict:
        return {"value": x * 2}

    tool = Tool(
        name="dummy_tool",
        description="A test tool that doubles a number",
        parameters={
            "type": "object",
            "properties": {"x": {"type": "integer"}},
            "required": ["x"],
        },
        func=dummy_func,
    )

    assert tool.name == "dummy_tool"
    assert tool(x=5) == {"value": 10}

    # Anthropic format
    anthro = tool.to_anthropic_tool()
    assert anthro["name"] == "dummy_tool"
    assert "input_schema" in anthro

    # OpenAI format
    openai = tool.to_openai_tool()
    assert openai["type"] == "function"
    assert openai["function"]["name"] == "dummy_tool"

    print("  ✓ Tool creation and both API formats work")


def test_tool_registry():
    """Test ToolRegistry registration and lookup."""
    print("\n=== Test 2: ToolRegistry ===")

    registry = ToolRegistry()
    assert len(registry) == 0

    register_all_tools(registry)
    assert len(registry) == 6

    expected_tools = [
        "resolve_compound",
        "generate_structure",
        "build_mindmap",
        "generate_scene",
        "detect_surface",
        "composite_final",
    ]
    for name in expected_tools:
        assert name in registry, f"Missing tool: {name}"
        tool = registry.get(name)
        assert tool.name == name

    print(f"  ✓ All {len(registry)} tools registered")

    # Test error on unknown tool
    try:
        registry.get("nonexistent_tool")
        assert False, "Should have raised KeyError"
    except KeyError:
        pass
    print("  ✓ Unknown tool raises KeyError")

    # Test tools_description
    desc = registry.tools_description()
    assert "resolve_compound" in desc
    print("  ✓ tools_description() works")


def test_anthropic_format():
    """Test Anthropic tool schema format."""
    print("\n=== Test 3: Anthropic Format ===")

    registry = ToolRegistry()
    register_all_tools(registry)

    tools = registry.to_anthropic_tools()
    assert len(tools) == 6

    for tool_def in tools:
        assert "name" in tool_def
        assert "description" in tool_def
        assert "input_schema" in tool_def
        schema = tool_def["input_schema"]
        assert schema["type"] == "object"
        assert "properties" in schema

    print(f"  ✓ All {len(tools)} tools have valid Anthropic schema")


def test_openai_format():
    """Test OpenAI tool schema format."""
    print("\n=== Test 4: OpenAI Format ===")

    registry = ToolRegistry()
    register_all_tools(registry)

    tools = registry.to_openai_tools()
    assert len(tools) == 6

    for tool_def in tools:
        assert tool_def["type"] == "function"
        fn = tool_def["function"]
        assert "name" in fn
        assert "parameters" in fn

    print(f"  ✓ All {len(tools)} tools have valid OpenAI schema")


# ── Tool implementation tests ──────────────────────────────────

def test_resolve_compound_tool():
    """Test resolve_compound tool execution."""
    print("\n=== Test 5: resolve_compound ===")
    if not HAS_RDKIT:
        print("  ⚠ Skipped — RDKit not installed")
        return

    registry = ToolRegistry()
    register_all_tools(registry)

    tool = registry.get("resolve_compound")
    result = tool(query="c1ccccc1O")
    assert result["status"] == "ok"
    assert "smiles" in result
    print(f"  ✓ SMILES resolved: {result['smiles']}")


def test_generate_structure_tool():
    """Test generate_structure tool execution."""
    print("\n=== Test 6: generate_structure ===")
    if not HAS_RDKIT:
        print("  ⚠ Skipped — RDKit not installed")
        return

    registry = ToolRegistry()
    register_all_tools(registry)

    tool = registry.get("generate_structure")
    result = tool(smiles="c1ccccc1O", style="minimal", width=200, height=200)
    assert result["status"] == "ok"
    assert Path(result["path"]).exists()
    print(f"  ✓ Structure generated: {result['path']}")


def test_build_mindmap_tool():
    """Test build_mindmap tool execution."""
    print("\n=== Test 7: build_mindmap ===")

    registry = ToolRegistry()
    register_all_tools(registry)

    tree = {
        "label": "Oxygen Compounds",
        "smiles": "",
        "children": [
            {"label": "Alcohols", "smiles": "CCO", "children": []},
            {"label": "Ketones", "smiles": "CC(=O)C", "children": []},
        ],
    }

    output = "outputs/test_mindmap_agent.png"
    tool = registry.get("build_mindmap")
    result = tool(tree_json=json.dumps(tree), output_path=output)
    assert result["status"] == "ok"
    assert Path(result["path"]).exists()
    assert result["root_label"] == "Oxygen Compounds"
    print(f"  ✓ Mindmap built: {result['path']} ({result['width']}x{result['height']})")


def test_generate_scene_tool():
    """Test generate_scene tool execution (mock mode)."""
    print("\n=== Test 8: generate_scene ===")

    registry = ToolRegistry()
    register_all_tools(registry)

    output = "outputs/test_scene_agent.png"
    tool = registry.get("generate_scene")
    result = tool(prompt="clean chemistry lab bench", style="academic", output_path=output)
    assert result["status"] == "ok"
    assert Path(result["path"]).exists()
    print(f"  ✓ Scene generated: {result['path']}")


def test_detect_surface_tool():
    """Test detect_surface tool execution."""
    print("\n=== Test 9: detect_surface ===")

    registry = ToolRegistry()
    register_all_tools(registry)

    # Create a dummy scene for detection
    dummy_scene = "outputs/test_dummy_scene.png"
    img = 255 * np.ones((600, 800, 3), dtype=np.uint8)
    cv2.imwrite(dummy_scene, img)

    tool = registry.get("detect_surface")
    result = tool(scene_path=dummy_scene, structure_count=3)
    assert result["status"] == "ok"
    assert len(result["positions"]) == 3
    assert result["scene_size"] == [800, 600]
    print(f"  ✓ {len(result['positions'])} positions detected")

    # Test error path
    bad = tool(scene_path="nonexistent.png")
    assert bad["status"] == "error"
    print("  ✓ Missing scene returns error")


def test_composite_final_tool():
    """Test composite_final tool execution."""
    print("\n=== Test 10: composite_final ===")
    if not HAS_RDKIT:
        print("  ⚠ Skipped — RDKit not installed")
        return

    registry = ToolRegistry()
    register_all_tools(registry)

    # Create a dummy scene
    scene_path = "outputs/test_composite_scene.png"
    scene = 255 * np.ones((400, 600, 3), dtype=np.uint8)
    cv2.imwrite(scene_path, scene)

    # Generate a structure to composite
    struct_tool = registry.get("generate_structure")
    struct_result = struct_tool(smiles="CCO", style="minimal", width=100, height=100)

    structures = [{
        "path": struct_result["path"],
        "x": 250, "y": 150, "w": 100, "h": 100,
    }]

    output = "outputs/test_composite_final_agent.png"
    tool = registry.get("composite_final")
    result = tool(
        scene_path=scene_path,
        structures_info=json.dumps(structures),
        output_path=output,
    )
    assert result["status"] == "ok"
    assert Path(result["path"]).exists()
    print(f"  ✓ Composite final: {result['path']} ({result['structure_count']} structures)")


# ── Orchestrator tests ─────────────────────────────────────────

def test_orchestrator_init():
    """Test AgentOrchestrator initialization."""
    print("\n=== Test 11: AgentOrchestrator Init ===")

    orch = AgentOrchestrator(llm_provider="claude")
    assert len(orch.registry) == 6
    assert orch.llm_provider == "claude"

    orch2 = AgentOrchestrator(llm_provider="openai")
    assert orch2.llm_provider == "openai"

    print("  ✓ AgentOrchestrator initializes with 6 tools")


def test_orchestrator_default_run():
    """Test AgentOrchestrator with default/mock pipeline."""
    print("\n=== Test 12: AgentOrchestrator Default Run ===")
    if not HAS_RDKIT:
        print("  ⚠ Skipped — RDKit not installed")
        return

    orch = AgentOrchestrator(llm_provider="default")

    result = orch.run("Test: generate a mindmap about alcohols", max_iterations=3)
    assert "workflow" in result
    assert "results" in result
    assert "iterations" in result
    assert result["iterations"] <= 3

    print(f"  ✓ Default run completed in {result['iterations']} iterations")
    print(f"    Workflow steps: {len(result['workflow'])}")


# ── Prompt tests ───────────────────────────────────────────────

def test_system_prompt():
    """Test SYSTEM_PROMPT content."""
    print("\n=== Test 13: SYSTEM_PROMPT ===")

    registry = ToolRegistry()
    register_all_tools(registry)
    desc = build_tools_description(registry.list_tools())

    filled = SYSTEM_PROMPT.format(tools_description=desc)
    assert "Chemistry Mind Map Generator" in filled
    assert "resolve_compound" in filled
    assert "composite_final" in filled
    print("  ✓ SYSTEM_PROMPT filled with all tool descriptions")


def test_build_user_prompt():
    """Test build_user_prompt function."""
    print("\n=== Test 14: build_user_prompt ===")

    prompt = build_user_prompt("生成苯酚的思维导图")
    assert "苯酚" in prompt
    assert "SMILES" in prompt
    print(f"  ✓ User prompt built ({len(prompt)} chars)")


def test_default_plan():
    """Test _default_plan heuristic compound detection."""
    print("\n=== Test 15: Default Plan ===")

    orch = AgentOrchestrator()

    plan = orch._default_plan("生成关于乙醇、乙酸和酯化反应的思维导图")
    assert "steps" in plan
    assert len(plan["steps"]) > 0
    print(f"  ✓ Default plan generated with {len(plan['steps'])} steps")

    # Verify the pipeline order (LLM provides SMILES directly → no resolve_compound needed)
    tool_names = [s["tool"] for s in plan["steps"]]
    assert "generate_structure" in tool_names
    assert "composite_final" in tool_names
    assert tool_names.index("generate_structure") < tool_names.index("composite_final")
    print("  ✓ Pipeline order is correct")


if __name__ == "__main__":
    print("Testing Agent orchestration (Phase 5)...")

    try:
        test_tool_dataclass()
        test_tool_registry()
        test_anthropic_format()
        test_openai_format()
        test_resolve_compound_tool()
        test_generate_structure_tool()
        test_build_mindmap_tool()
        test_generate_scene_tool()
        test_detect_surface_tool()
        test_composite_final_tool()
        test_orchestrator_init()
        test_orchestrator_default_run()
        test_system_prompt()
        test_build_user_prompt()
        test_default_plan()

        print("\n" + "=" * 50)
        print("✓ All Phase 5 tests passed!")
        print("=" * 50)

    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
