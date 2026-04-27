"""Agent Tool system for chem-mindmap.

Defines Tool dataclass, ToolRegistry, and all tool implementation functions
that the orchestrator wires together.

Tools:
    resolve_compound  — resolve compound name/smiles via PubChem
    generate_structure — render a 2D structure image via RDKit
    build_mindmap     — layout and render a mindmap tree
    generate_scene    — generate/enhance a scene image via AI
    detect_surface    — find placement regions in a scene
    composite_final   — composite structures into the final scene
"""

import json
import logging
from dataclasses import dataclass
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


# ── Tool definition ────────────────────────────────────────────

@dataclass
class Tool:
    """A callable tool with JSON Schema parameter definition.

    Supports both Anthropic (input_schema) and OpenAI (function.parameters) formats.
    """

    name: str
    description: str
    parameters: dict  # JSON Schema
    func: Callable

    def to_anthropic_tool(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.parameters,
        }

    def to_openai_tool(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def __call__(self, **kwargs) -> Any:
        return self.func(**kwargs)


# ── Tool registry ───────────────────────────────────────────────

class ToolRegistry:
    """Registry that holds named Tools and can export schema lists.

    Usage:
        registry = ToolRegistry()
        register_all_tools(registry)
        tools_schema = registry.to_anthropic_tools()
    """

    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool):
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        if name not in self._tools:
            raise KeyError(f"Tool not found: {name}")
        return self._tools[name]

    def list_tools(self) -> list[Tool]:
        return list(self._tools.values())

    def to_anthropic_tools(self) -> list[dict]:
        return [t.to_anthropic_tool() for t in self._tools.values()]

    def to_openai_tools(self) -> list[dict]:
        return [t.to_openai_tool() for t in self._tools.values()]

    def tools_description(self) -> str:
        lines = []
        for t in self._tools.values():
            params = t.parameters.get("properties", {}).keys()
            lines.append(f"- **{t.name}**({', '.join(params)}): {t.description}")
        return "\n".join(lines)

    def __len__(self):
        return len(self._tools)

    def __contains__(self, name: str):
        return name in self._tools


# ── Tool implementations ────────────────────────────────────────

def _resolve_compound_impl(query: str) -> dict:
    """Resolve a compound name to SMILES — or validate an existing SMILES string."""
    from rdkit import Chem
    from src.structure_gen.generator import StructureGenerator

    # If the query is already a valid SMILES, return it directly
    mol = Chem.MolFromSmiles(query)
    if mol is not None:
        return {"query": query, "smiles": Chem.MolToSmiles(mol), "status": "ok"}

    # Otherwise, treat as a name and look up via PubChem
    gen = StructureGenerator()
    try:
        smiles = gen.resolve(query)
        return {"query": query, "smiles": smiles, "status": "ok"}
    except Exception as e:
        return {"query": query, "smiles": "", "status": "error", "error": str(e)}


def _generate_structure_impl(
    smiles: str,
    style: str = "ACS_1996",
    output_path: Optional[str] = None,
    width: int = 800,
    height: int = 600,
) -> dict:
    """Generate a 2D chemical structure image from SMILES."""
    from src.structure_gen.generator import StructureGenerator

    gen = StructureGenerator()
    try:
        path, _ = gen.generate_from_smiles(
            smiles, style=style, output_path=output_path, width=width, height=height,
        )
        return {"path": str(path), "smiles": smiles, "style": style, "status": "ok"}
    except Exception as e:
        return {"path": "", "smiles": smiles, "status": "error", "error": str(e)}


def _build_mindmap_impl(
    tree_json: str, output_path: str, node_width: int = 200, node_height: int = 150,
) -> dict:
    """Build a mindmap layout from a JSON tree and render to image."""
    import cv2
    from src.mindmap.layout import MindMapLayout, Node

    data = json.loads(tree_json)

    def build_node(d: dict) -> Node:
        n = Node(id=d.get("id", d["label"]), label=d["label"], smiles=d.get("smiles"))
        for child in d.get("children", []):
            n.add_child(build_node(child))
        return n

    try:
        root = build_node(data)
        layout = MindMapLayout(node_width=node_width, node_height=node_height)
        img = layout.render(root)
        cv2.imwrite(output_path, img)
        return {
            "path": output_path,
            "width": img.shape[1],
            "height": img.shape[0],
            "root_label": root.label,
            "status": "ok",
        }
    except Exception as e:
        return {"path": "", "status": "error", "error": str(e)}


def _generate_scene_impl(
    prompt: str, output_path: str, style: str = "academic", mindmap_path: Optional[str] = None,
) -> dict:
    """Generate an AI scene, optionally enhanced from a mindmap skeleton."""
    import cv2
    from src.scene_gen.generator import SceneGenerator

    try:
        if mindmap_path:
            mindmap = cv2.imread(mindmap_path)
            if mindmap is None:
                return {"path": "", "status": "error", "error": f"Cannot read: {mindmap_path}"}
            gen = SceneGenerator(provider="mock")
            img = gen.enhance_mindmap(mindmap, style_prompt=style)
        else:
            gen = SceneGenerator(provider="mock")
            img = gen.generate(prompt)

        img.save(output_path)
        return {"path": output_path, "style": style, "status": "ok"}
    except Exception as e:
        return {"path": "", "status": "error", "error": str(e)}


def _detect_surface_impl(scene_path: str, structure_count: int = 1) -> dict:
    """Detect placement regions in a scene for compositing structures.

    Returns evenly-spaced default positions. Real surface detection (SAM /
    Grounding DINO) is an optional Phase-4 extension.
    """
    import cv2

    img = cv2.imread(scene_path)
    if img is None:
        return {"positions": [], "status": "error", "error": f"Cannot read: {scene_path}"}

    h, w = img.shape[:2]
    positions = []
    margin = 100
    spacing = (w - 2 * margin) // max(structure_count, 1)

    for i in range(structure_count):
        positions.append({
            "x": margin + i * spacing + spacing // 4,
            "y": h // 2 - 75,
            "w": spacing // 2,
            "h": 150,
        })

    return {"positions": positions, "scene_size": [w, h], "status": "ok"}


def _composite_final_impl(scene_path: str, structures_info: str, output_path: str) -> dict:
    """Composite structure images into a scene image with lighting match."""
    import cv2
    from src.compositor.basic import load_image, resize_with_alpha
    from src.compositor.lighting import match_and_blend

    try:
        bg = load_image(scene_path)
        structures = json.loads(structures_info)

        for s in structures:
            overlay = load_image(s["path"])
            x, y, w, h = s["x"], s["y"], s["w"], s["h"]
            overlay = resize_with_alpha(overlay, w, h)

            roi_h = min(h, bg.shape[0] - y)
            roi_w = min(w, bg.shape[1] - x)
            if roi_h <= 0 or roi_w <= 0:
                continue

            overlay_crop = overlay[:roi_h, :roi_w]
            bg = match_and_blend(
                bg, overlay_crop, (x, y),
                color_match=True, texture_blend=0.08, shadow=True,
            )

        cv2.imwrite(output_path, bg)
        return {"path": output_path, "structure_count": len(structures), "status": "ok"}
    except Exception as e:
        return {"path": "", "status": "error", "error": str(e)}


# ── Registration ────────────────────────────────────────────────

def register_all_tools(registry: ToolRegistry):
    """Register all chem-mindmap tools into the given registry."""

    registry.register(Tool(
        name="resolve_compound",
        description="通过名称或 SMILES 解析化合物，返回标准 SMILES 字符串",
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "化合物名称（如'苯酚'、'乙醇'）或 SMILES 字符串",
                },
            },
            "required": ["query"],
        },
        func=_resolve_compound_impl,
    ))

    registry.register(Tool(
        name="generate_structure",
        description="为化合物生成精确的 2D 化学结构图（RDKit 渲染）",
        parameters={
            "type": "object",
            "properties": {
                "smiles": {"type": "string", "description": "化合物的 SMILES 字符串"},
                "style": {
                    "type": "string",
                    "enum": ["ACS_1996", "dark_mode", "color_on_white", "minimal"],
                    "description": "结构图渲染风格",
                },
                "output_path": {"type": "string", "description": "输出 PNG 路径（可选）"},
                "width": {"type": "integer", "description": "图片宽度（默认 800）"},
                "height": {"type": "integer", "description": "图片高度（默认 600）"},
            },
            "required": ["smiles"],
        },
        func=_generate_structure_impl,
    ))

    registry.register(Tool(
        name="build_mindmap",
        description="根据 JSON 树结构构建思维导图布局并渲染为图像",
        parameters={
            "type": "object",
            "properties": {
                "tree_json": {
                    "type": "string",
                    "description": "思维导图的 JSON 树，每个节点含 label、smiles、children",
                },
                "output_path": {"type": "string", "description": "输出 PNG 路径"},
                "node_width": {"type": "integer", "description": "节点宽度（默认 200）"},
                "node_height": {"type": "integer", "description": "节点高度（默认 150）"},
            },
            "required": ["tree_json", "output_path"],
        },
        func=_build_mindmap_impl,
    ))

    registry.register(Tool(
        name="generate_scene",
        description="生成或美化思维导图场景图（AI 图像生成）",
        parameters={
            "type": "object",
            "properties": {
                "prompt": {"type": "string", "description": "场景描述 prompt"},
                "style": {
                    "type": "string",
                    "enum": ["academic", "modern", "minimal"],
                    "description": "场景风格",
                },
                "output_path": {"type": "string", "description": "输出 PNG 路径"},
                "mindmap_path": {
                    "type": "string",
                    "description": "思维导图骨架图路径（提供时走 ControlNet 美化）",
                },
            },
            "required": ["prompt", "output_path"],
        },
        func=_generate_scene_impl,
    ))

    registry.register(Tool(
        name="detect_surface",
        description="检测场景图中适合放置结构图的区域，返回位置坐标列表",
        parameters={
            "type": "object",
            "properties": {
                "scene_path": {"type": "string", "description": "场景图路径"},
                "structure_count": {
                    "type": "integer",
                    "description": "需要放置的结构图数量（默认 1）",
                },
            },
            "required": ["scene_path"],
        },
        func=_detect_surface_impl,
    ))

    registry.register(Tool(
        name="composite_final",
        description="将化学结构图合成到场景图中，含光照匹配和阴影，输出最终图像",
        parameters={
            "type": "object",
            "properties": {
                "scene_path": {"type": "string", "description": "场景图路径"},
                "structures_info": {
                    "type": "string",
                    "description": "结构图信息 JSON，每项含 path/x/y/w/h",
                },
                "output_path": {"type": "string", "description": "输出 PNG 路径"},
            },
            "required": ["scene_path", "structures_info", "output_path"],
        },
        func=_composite_final_impl,
    ))

    logger.info("Registered %d tools", len(registry))
