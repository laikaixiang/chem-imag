"""Phase 3: MindMapLayout 测试"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.mindmap.layout import Node, MindMapLayout
from src.config import settings


def build_sample_tree() -> Node:
    """构建3层7节点的示例树"""
    root = Node(id="r1", label="有机化学")

    alcohols = Node(id="a1", label="醇类")
    alcohols.add_child(Node(id="e1", label="乙醇", smiles="CCO"))
    alcohols.add_child(Node(id="m1", label="甲醇", smiles="CO"))

    acids = Node(id="c1", label="酸类")
    acids.add_child(Node(id="aa1", label="乙酸", smiles="CC(=O)O"))

    ketones = Node(id="k1", label="酮类")
    ketones.add_child(Node(id="ac1", label="丙酮", smiles="CC(=O)C"))

    root.add_child(alcohols)
    root.add_child(acids)
    root.add_child(ketones)

    return root


def test_node_basics():
    """测试 Node 基础功能"""
    root = build_sample_tree()

    assert root.label == "有机化学"
    assert root.depth() == 0
    assert not root.is_leaf()
    assert len(root.children) == 3

    ethanol = root.children[0].children[0]
    assert ethanol.label == "乙醇"
    assert ethanol.smiles == "CCO"
    assert ethanol.depth() == 2
    assert ethanol.is_leaf()

    print("✓ test_node_basics passed")


def test_layout_coordinates():
    """测试布局坐标计算"""
    root = build_sample_tree()
    layout = MindMapLayout(
        node_width=200, node_height=150,
        horizontal_spacing=100, vertical_spacing=60, padding=50,
    )
    total_w, total_h = layout.layout_tree(root)

    # 根节点应该在左侧
    assert root.x == layout.padding, f"root.x={root.x}, expected {layout.padding}"
    assert total_w > 0
    assert total_h > 0

    # 子节点 x > 父节点 x（深度越大 x 越大）
    for child in root.children:
        assert child.x > root.x, f"子节点 {child.label} x={child.x} <= 父节点 x={root.x}"

    # 所有节点都有有效坐标
    all_nodes = [root] + root.children
    for child in root.children:
        all_nodes.extend(child.children)

    for node in all_nodes:
        assert node.x >= 0, f"节点 {node.label} x < 0: {node.x}"
        assert node.y >= 0, f"节点 {node.label} y < 0: {node.y}"

    print(f"✓ test_layout_coordinates passed (canvas: {total_w}×{total_h})")


def test_render_output():
    """测试渲染输出"""
    root = build_sample_tree()
    layout = MindMapLayout(
        node_width=200, node_height=150,
        horizontal_spacing=100, vertical_spacing=60, padding=50,
    )

    result = layout.render(root)

    assert isinstance(result, np.ndarray)
    assert result.ndim == 3
    assert result.shape[2] == 3

    # 保存图片
    out = settings.OUTPUT_DIR / "mindmaps" / "test_layout.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    import cv2
    cv2.imwrite(str(out), result)
    print(f"✓ test_render_output passed (saved to {out})")


def test_from_json():
    """测试 from_json 构建树"""
    data = {
        "label": "有机化学",
        "children": [
            {
                "label": "醇类",
                "children": [
                    {"label": "乙醇", "smiles": "CCO"},
                    {"label": "甲醇", "smiles": "CO"},
                ],
            },
            {
                "label": "酸类",
                "children": [
                    {"label": "乙酸", "smiles": "CC(=O)O"},
                ],
            },
        ],
    }

    root = MindMapLayout.from_json(data)

    assert root.label == "有机化学"
    assert root.depth() == 0
    assert len(root.children) == 2
    assert root.children[0].label == "醇类"
    assert root.children[0].parent is root
    assert len(root.children[0].children) == 2

    ethanol = root.children[0].children[0]
    assert ethanol.label == "乙醇"
    assert ethanol.smiles == "CCO"
    assert ethanol.depth() == 2

    # 验证可以 layout + render
    layout = MindMapLayout()
    result = layout.render(root)
    assert isinstance(result, np.ndarray)

    print("✓ test_from_json passed")


def test_set_structure():
    """测试 set_structure"""
    root = build_sample_tree()
    layout = MindMapLayout()

    fake_struct = np.ones((100, 80, 4), dtype=np.uint8) * 200
    fake_struct[:, :, 3] = 255

    ethanol = root.children[0].children[0]
    layout.set_structure(ethanol, fake_struct)
    assert ethanol.structure_image is not None

    result = layout.render(root)
    assert isinstance(result, np.ndarray)

    out = settings.OUTPUT_DIR / "mindmaps" / "test_with_structure.png"
    import cv2
    cv2.imwrite(str(out), result)
    print(f"✓ test_set_structure passed (saved to {out})")


if __name__ == "__main__":
    test_node_basics()
    test_layout_coordinates()
    test_render_output()
    test_from_json()
    test_set_structure()
    print("\nAll Phase 3 tests passed!")
