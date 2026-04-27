"""思维导图层级布局引擎 — Reingold-Tilford 树布局 + clean 风格渲染"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


@dataclass
class Node:
    """思维导图树节点"""

    id: str
    label: str
    smiles: Optional[str] = None
    children: list[Node] = field(default_factory=list)
    parent: Optional[Node] = None

    # 布局后填充
    x: int = 0
    y: int = 0
    width: int = 0
    height: int = 0

    # 结构图（由外部生成后设置）
    structure_image: Optional[np.ndarray] = None

    def add_child(self, child: Node):
        child.parent = self
        self.children.append(child)

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def depth(self) -> int:
        d = 0
        n = self
        while n.parent:
            d += 1
            n = n.parent
        return d


class MindMapLayout:
    """Reingold-Tilford 树布局 + clean 风格渲染"""

    def __init__(
        self,
        node_width: int = 200,
        node_height: int = 150,
        horizontal_spacing: int = 100,
        vertical_spacing: int = 60,
        padding: int = 50,
    ):
        self.node_width = node_width
        self.node_height = node_height
        self.horizontal_spacing = horizontal_spacing
        self.vertical_spacing = vertical_spacing
        self.padding = padding

    # ── layout ─────────────────────────────────────────────

    def layout_tree(self, root: Node) -> tuple[int, int]:
        """Reingold-Tilford 算法：递归计算所有节点坐标，返回画布 (w, h)"""
        self._contour: dict[int, float] = {}  # depth → rightmost x of level
        self._next_y: dict[int, float] = {}   # depth → next available y

        self._layout_subtree(root)
        root.width = self.node_width
        root.height = self.node_height

        max_x = max(n.x + self.node_width for n in self._all_nodes(root))
        max_y = max(n.y + self.node_height for n in self._all_nodes(root))
        return max_x + self.padding, max_y + self.padding

    def _layout_subtree(self, node: Node):
        depth = node.depth()
        col_x = depth * (self.node_width + self.horizontal_spacing) + self.padding

        node.x = col_x

        if node.is_leaf():
            # 分配下一个可用的 y 位置
            base_y = self._next_y.get(depth, self.padding)
            node.y = int(base_y)
            self._next_y[depth] = base_y + self.node_height + self.vertical_spacing
        else:
            # 递归布局子节点
            for child in node.children:
                self._layout_subtree(child)

            # 当前节点放在子节点 y 范围的中间
            children_y = [c.y + self.node_height / 2 for c in node.children]
            mid_y = sum(children_y) / len(children_y)
            node.y = int(mid_y - self.node_height / 2)

            # 确保不与左侧节点重叠
            left_boundary = self._contour.get(depth, self.padding)
            if node.y < left_boundary:
                shift = left_boundary - node.y
                self._shift_subtree(node, shift)

        # 更新当前层级的右边界
        right_edge = node.y + self.node_height + self.vertical_spacing
        self._contour[depth] = max(self._contour.get(depth, 0), right_edge)
        self._next_y[depth] = max(self._next_y.get(depth, self.padding), right_edge)

    def _shift_subtree(self, node: Node, shift: int):
        """将子树整体向下平移"""
        node.y += shift
        for child in node.children:
            self._shift_subtree(child, shift)

    # ── render ─────────────────────────────────────────────

    def render(self, root: Node) -> np.ndarray:
        """渲染思维导图为 BGR numpy array（clean 风格）"""
        total_w, total_h = self.layout_tree(root)

        # 白色画布
        canvas = np.full((total_h, total_w, 3), 255, dtype=np.uint8)

        # 先画连接线，再画节点框（线在框下方）
        self._draw_edges(canvas, root)

        for node in self._all_nodes(root):
            self._draw_node_box(canvas, node)

        return canvas

    def _draw_edges(self, canvas: np.ndarray, node: Node):
        """用正交线连接父子节点"""
        for child in node.children:
            # 父节点右边中点 → 子节点左边中点
            x1 = node.x + self.node_width
            y1 = node.y + self.node_height // 2
            x2 = child.x
            y2 = child.y + self.node_height // 2
            mid_x = (x1 + x2) // 2

            cv2.line(canvas, (x1, y1), (mid_x, y1), (120, 120, 120), 2, cv2.LINE_AA)
            cv2.line(canvas, (mid_x, y1), (mid_x, y2), (120, 120, 120), 2, cv2.LINE_AA)
            cv2.line(canvas, (mid_x, y2), (x2, y2), (120, 120, 120), 2, cv2.LINE_AA)

            self._draw_edges(canvas, child)

    def _draw_node_box(self, canvas: np.ndarray, node: Node):
        """在 PIL 中绘制单个节点框"""
        x, y, w, h = node.x, node.y, self.node_width, self.node_height
        radius = 12

        # 转 PIL 绘制圆角矩形和文字
        canvas_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(canvas_rgb)
        draw = ImageDraw.Draw(pil)

        # 圆角矩形
        draw.rounded_rectangle(
            [x, y, x + w, y + h],
            radius=radius,
            fill=(248, 248, 250),
            outline=(180, 180, 185),
            width=2,
        )

        # 文字
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except OSError:
            try:
                font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 16)
            except OSError:
                font = ImageFont.load_default()

        text = node.label
        bbox = draw.textbbox((0, 0), text, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]

        text_x = x + (w - tw) // 2
        text_y = y + h - th - 12
        draw.text((text_x, text_y), text, fill=(40, 40, 45), font=font)

        # 转回 BGR
        canvas[:] = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

        # 绘制结构图（如果有）
        if node.structure_image is not None:
            self._draw_structure(canvas, node)

    def _draw_structure(self, canvas: np.ndarray, node: Node):
        """在节点框内绘制结构图"""
        struct = node.structure_image
        h, w = struct.shape[:2]

        max_w = self.node_width - 30
        max_h = self.node_height - 45
        scale = min(max_w / w, max_h / h)
        new_w, new_h = int(w * scale), int(h * scale)

        struct_resized = cv2.resize(struct, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

        sx = node.x + (self.node_width - new_w) // 2
        sy = node.y + 10

        if struct_resized.ndim == 3 and struct_resized.shape[2] == 4:
            # RGBA → alpha 混合
            roi = canvas[sy:sy + new_h, sx:sx + new_w]
            alpha = struct_resized[:, :, 3:4].astype(np.float32) / 255.0
            rgb = struct_resized[:, :, :3].astype(np.float32)
            blended = rgb * alpha + roi.astype(np.float32) * (1 - alpha)
            canvas[sy:sy + new_h, sx:sx + new_w] = blended.astype(np.uint8)
        elif struct_resized.ndim == 3:
            roi = canvas[sy:sy + new_h, sx:sx + new_w]
            alpha = 0.9
            blended = cv2.addWeighted(struct_resized[:, :, :3], alpha, roi, 1 - alpha, 0)
            canvas[sy:sy + new_h, sx:sx + new_w] = blended
        else:
            canvas[sy:sy + new_h, sx:sx + new_w] = cv2.cvtColor(struct_resized, cv2.COLOR_GRAY2BGR)

    # ── helpers ────────────────────────────────────────────

    def set_structure(self, node: Node, structure_img: np.ndarray):
        node.structure_image = structure_img

    def _all_nodes(self, node: Node) -> list[Node]:
        nodes = [node]
        for child in node.children:
            nodes.extend(self._all_nodes(child))
        return nodes

    # ── from_json ──────────────────────────────────────────

    @staticmethod
    def from_json(data: dict, parent: Optional[Node] = None) -> Node:
        """从嵌套 dict 递归构建树"""
        import uuid

        node = Node(
            id=data.get("id", str(uuid.uuid4())[:8]),
            label=data["label"],
            smiles=data.get("smiles"),
        )
        node.parent = parent
        node.width = 200
        node.height = 150

        for child_data in data.get("children", []):
            child = MindMapLayout.from_json(child_data, parent=node)
            node.children.append(child)

        return node
