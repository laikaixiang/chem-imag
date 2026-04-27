"""RDKit 化学结构图生成器"""
import io
import math
from pathlib import Path
from typing import Optional

from PIL import Image, ImageDraw, ImageFont
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdMolDraw2D

from src.config import settings

STYLE_PRESETS = {
    "ACS_1996": {
        "bond_line_width": 2.5,
        "font_size": 28,
        "background_color": (1, 1, 1, 0),
        "atom_color": (0, 0, 0),
        "bond_color": (0, 0, 0),
    },
    "dark_mode": {
        "bond_line_width": 3.0,
        "font_size": 32,
        "background_color": (0.1, 0.1, 0.12, 0),
        "atom_color": (0.9, 0.9, 0.9),
        "bond_color": (0.7, 0.8, 1.0),
    },
    "color_on_white": {
        "bond_line_width": 2.0,
        "font_size": 26,
        "background_color": (1, 1, 1, 1),
        "atom_color": (0.2, 0.2, 0.6),
        "bond_color": (0.3, 0.3, 0.3),
    },
    "minimal": {
        "bond_line_width": 1.5,
        "font_size": 22,
        "background_color": (1, 1, 1, 0),
        "atom_color": (0.15, 0.15, 0.15),
        "bond_color": (0.15, 0.15, 0.15),
    },
}

# TODO: 扩展更多风格预设（如 Nature/Science 期刊风格、手绘风格、高对比度等）
# TODO: 支持球棍模型（ball-and-stick）3D 渲染，可能需要集成 Py3Dmol 或 RDKit 的 3D 绘制功能


class StructureGenerator:
    def __init__(
        self,
        default_width: int = 1200,
        default_height: int = 800,
        default_style: str = "ACS_1996",
        output_dir: Optional[Path] = None,
    ):
        self.default_width = default_width
        self.default_height = default_height
        self.default_style = default_style
        self.style = STYLE_PRESETS[default_style]
        self.output_dir = output_dir or (settings.OUTPUT_DIR / "structures")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _apply_style(self, drawer: rdMolDraw2D.MolDraw2DCairo, style: dict):
        opts = drawer.drawOptions()
        bg = style["background_color"]
        opts.setBackgroundColour(bg)
        opts.bondLineWidth = style["bond_line_width"]
        # 新版 RDKit 使用 setSymbolColour 设置原子颜色
        opts.setSymbolColour(style["atom_color"])

    def _render_mol(self, mol, width: int, height: int, style: dict) -> Image.Image:
        AllChem.Compute2DCoords(mol)
        drawer = rdMolDraw2D.MolDraw2DCairo(width, height)
        self._apply_style(drawer, style)
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        png_data = drawer.GetDrawingText()
        return Image.open(io.BytesIO(png_data)).convert("RGBA")

    def generate_from_smiles(
        self,
        smiles: str,
        output_path: Optional[Path] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        style: Optional[str] = None,
        **kwargs,
    ) -> tuple[Path, Image.Image]:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")

        active_style = STYLE_PRESETS[style] if style else self.style
        w = width or self.default_width
        h = height or self.default_height
        img = self._render_mol(mol, w, h, active_style)

        if output_path is None:
            safe_name = smiles.replace("/", "_").replace("\\", "_")[:30]
            output_path = self.output_dir / f"{safe_name}.png"
        else:
            output_path = Path(output_path)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(str(output_path), "PNG")
        return output_path, img

    def resolve(self, name: str) -> str:
        import pubchempy as pcp

        results = pcp.get_compounds(name, "name")
        if not results:
            raise ValueError(f"化合物 '{name}' 未在 PubChem 中找到")
        return results[0].connectivity_smiles

    def generate_from_name(
        self,
        name: str,
        output_path: Optional[Path] = None,
        **kwargs,
    ) -> tuple[Path, Image.Image]:
        smiles = self.resolve(name)
        if output_path is None:
            output_path = self.output_dir / f"{name.lower().replace(' ', '_')}.png"
        return self.generate_from_smiles(smiles, output_path=output_path, **kwargs)

    def generate_multiple(
        self,
        compounds: list[dict],
        layout: str = "horizontal",
        spacing: int = 50,
        output_path: Optional[Path] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        **kwargs,
    ) -> tuple[Path, Image.Image]:
        w = width or self.default_width
        h = height or self.default_height
        label_h = 40

        images: list[Image.Image] = []
        labels: list[str] = []
        for comp in compounds:
            smiles = comp.get("smiles")
            if not smiles and comp.get("name"):
                smiles = self.resolve(comp["name"])
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                raise ValueError(f"Invalid SMILES: {smiles}")
            images.append(self._render_mol(mol, w, h, self.style))
            labels.append(comp.get("label", ""))

        n = len(images)
        if layout == "horizontal":
            canvas_w = n * w + (n - 1) * spacing
            canvas_h = h + label_h
            positions = [(i * (w + spacing), label_h) for i in range(n)]
        elif layout == "vertical":
            canvas_w = w
            canvas_h = n * (h + label_h) + (n - 1) * spacing
            positions = [(0, i * (h + label_h + spacing) + label_h) for i in range(n)]
        elif layout == "grid":
            cols = math.ceil(math.sqrt(n))
            rows = math.ceil(n / cols)
            canvas_w = cols * w + (cols - 1) * spacing
            canvas_h = rows * (h + label_h) + (rows - 1) * spacing
            positions = []
            for i in range(n):
                r, c = divmod(i, cols)
                x = c * (w + spacing)
                y = r * (h + label_h + spacing) + label_h
                positions.append((x, y))
        else:
            raise ValueError(f"Unknown layout: {layout}")

        canvas = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(canvas)

        try:
            font = ImageFont.truetype("arial.ttf", 24)
        except OSError:
            font = ImageFont.load_default()

        for img, label, (x, y) in zip(images, labels, positions):
            if label:
                bbox = draw.textbbox((0, 0), label, font=font)
                text_w = bbox[2] - bbox[0]
                text_x = x + (w - text_w) // 2
                draw.text((text_x, y - label_h), label, fill=(0, 0, 0, 255), font=font)
            canvas.paste(img, (x, y), img)

        if output_path is None:
            output_path = self.output_dir / f"multi_{layout}_{n}.png"
        else:
            output_path = Path(output_path)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(str(output_path), "PNG")
        return output_path, canvas
