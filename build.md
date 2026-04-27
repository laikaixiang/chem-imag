# 🧪 Chemical Mind Map Generator — 完整构建指南

> **目标**：构建一个 GUI 应用，用户输入"生成有机化学里关于【xxx】的思维导图"，系统自动输出包含精确化学结构的高质量图像，可直接用于学术论文。
>
> **架构**：Agent 编排层 → 精确结构生成（RDKit）→ AI 场景生成 → 智能合成
>
> **使用方式**：每个 Phase 在**独立对话**中执行。Agent 读取本文件对应章节 + 上一 Phase 输出文件即可恢复上下文。

---

## 目录

| Phase | 内容           | 预计对话数 | 输出产物                   |
| ----- | -------------- | ---------- | -------------------------- |
| 0     | 环境搭建       | 1          | 项目骨架 + 依赖            |
| 1     | 结构图生成器   | 1          | `StructureGenerator` 类    |
| 2     | 合成引擎       | 2-3        | `Compositor` 完整实现      |
| 3     | 思维导图布局器 | 1          | `MindMapLayout` 类         |
| 4     | AI 场景生成    | 1          | `SceneGenerator` 接口      |
| 5     | Agent 编排层   | 1          | `Orchestrator` + Tool 系统 |
| 6     | GUI 界面       | 1          | Gradio/Streamlit 应用      |
| 7     | 端到端联调     | 1          | 完整流水线                 |

**总计：约 8-10 次对话，每次 150k-200k tokens 以内。**

---

## 层级结构（解决记忆限制）

```
Level 0: 基础设施
├── Phase 0: 环境搭建        ← 独立对话，产生项目骨架
├── Phase 1: 结构图生成器     ← 独立对话，产生精确结构生成模块
└── Phase 2: 合成引擎         ← 独立对话（可拆2次），产生合成模块

Level 1: 核心功能
├── Phase 3: 思维导图布局     ← 独立对话，产生布局引擎
└── Phase 4: AI 场景生成      ← 独立对话，产生场景生成接口

Level 2: 编排与界面
├── Phase 5: Agent 编排层     ← 独立对话，产生编排逻辑
├── Phase 6: GUI 界面         ← 独立对话，产生用户界面
└── Phase 7: 端到端联调       ← 独立对话，整合测试
```

**每个 Level 的模块通过 `import` 接口耦合，Agent 只需读取模块接口签名即可使用，不需要记忆实现细节。**

---

## Phase 0: 环境搭建

### 📋 Agent 指令（复制到新对话）

```
你是一个资深 Python 工程师。请帮我搭建一个名为 "chem-mindmap" 的项目骨架。

项目目标是：构建一个 GUI 应用，用户输入有机化学思维导图的想法，系统自动生成包含精确化学结构的高质量论文级图像。

请按以下步骤执行：

### 步骤 1：创建目录结构
```
chem-mindmap/
├── pyproject.toml
├── .env.example
├── .gitignore
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── structure_gen/
│   │   ├── __init__.py
│   │   └── generator.py        ← 待填充（Phase 1）
│   ├── compositor/
│   │   ├── __init__.py
│   │   ├── basic.py            ← 基础合成
│   │   ├── lighting.py         ← 光照匹配
│   │   └── pyramid.py          ← 金字塔融合
│   ├── mindmap/
│   │   ├── __init__.py
│   │   └── layout.py           ← 思维导图布局器（Phase 3）
│   ├── scene_gen/
│   │   ├── __init__.py
│   │   └── generator.py        ← AI 场景生成（Phase 4）
│   ├── agent/
│   │   ├── __init__.py
│   │   ├── tools.py            ← Tool 系统（Phase 5）
│   │   └── orchestrator.py     ← 编排层（Phase 5）
│   └── gui/
│       ├── __init__.py
│       └── app.py              ← GUI 界面（Phase 6）
├── tests/
│   ├── __init__.py
│   ├── test_structure_gen.py
│   ├── test_compositor.py
│   ├── test_mindmap.py
│   └── test_pipeline.py
├── outputs/
│   ├── structures/
│   ├── scenes/
│   ├── mindmaps/
│   └── final/
└── assets/
    └── templates/
```

### 步骤 2：创建 pyproject.toml

```toml
[project]
name = "chem-mindmap"
version = "0.1.0"
description = "AI-powered organic chemistry mind map generator for academic papers"
requires-python = ">=3.10"
dependencies = [
    "Pillow>=10.0",
    "numpy>=1.24",
    "opencv-python-headless>=4.8",
    "pubchempy>=1.0.4",
    "httpx>=0.25",
    "python-dotenv>=1.0",
    "pydantic>=2.0",
    "gradio>=4.0",
    "scipy>=1.11",
]

[project.optional-dependencies]
detection = [
    "torch>=2.0",
    "torchvision>=0.15",
    "transformers>=4.36",
    "segment-anything>=1.0",
]
rwkv = [
    "rwkv>=0.7",
]

[tool.setuptools.packages.find]
where = ["src"]

[build-system]
requires = ["setuptools>=68.0"]
build-backend = "setuptools.backends._legacy:_Backend"
```

### 步骤 3：创建 .env.example
```
# 图像生成 API
AI_IMAGE_PROVIDER=sd_webui   # sd_webui | openai | replicate
SD_WEBUI_URL=http://127.0.0.1:7860
OPENAI_API_KEY=
REPLICATE_API_TOKEN=

# LLM API（用于 Agent 解析）
LLM_PROVIDER=claude  # claude | openai | local
ANTHROPIC_API_KEY=
OPENAI_API_KEY=

# 输出配置
OUTPUT_DIR=./outputs
TEMP_DIR=./outputs/temp

# 硬件
DEVICE=cuda  # cuda | mps | cpu
```

### 步骤 4：创建 src/config.py

```python
"""全局配置"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# 项目根目录
ROOT = Path(__file__).resolve().parent.parent

class Settings:
    OUTPUT_DIR: Path = Path(os.getenv("OUTPUT_DIR", ROOT / "outputs"))
    TEMP_DIR: Path = Path(os.getenv("TEMP_DIR", ROOT / "outputs" / "temp"))
  
    # AI 图像生成
    AI_IMAGE_PROVIDER: str = os.getenv("AI_IMAGE_PROVIDER", "sd_webui")
    SD_WEBUI_URL: str = os.getenv("SD_WEBUI_URL", "http://127.0.0.1:7860")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    REPLICATE_API_TOKEN: str = os.getenv("REPLICATE_API_TOKEN", "")
  
    # LLM
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "claude")
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
  
    # 硬件
    DEVICE: str = os.getenv("DEVICE", "cpu")
  
    def ensure_dirs(self):
        for d in [self.OUTPUT_DIR, self.TEMP_DIR, 
                  self.OUTPUT_DIR / "structures",
                  self.OUTPUT_DIR / "scenes",
                  self.OUTPUT_DIR / "mindmaps",
                  self.OUTPUT_DIR / "final"]:
            d.mkdir(parents=True, exist_ok=True)
        return self

    def get_device(self) -> str:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

settings = Settings().ensure_dirs()
```

### 步骤 5：创建 .gitignore
```
__pycache__/
*.pyc
.env
.env.local
outputs/
*.egg-info/
dist/
build/
```

### 步骤 6：安装并验证
请执行：
```bash
cd chem-mindmap
pip install -e .
python -c "
from src.config import settings
print(f'✅ 配置加载成功')
print(f'   OUTPUT_DIR: {settings.OUTPUT_DIR}')
print(f'   设备: {settings.get_device()}')
"
```

完成后告诉我，并列出当前项目目录的结构。
```

---

## Phase 1: 结构图生成器

### 上下文恢复
> Phase 0 已完成。项目根目录 `chem-mindmap/`，`src/config.py` 可用。
> 本 Phase 目标：实现 `src/structure_gen/generator.py`，用 RDKit 生成精确化学结构图。

### 📋 Agent 指令

```
我在 `chem-mindmap/` 项目下，Phase 0 已完成。现在要实现 Phase 1：结构图生成器。

项目根目录: /path/to/chem-mindmap  （根据你的实际路径替换）
src/config.py 中定义了 settings，可以通过 settings.OUTPUT_DIR 获取输出目录。

请创建 src/structure_gen/generator.py，实现以下类：

## StructureGenerator 类

### 构造参数
```python
def __init__(
    self,
    default_width: int = 1200,
    default_height: int = 800,
    default_style: str = "ACS_1996",
    output_dir: Optional[Path] = None,
):
```
- default_style 选项: "ACS_1996", "dark_mode", "color_on_white", "minimal"
- output_dir 默认为 settings.OUTPUT_DIR / "structures"

### 核心方法

#### 1. generate_from_smiles(smiles, output_path=None, **kwargs) -> tuple[Path, Image.Image]
- 输入 SMILES 字符串
- 使用 RDKit 生成 2D 结构图
- 输出透明背景 PNG
- 返回 (文件路径, PIL Image)

**实现细节**：
```python
from rdkit import Chem
from rdkit.Chem import Draw, AllChem
from rdkit.Chem.Draw import rdMolDraw2D
import io
from PIL import Image

mol = Chem.MolFromSmiles(smiles)
if mol is None:
    raise ValueError(f"Invalid SMILES: {smiles}")
AllChem.Compute2DCoords(mol)

# 使用 Cairo 绘制器（支持透明）
drawer = rdMolDraw2D.MolDraw2DCairo(width, height)
drawer.SetBackgroundColour(self.style["background_color"][:3])

# 配置绘制选项
opts = drawer.drawOptions()
opts.bondLineWidth = self.style["bond_line_width"]

drawer.DrawMolecule(mol)
drawer.FinishDrawing()

# 转为 PIL Image
png_data = drawer.GetDrawingText()
img = Image.open(io.BytesIO(png_data))
```

#### 2. generate_from_name(name, output_path=None, **kwargs) -> tuple[Path, Image.Image]
- 通过 PubChem 查询化合物名称
- 自动获取 SMILES 后调用 generate_from_smiles
- 查询失败时抛出 ValueError

```python
import pubchempy as pcp
results = pcp.get_compounds(name, 'name')
if not results:
    raise ValueError(f"化合物 '{name}' 未在 PubChem 中找到")
smiles = results[0].canonical_smiles
```

#### 3. generate_multiple(compounds, layout="horizontal", spacing=50, **kwargs) -> tuple[Path, Image.Image]
- compounds: list of dict [{"smiles": "...", "label": "名称"}, ...]
- layout: "horizontal" | "vertical" | "grid"
- 将多个结构图排列在一张图上
- 每个结构图上方标注 label

### 风格预设字典（请在代码中包含完整实现）

```python
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
    }
}
```

### 同时创建 tests/test_structure_gen.py

测试用例：
1. 生成阿司匹林结构图 (SMILES: CC(=O)Oc1ccccc1C(=O)O)
2. 通过名称生成咖啡因
3. 测试四种风格输出
4. 测试透明背景（RGBA 模式，alpha 通道应为 0 的区域）
5. 测试无效 SMILES 抛出异常
6. 测试水平布局多分子合成
7. 测试网格布局

测试脚本应直接运行 `python tests/test_structure_gen.py`，打印每个测试结果。

### 完成后自动执行
1. 创建 outputs/structures/ 目录（通过 settings）
2. 生成以下测试图片并保存在 outputs/structures/ 下：
   - aspirin_acs1996.png
   - caffeine_dark.png
   - caffeine_aspirin_horizontal.png
3. 打印图片路径，供我手动验证视觉效果

请提供完整可运行的代码，不要省略任何部分。
```

---

## Phase 2: 合成引擎（分 2-3 次对话）

### Phase 2a：基础合成（直接贴图 + 透视变换）

### 上下文恢复
> Phase 1 完成。`src/structure_gen/generator.py` 可用，可生成透明背景结构图。
> 本 Phase 2a 目标：实现基础合成算法。

### 📋 Agent 指令

```
项目: chem-mindmap/，Phase 1 已完成。现在实现 Phase 2a：基础合成器。

请创建 src/compositor/basic.py，实现以下模块级函数：

### 1. load_image(path, mode='auto') -> np.ndarray
- 用 cv2 加载图片
- mode='auto': 自动检测，有 alpha 返回 RGBA，否则 BGR
- mode='rgb': 返回 RGB
- mode='bgr': 返回 BGR
- mode='rgba': 返回 RGBA（无 alpha 则补全不透明）

### 2. alpha_composite(bg, overlay, x, y) -> np.ndarray
- bg: H×W×3 BGR numpy array
- overlay: H×W×4 BGRA numpy array
- 将 overlay 合成到 bg 的 (x, y) 位置
- 使用标准 alpha 混合公式: output = overlay_rgb * alpha + bg * (1 - alpha)
- 检查边界，不越界

### 3. resize_with_alpha(img, target_w, target_h) -> np.ndarray
- 缩放 RGBA 图像
- 使用 cv2.INTER_LANCZOS4 插值

### 4. perspective_transform(img, src_points, dst_points, output_size) -> np.ndarray
- img: RGBA numpy array
- src_points: 4×2 numpy array (源四边形四个角)
- dst_points: 4×2 numpy array (目标四边形四个角)
- 计算透视变换矩阵 cv2.getPerspectiveTransform
- 用 cv2.warpPerspective 变换
- 返回 RGBA 结果

### 5. feather_alpha(alpha, radius=3) -> np.ndarray
- 对 alpha mask 做高斯模糊，实现边缘羽化
- radius: 模糊半径

### 6. add_drop_shadow(bg, alpha_mask, x, y, w, h, offset=(3,3), blur=5, opacity=0.3) -> np.ndarray
- 在 bg 上添加投影
- offset: 投影偏移（像素）
- blur: 高斯模糊半径
- opacity: 投影不透明度

### 同时创建 src/compositor/__init__.py
```python
from .basic import (
    load_image, alpha_composite, resize_with_alpha,
    perspective_transform, feather_alpha, add_drop_shadow,
)
```

### 同时创建 tests/test_compositor_basic.py

测试用例：
1. load_image 测试：加载 RGBA 和 RGB 图片
2. alpha_composite 测试：将半透明方形合成到背景
3. resize_with_alpha 测试：缩放并验证尺寸
4. perspective_transform 测试：变换一个矩形
5. feather_alpha 测试：验证边缘模糊效果
6. add_drop_shadow 测试：验证投影生成

请用测试图片（用 numpy 生成简单的彩色矩形代替实际图片）完成测试。

请提供完整可运行的代码。
```

---

### Phase 2b：光照匹配 + 纹理融合

### 上下文恢复
> Phase 2a 完成。`src/compositor/basic.py` 实现了基础合成。
> 本 Phase 2b 目标：实现光照匹配和纹理融合，消除"PS 感"。

### 📋 Agent 指令

```
项目: chem-mindmap/，Phase 2a 完成。现在实现 Phase 2b：光照匹配与纹理融合。

请创建 src/compositor/lighting.py，实现以下函数：

### 1. match_color_histogram(source, target, mask=None) -> np.ndarray
- source: H×W×3 结构图 RGB
- target: H×W×3 目标区域 RGB
- mask: 可选，仅对 mask 区域做匹配
- 使用直方图匹配（每个通道独立），将 source 的色彩分布匹配到 target
- 返回匹配后的图像

实现细节：
```python
def match_color_histogram(source, target, mask=None):
    """将 source 的颜色分布匹配到 target"""
    result = source.copy()
    for i in range(3):  # 三个通道
        if mask is not None:
            s = source[:,:,i][mask > 0]
            t = target[:,:,i][mask > 0]
        else:
            s = source[:,:,i].ravel()
            t = target[:,:,i].ravel()
      
        # 计算直方图并做匹配
        s_hist, _ = np.histogram(s, 256, [0, 256])
        t_hist, _ = np.histogram(t, 256, [0, 256])
      
        # CDF
        s_cdf = s_hist.cumsum() / s_hist.sum()
        t_cdf = t_hist.cumsum() / t_hist.sum()
      
        # 映射
        mapping = np.interp(s_cdf, t_cdf, np.arange(256))
        if mask is not None:
            result[:,:,i][mask > 0] = np.interp(source[:,:,i][mask > 0], 
                                                np.arange(256), mapping)
        else:
            result[:,:,i] = np.interp(source[:,:,i], np.arange(256), mapping)
  
    return result.astype(np.uint8)
```

### 2. match_color_stats(source, target) -> np.ndarray
- 简单的均值-标准差匹配（更轻量）
- source_matched = (source - mean_s) * (std_t / std_s) + mean_t

### 3. blend_surface_texture(struct_rgb, target_region, strength=0.08) -> np.ndarray
- 提取 target_region 的高频纹理成分
- 微量叠加到结构图上

```python
def blend_surface_texture(struct_rgb, target_region, strength=0.08):
    target_f = target_region.astype(np.float32)
    blurred = cv2.GaussianBlur(target_f, (21, 21), 0)
    texture = target_f - blurred  # 高频细节
  
    result = struct_rgb.astype(np.float32) + texture * strength
    return np.clip(result, 0, 255).astype(np.uint8)
```

### 4. add_ambient_shadow(bg, alpha_mask, x, y, w, h, ambient_strength=0.05) -> np.ndarray
- 在结构图周围添加环境光遮蔽（非常微妙的暗角效果）
- 使结构图看起来像是"嵌入"表面的

### 5. composite_with_lighting(bg, overlay_fg, x, y, w, h, 
                             color_match='stats', texture_strength=0.08, 
                             shadow=True) -> np.ndarray
- 完整流程：resize → color match → texture blend → alpha composite → shadow
- 返回合成后的 BG 图像

### 更新 src/compositor/__init__.py
添加新导出的函数

### 创建 tests/test_compositor_lighting.py
测试：加载一张真实照片和一张结构图，验证光照匹配效果

请提供完整代码。
```

---

### Phase 2c：拉普拉斯金字塔融合

### 上下文恢复
> Phase 2a（基础合成）和 Phase 2b（光照匹配）已完成。
> 本 Phase 2c 目标：实现最自然的金字塔融合算法。

### 📋 Agent 指令

```
项目: chem-mindmap/，Phase 2a 和 2b 完成。现在实现 Phase 2c：拉普拉斯金字塔融合。

请创建 src/compositor/pyramid.py，实现以下内容：

### 1. build_gaussian_pyramid(img, levels) -> list[np.ndarray]
- 从原图构建 levels 层高斯金字塔
- 每层 cv2.pyrDown
- 返回 [img_0, img_1, ..., img_{levels-1}]

### 2. build_laplacian_pyramid(img, levels) -> list[np.ndarray]
- 构建拉普拉斯金字塔
- LP_i = GP_i - pyrUp(GP_{i+1})
- 最低层 LP_{n-1} = GP_{n-1}

### 3. pyramid_blend(bg_pyramid, fg_pyramid, mask_pyramid, levels) -> list[np.ndarray]
- 每层: blended_i = bg_i * (1 - mask_i) + fg_i * mask_i

### 4. reconstruct_from_pyramid(pyramid) -> np.ndarray
- 从拉普拉斯金字塔重建图像
- 从顶层开始，逐层 pyrUp + 相加

### 5. seamless_composite(bg, overlay, x, y, w, h, levels=3) -> np.ndarray
- 完整无缝合成流程
- 从 bg 中提取 ROI
- 调整 overlay 尺寸到 (w, h)
- 构建金字塔
- 融合
- 重建
- 写回 bg
- 返回最终结果

### 更新 src/compositor/__init__.py
添加 pyramid_blend, seamless_composite

### 创建 tests/test_compositor_pyramid.py
测试：合成两张不同纹理的图片，验证边界过渡是否自然

请提供完整代码。
```

---

## Phase 3: 思维导图布局器

### 上下文恢复
> Phase 1-2 完成。结构图和合成引擎就绪。
> 本 Phase 目标：实现思维导图布局，将多个化合物节点组织成层级结构。

### 📋 Agent 指令

```
项目: chem-mindmap/，Phase 1-2 已完成。现在实现 Phase 3：思维导图布局器。

请创建 src/mindmap/layout.py，实现以下类：

## Node 类
```python
@dataclass
class Node:
    id: str
    label: str                    # 显示名称
    smiles: Optional[str] = None  # 化合物 SMILES
    children: list[Node] = field(default_factory=list)
    parent: Optional[Node] = None
  
    # 布局后填充
    x: int = 0
    y: int = 0
    width: int = 0
    height: int = 0
  
    # 结构图（由外部生成后设置）
    structure_image: Optional[np.ndarray] = None
  
    def add_child(self, child: 'Node'):
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
```

## MindMapLayout 类

### 构造参数
```python
def __init__(
    self,
    node_width: int = 200,      # 每个节点的宽度
    node_height: int = 150,     # 每个节点的高度
    horizontal_spacing: int = 100,  # 水平间距
    vertical_spacing: int = 60,     # 垂直间距
    padding: int = 50,          # 画布边距
):
```

### 核心方法

#### 1. layout_tree(root: Node) -> tuple[int, int]
- 输入根节点
- 计算所有节点的 (x, y) 坐标
- 使用 Reingold-Tilford 算法计算树布局
- 返回 (total_width, total_height)

**布局算法实现要点**：
```
对每个节点：
1. 如果是叶节点，分配一个垂直位置
2. 如果是内部节点：
   a. 递归布局子节点
   b. 将子节点平均分布
   c. 将当前节点放在子节点中间
3. 水平位置 = 深度 × (node_width + horizontal_spacing)
4. 垂直位置 = 计算出的 y
```

#### 2. render(root: Node, style='clean') -> np.ndarray
- 根据计算出的布局生成图像
- 返回 BGR numpy array

渲染细节：
```
画布尺寸 = (total_width + 2*padding, total_height + 2*padding)
背景色 = 白色或浅灰

对每个节点：
  - 绘制矩形框（圆角，浅色边框）
  - 在框内绘制标签文字
  - 如果节点有 structure_image，在框内适当位置绘制

对每条边（父→子）：
  - 绘制连接线（从父节点右侧到子节点左侧）
  - 可选：使用阶梯线（orthogonal path）
```

#### 3. set_structure(node: Node, structure_img: np.ndarray)
- 为节点设置结构图

#### 4. from_json(data: dict) -> Node
- 从 JSON/dict 构建树结构
- 期望格式：
```json
{
    "label": "有机化学",
    "children": [
        {"label": "醇类", "children": [
            {"label": "乙醇", "smiles": "CCO"},
            {"label": "甲醇", "smiles": "CO"}
        ]},
        {"label": "酸类", "children": [
            {"label": "乙酸", "smiles": "CC(=O)O"}
        ]}
    ]
}
```

### 同时创建 tests/test_mindmap.py

测试：
1. 构建一个简单的树（3 层，7 个节点）
2. 测试布局坐标计算
3. 测试渲染输出（保存图片到 outputs/mindmaps/）
4. 测试从 JSON 构建树

### 重要提示
- 这个布局器生成的是"骨架图"——包含框、连接线、文字和结构图占位
- 最终效果的"美化"（背景、光影、材质）由 Phase 4 的 AI 场景生成和合成引擎处理
- 所以 render() 生成的是 clean 风格（白色背景，干净线条）

请提供完整代码。
```

---

## Phase 4: AI 场景生成

### 上下文恢复
> Phase 1-3 完成。结构图、合成器、布局器都可用。
> 本 Phase 目标：实现 AI 图像生成接口，将思维导图骨架"美化"为论文级图片。

### 📋 Agent 指令

```
项目: chem-mindmap/，Phase 1-3 已完成。现在实现 Phase 4：AI 场景生成接口。

请创建 src/scene_gen/generator.py，实现以下内容：

## SceneGenerator 类

### 构造参数
```python
def __init__(self, provider: str = "sd_webui", config: Optional[dict] = None):
```
- provider: "sd_webui" | "openai" | "replicate" | "mock"
- config: 可选，覆盖 settings 中的配置

### 核心方法

#### 1. generate(prompt: str, negative_prompt: str = "", 
               width: int = 1024, height: int = 768,
               guidance_scale: float = 7.5, num_steps: int = 30,
               control_image: Optional[np.ndarray] = None,
               control_type: str = "canny") -> Image.Image
- 生成 AI 图像
- control_image: ControlNet 条件图（可选）
- control_type: "canny" | "depth" | "scribble"

#### 2. enhance_mindmap(mindmap_image: np.ndarray, 
                       style_prompt: str = "") -> Image.Image
- 将思维导图骨架图"美化"
- 流程：
  1. 构建增强 prompt（包含思维导图风格描述）
  2. 从骨架图提取 Canny 边缘
  3. 用 ControlNet + Canny 条件生成
  4. 返回生成的美化图像

**增强 prompt 模板**：
```python
ENHANCE_PROMPTS = {
    "academic": (
        "clean academic mind map for organic chemistry journal paper, "
        "white background with subtle gradient, professional diagram style, "
        "clear hierarchical layout showing chemical compounds and reactions, "
        "sharp text, high resolution, suitable for publication, "
        "molecular structures are clearly visible and scientifically accurate"
    ),
    "modern": (
        "modern scientific infographic style mind map, "
        "vibrant but professional colors, gradient background, "
        "organic chemistry concepts organized hierarchically, "
        "3D effects on molecular structures, "
        "clean typography, presentation quality"
    ),
    "minimal": (
        "minimalist black and white scientific diagram, "
        "clean lines, no colors except structural formulas, "
        "chemistry mind map for academic paper, "
        "high contrast, sharp edges, publication ready"
    )
}
```

#### 3. generate_style_prompt(mindmap_json: dict, style: str = "academic") -> str
- 根据思维导图内容和风格自动生成 prompt

### 后端实现

#### SD WebUI 后端
```python
def _generate_sd_webui(self, payload: dict) -> Image.Image:
    import requests
  
    url = f"{settings.SD_WEBUI_URL}/sdapi/v1/txt2img"
    response = requests.post(url, json=payload, timeout=300)
    response.raise_for_status()
  
    data = response.json()
    # data["images"] 是 base64 编码的列表
    import base64
    from io import BytesIO
  
    img_data = base64.b64decode(data["images"][0])
    return Image.open(BytesIO(img_data))
```

#### ControlNet 支持
```python
def _generate_with_controlnet(self, prompt, control_image, control_type):
    # 预处理控制图像
    if control_type == "canny":
        from cv2 import Canny
        edges = Canny(control_image, 100, 200)
        control_processed = edges
    elif control_type == "depth":
        # 使用 MiDaS 或其他深度估计
        pass
  
    # SD WebUI ControlNet API
    import base64
    _, buffer = cv2.imencode('.png', control_processed)
    control_base64 = base64.b64encode(buffer).decode('utf-8')
  
    payload = {
        "prompt": prompt,
        "negative_prompt": "low quality, blurry, distorted text",
        "width": 1024,
        "height": 768,
        "steps": 30,
        "cfg_scale": 7.5,
        "controlnet_units": [{
            "input_image": control_base64,
            "module": control_type,
            "model": "control_v11p_sd15_canny [d14c2d8b]",
            "weight": 0.85,
        }]
    }
  
    return self._generate_sd_webui(payload)
```

### Mock 模式
当 provider 为 "mock" 时，生成简单的纯色图片用于测试。

### 创建 src/scene_gen/__init__.py
导出 SceneGenerator 类

### 创建 tests/test_scene_gen.py
测试：
1. Mock 模式生成 512x512 图片
2. SD WebUI 连接测试（如果可用）
3. Canny 边缘提取测试

请提供完整代码。
```

---

## Phase 5: Agent 编排层

### 上下文恢复
> Phase 1-4 全部完成。所有核心模块就绪。
> 本 Phase 目标：实现 Agent 系统，串联所有模块，由 LLM 驱动工作流。

### 📋 Agent 指令

```
项目: chem-mindmap/，Phase 1-4 已完成。现在实现 Phase 5：Agent 编排层。

这是核心编排系统，包含三个文件：
1. src/agent/tools.py — Tool 定义
2. src/agent/orchestrator.py — 编排逻辑
3. src/agent/prompts.py — LLM 提示词

### 文件 1: src/agent/tools.py

定义 Agent 可用的 Tools：

```python
"""
Agent Tool 系统
每个 Tool 是一个可调用的函数，带有 schema 描述。
"""
from dataclasses import dataclass, field
from typing import Any, Callable
import json


@dataclass
class Tool:
    """工具定义"""
    name: str
    description: str
    parameters: dict  # JSON Schema
    func: Callable
  
    def to_openai_tool(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            }
        }
  
    def to_anthropic_tool(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.parameters,
        }
  
    def __call__(self, **kwargs) -> Any:
        return self.func(**kwargs)


class ToolRegistry:
    """工具注册中心"""
  
    def __init__(self):
        self._tools: dict[str, Tool] = {}
  
    def register(self, tool: Tool):
        self._tools[tool.name] = tool
  
    def get(self, name: str) -> Tool:
        return self._tools[name]
  
    def list_tools(self) -> list[Tool]:
        return list(self._tools.values())
  
    def to_anthropic_tools(self) -> list[dict]:
        return [t.to_anthropic_tool() for t in self._tools.values()]
  
    def to_openai_tools(self) -> list[dict]:
        return [t.to_openai_tool() for t in self._tools.values()]


# 以下是所有可用 Tools 的定义

def register_all_tools(registry: ToolRegistry):
    """注册所有项目工具"""
  
    # 1. 解析化合物
    registry.register(Tool(
        name="resolve_compound",
        description="通过名称或 SMILES 解析化合物信息",
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "化合物名称或 SMILES 字符串"
                }
            },
            "required": ["query"]
        },
        func=lambda query: _resolve_compound_impl(query)
    ))
  
    # 2. 生成结构图
    registry.register(Tool(
        name="generate_structure",
        description="为化合物生成精确的 2D 结构图",
        parameters={
            "type": "object",
            "properties": {
                "smiles": {"type": "string", "description": "SMILES 字符串"},
                "style": {
                    "type": "string",
                    "enum": ["ACS_1996", "dark_mode", "color_on_white", "minimal"],
                    "description": "结构图风格"
                },
                "output_path": {"type": "string", "description": "输出路径（可选）"}
            },
            "required": ["smiles"]
        },
        func=lambda **kwargs: _generate_structure_impl(**kwargs)
    ))
  
    # 3. 构建思维导图
    registry.register(Tool(
        name="build_mindmap",
        description="根据化合物关系构建思维导图布局",
        parameters={
            "type": "object",
            "properties": {
                "tree_json": {
                    "type": "string",
                    "description": "思维导图的 JSON 树结构"
                },
                "output_path": {"type": "string", "description": "输出路径"}
            },
            "required": ["tree_json", "output_path"]
        },
        func=lambda **kwargs: _build_mindmap_impl(**kwargs)
    ))
  
    # 4. 生成场景
    registry.register(Tool(
        name="generate_scene",
        description="生成思维导图的美化场景图",
        parameters={
            "type": "object",
            "properties": {
                "prompt": {"type": "string", "description": "场景描述"},
                "style": {"type": "string", "enum": ["academic", "modern", "minimal"]},
                "output_path": {"type": "string", "description": "输出路径"}
            },
            "required": ["prompt", "output_path"]
        },
        func=lambda **kwargs: _generate_scene_impl(**kwargs)
    ))
  
    # 5. 合成最终图
    registry.register(Tool(
        name="composite_final",
        description="将结构图合成到场景图中，生成最终图像",
        parameters={
            "type": "object",
            "properties": {
                "scene_path": {"type": "string", "description": "场景图路径"},
                "structures_info": {
                    "type": "string",
                    "description": "结构图信息 JSON（路径+位置）"
                },
                "output_path": {"type": "string", "description": "输出路径"}
            },
            "required": ["scene_path", "structures_info", "output_path"]
        },
        func=lambda **kwargs: _composite_final_impl(**kwargs)
    ))


# 实现函数（导入实际模块）
def _resolve_compound_impl(query):
    """解析化合物"""
    from src.structure_gen.generator import StructureGenerator
    gen = StructureGenerator()
    return gen.resolve_compound(query)

def _generate_structure_impl(**kwargs):
    """生成结构图"""
    from src.structure_gen.generator import StructureGenerator
    gen = StructureGenerator()
    path, _ = gen.generate_from_smiles(
        kwargs["smiles"],
        style=kwargs.get("style", "ACS_1996"),
        output_path=kwargs.get("output_path")
    )
    return {"path": str(path)}

def _build_mindmap_impl(**kwargs):
    """构建思维导图"""
    import json
    from src.mindmap.layout import MindMapLayout, Node
  
    data = json.loads(kwargs["tree_json"])
  
    def build_node(d):
        n = Node(id=d.get("id", d["label"]), label=d["label"], smiles=d.get("smiles"))
        for child in d.get("children", []):
            n.add_child(build_node(child))
        return n
  
    root = build_node(data)
    layout = MindMapLayout()
    layout.layout_tree(root)
    img = layout.render(root)
  
    import cv2
    cv2.imwrite(kwargs["output_path"], img)
    return {"path": kwargs["output_path"], "width": img.shape[1], "height": img.shape[0]}

def _generate_scene_impl(**kwargs):
    """生成场景"""
    from src.scene_gen.generator import SceneGenerator
    gen = SceneGenerator()
  
    # 如果是美化模式，先加载思维导图
    if kwargs.get("mindmap_path"):
        import cv2
        mindmap = cv2.imread(kwargs["mindmap_path"])
        img = gen.enhance_mindmap(mindmap, kwargs.get("style", "academic"))
    else:
        img = gen.generate(kwargs["prompt"])
  
    img.save(kwargs["output_path"])
    return {"path": kwargs["output_path"]}

def _composite_final_impl(**kwargs):
    """合成最终图"""
    import json
    from src.compositor.basic import load_image, alpha_composite, resize_with_alpha
    from src.compositor.lighting import composite_with_lighting
    import cv2
  
    bg = load_image(kwargs["scene_path"])
    structures = json.loads(kwargs["structures_info"])
  
    for s in structures:
        overlay = load_image(s["path"])
        x, y, w, h = s["x"], s["y"], s["w"], s["h"]
        overlay = resize_with_alpha(overlay, w, h)
        bg = composite_with_lighting(
            bg, overlay, x, y, w, h,
            color_match='stats',
            texture_strength=0.08,
            shadow=True
        )
  
    cv2.imwrite(kwargs["output_path"], bg)
    return {"path": kwargs["output_path"]}
```

### 文件 2: src/agent/orchestrator.py

```python
"""
Agent 编排器
根据用户输入，调用 LLM 决定调用哪些 Tools，执行工作流。
"""
import json
import logging
from pathlib import Path
from typing import Optional
from .tools import ToolRegistry, register_all_tools
from .prompts import SYSTEM_PROMPT, build_user_prompt

logger = logging.getLogger(__name__)


class Orchestrator:
    """
    Agent 编排器。
  
    工作流程：
    1. 接收用户输入
    2. 调用 LLM，LLM 决定调用哪些 Tools
    3. 执行 Tools
    4. 将结果返回给 LLM 进行下一步决策
    5. 循环直至完成
  
    使用方式：
        orch = Orchestrator()
        result = orch.run("生成关于醇类化学反应的思维导图")
    """
  
    def __init__(self, llm_provider: str = "claude"):
        self.registry = ToolRegistry()
        register_all_tools(self.registry)
        self.llm_provider = llm_provider
        self.conversation_history = []
  
    def run(self, user_input: str, output_dir: Optional[str] = None) -> dict:
        """
        执行完整的生成流程。
      
        Args:
            user_input: 用户描述（如"生成关于苯酚及其衍生物的思维导图"）
            output_dir: 输出目录
          
        Returns:
            {
                "final_image": "outputs/final/result.png",
                "workflow": [...],
                "compounds": [...],
                "mindmap": {...}
            }
        """
        # 初始化对话
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(user_input)}
        ]
      
        # 调用 LLM 获取行动计划
        plan = self._call_llm_with_tools(messages)
        logger.info(f"LLM 计划: {plan}")
      
        # 执行计划
        result = self._execute_plan(plan, output_dir)
      
        return result
  
    def _call_llm_with_tools(self, messages: list) -> dict:
        """调用 LLM 并返回工具调用计划"""
        if self.llm_provider == "claude":
            return self._call_claude(messages)
        elif self.llm_provider == "openai":
            return self._call_openai(messages)
        else:
            # 本地模式：直接执行默认流程
            return {"plan": "default"}
  
    def _call_claude(self, messages: list) -> dict:
        """调用 Claude API（示例实现，实际需要 API Key）"""
        # 这里需要导入 anthropic 包
        # 但由于我们在 CLI 环境中，实际调用在 GUI 中处理
        # 这里返回一个模拟计划
        return {
            "steps": [
                {"tool": "resolve_compound", "params": {"query": "苯酚"}},
                {"tool": "resolve_compound", "params": {"query": "苯甲酸"}},
                {"tool": "generate_structure", "params": {"smiles": "...", "style": "ACS_1996"}},
                {"tool": "build_mindmap", "params": {"tree_json": "..."}},
                {"tool": "generate_scene", "params": {"prompt": "...", "style": "academic"}},
                {"tool": "composite_final", "params": {"structures_info": "..."}},
            ]
        }
  
    def _call_openai(self, messages: list) -> dict:
        """调用 OpenAI API"""
        # 同上面类似，需要 API Key
        raise NotImplementedError("OpenAI 后端待实现")
  
    def _execute_plan(self, plan: dict, output_dir: Optional[str] = None) -> dict:
        """执行工具调用计划"""
        results = {}
      
        for step in plan.get("steps", []):
            tool_name = step["tool"]
            params = step.get("params", {})
          
            tool = self.registry.get(tool_name)
            result = tool(**params)
            results[tool_name] = result
          
            logger.info(f"执行 {tool_name}: {result}")
      
        return {
            "final_image": results.get("composite_final", {}).get("path", ""),
            "workflow": plan.get("steps", []),
            "results": results,
        }
```

### 文件 3: src/agent/prompts.py

```python
"""
LLM 提示词模板
"""

SYSTEM_PROMPT = """你是 Chemistry Mind Map Generator，一个有机化学思维导图生成 AI 助手。

## 你的职责
根据用户的描述，生成包含精确化学结构的高质量思维导图，可发表在学术论文上。

## 能力
1. 自动识别用户描述中的有机化合物名称
2. 调用工具生成精确的化学结构图（使用 RDKit，分子结构 100% 精确）
3. 构建层级化的思维导图布局
4. 生成美观的学术风格场景
5. 将结构图无缝合成到最终图像中

## 工作流程
1. 【解析】分析用户的描述，提取所有化合物名称、关系和层级
2. 【查证】通过 PubChem 解析每个化合物的 SMILES
3. 【结构图】为每个化合物生成精确的 2D 结构图
4. 【布局】根据化合物关系构建思维导图树
5. 【场景】生成学术风格的背景场景
6. 【合成】将结构图合成到场景中
7. 【输出】返回最终图像路径

## 重要规则
- 化学结构必须 100% 精确，使用 SMILES 生成，不依赖 AI 绘制
- 思维导图层级要逻辑清晰
- 最终图像要符合学术论文的风格要求
- 如果用户描述不明确，主动询问澄清

## 可用工具
{tools_description}

请开始处理用户的请求。
"""


def build_user_prompt(user_input: str) -> str:
    """构建用户提示词"""
    return f"""请根据以下描述，生成有机化学思维导图：

{user_input}

请先分析描述中包含的化合物和它们的关系，然后逐步生成。
注意：
1. 识别所有化合物名称
2. 理解它们的层级关系（如分类、反应类型）
3. 选择合适的风格（默认：学术风格）
4. 输出应包含思维导图的 JSON 树结构和每个节点的 SMILES
"""


def build_tools_description(tools: list) -> str:
    """构建工具描述"""
    descs = []
    for t in tools:
        descs.append(f"- {t.name}: {t.description}")
    return "\n".join(descs)
```

### 创建 src/agent/__init__.py
```python
from .tools import Tool, ToolRegistry, register_all_tools
from .orchestrator import Orchestrator
from .prompts import SYSTEM_PROMPT, build_user_prompt
```

### 创建 tests/test_agent.py
测试 Tool 注册和编排器初始化

请提供所有三个文件的完整代码。
```

---

## Phase 6: GUI 界面

### 上下文恢复
> Phase 1-5 完成。所有核心逻辑和编排系统就绪。
> 本 Phase 目标：用 Gradio 构建 GUI 界面。

### 📋 Agent 指令

```
项目: chem-mindmap/，Phase 1-5 已完成。现在实现 Phase 6：GUI 界面。

请创建 src/gui/app.py，用 Gradio 构建用户界面。

## 界面设计

### 标签页 1: 主生成界面
```
┌─────────────────────────────────────────────────┐
│ 🧪 有机化学思维导图生成器                        │
├─────────────────────────────────────────────────┤
│                                                   │
│  📝 输入描述:                                     │
│  ┌─────────────────────────────────────────────┐ │
│  │ 生成关于苯酚、苯甲酸及其酯化反应的思维导图   │ │
│  │                                           │ │
│  └─────────────────────────────────────────────┘ │
│                                                   │
│  ⚙️ 参数设置:                                    │
│  风格: [学术风格 ▼]  尺寸: [1920×1080 ▼]         │
│  AI 后端: [本地 SD WebUI ▼]                      │
│                                                   │
│  [🔬 开始生成]  [💾 保存配置]  [📖 历史记录]     │
│                                                   │
│  📊 生成进度:                                     │
│  ████████████████░░░░░░░ 75%                      │
│  ┌─────────────────────────────────────────────┐ │
│  │ [实时预览区域 - 显示生成过程中的中间结果]     │ │
│  │                                              │ │
│  │                                              │ │
│  └─────────────────────────────────────────────┘ │
│                                                   │
│  📋 生成结果:                                     │
│  ┌─────────────────────────────────────────────┐ │
│  │ [最终图像展示区域]                            │ │
│  │                                              │ │
│  │                                              │ │
│  └─────────────────────────────────────────────┘ │
│  [📥 下载 PNG] [📥 下载 SVG] [🔄 重新生成]      │
└─────────────────────────────────────────────────┘
```

### 标签页 2: 结构图预览
```
┌─────────────────────────────────────────────────┐
│ 分子结构图预览器                                 │
├─────────────────────────────────────────────────┤
│                                                   │
│  化合物名称: [aspirin      ▼] [🔍 查询]          │
│  SMILES:   CC(=O)Oc1ccccc1C(=O)O                  │
│  分子式:   C₉H₈O₄     分子量: 180.16              │
│                                                   │
│  [生成结构图]  风格: [ACS_1996 ▼]                 │
│                                                   │
│  ┌─────────────────────────────────────────────┐ │
│  │ [结构图预览]                                 │ │
│  │                                              │ │
│  └─────────────────────────────────────────────┘ │
│                                                   │
│  [📥 下载 PNG]                                    │
└─────────────────────────────────────────────────┘
```

### 标签页 3: 设置
```
┌─────────────────────────────────────────────────┐
│ 设置                                             │
├─────────────────────────────────────────────────┤
│                                                   │
│  🔗 API 连接:                                     │
│  图像生成 API: [sd_webui ▼]                       │
│  API URL: [http://127.0.0.1:7860]                 │
│  LLM API: [claude ▼]                              │
│  API Key: [***********]                            │
│                                                   │
│  🖼️ 图像默认设置:                                 │
│  默认宽度: [1920]  默认高度: [1080]                │
│  默认风格: [学术风格]                               │
│                                                   │
│  📂 输出设置:                                     │
│  输出目录: [./outputs]                             │
│                                                   │
│  [💾 保存设置]                                    │
└─────────────────────────────────────────────────┘
```

## 代码实现

### src/gui/app.py

```python
"""
GUI 应用程序
基于 Gradio 构建
"""
import gradio as gr
import json
from pathlib import Path
from typing import Optional
import threading
import time
import logging

from src.config import settings
from src.agent.orchestrator import Orchestrator
from src.structure_gen.generator import StructureGenerator
from src.mindmap.layout import MindMapLayout, Node

logger = logging.getLogger(__name__)

# 全局状态
_current_result = None


def build_interface():
    """构建 Gradio 界面"""
  
    with gr.Blocks(title="🧪 有机化学思维导图生成器", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🧪 有机化学思维导图生成器")
        gr.Markdown("输入你的想法，自动生成包含精确化学结构的学术级思维导图")
      
        with gr.Tabs():
            # === Tab 1: 主生成界面 ===
            with gr.TabItem("📝 生成思维导图"):
                _build_generation_tab()
          
            # === Tab 2: 结构图预览 ===
            with gr.TabItem("🔬 结构图预览"):
                _build_structure_preview_tab()
          
            # === Tab 3: 设置 ===
            with gr.TabItem("⚙️ 设置"):
                _build_settings_tab()
  
    return demo


def _build_generation_tab():
    """构建生成选项卡"""
  
    with gr.Row():
        with gr.Column(scale=2):
            # 输入区域
            input_text = gr.Textbox(
                label="📝 输入描述",
                placeholder="例如：生成关于苯酚、苯甲酸及其酯化反应的思维导图",
                lines=3,
            )
          
            # 参数
            with gr.Row():
                style = gr.Dropdown(
                    choices=["academic", "modern", "minimal"],
                    value="academic",
                    label="🎨 风格"
                )
                size = gr.Dropdown(
                    choices=["1920×1080", "2560×1440", "3840×2160"],
                    value="1920×1080",
                    label="📐 尺寸"
                )
                provider = gr.Dropdown(
                    choices=["sd_webui", "openai", "replicate", "mock"],
                    value=settings.AI_IMAGE_PROVIDER,
                    label="🤖 AI 后端"
                )
      
        with gr.Column(scale=1):
            # 状态和操作
            generate_btn = gr.Button("🔬 开始生成", variant="primary", size="lg")
            status = gr.Markdown("✅ 就绪，等待输入...")
  
    # 进度和预览
    with gr.Row():
        with gr.Column():
            progress = gr.HTML(
                value="""<div style="width:100%;height:20px;background:#e0e0e0;border-radius:10px;overflow:hidden;">
                    <div id="progress-bar" style="width:0%;height:100%;background:#4CAF50;transition:width 0.5s;"></div>
                </div>"""
            )
            preview = gr.Image(label="🖼️ 中间预览", type="filepath", height=400)
      
        with gr.Column():
            result_image = gr.Image(label="🎯 最终结果", type="filepath", height=400)
  
    # 操作按钮
    with gr.Row():
        download_btn = gr.Button("📥 下载 PNG")
        refresh_btn = gr.Button("🔄 重新生成")
        save_config_btn = gr.Button("💾 保存配置")
  
    # 事件绑定
    generate_btn.click(
        fn=_on_generate,
        inputs=[input_text, style, size, provider],
        outputs=[result_image, preview, status, progress],
    )
  
    refresh_btn.click(
        fn=lambda: None,
        outputs=[],
    )
  
    return input_text, style, size, provider, generate_btn, result_image, status


def _build_structure_preview_tab():
    """构建结构图预览选项卡"""
  
    with gr.Row():
        compound_name = gr.Dropdown(
            label="化合物名称",
            choices=["aspirin", "caffeine", "paracetamol", "ibuprofen",
                     "benzene", "phenol", "benzoic_acid", "ethanol", "acetic_acid"],
            value="aspirin",
            allow_custom_value=True,
        )
        smiles_input = gr.Textbox(
            label="SMILES",
            placeholder="自动填充或手动输入",
            value="CC(=O)Oc1ccccc1C(=O)O",
        )
        query_btn = gr.Button("🔍 查询")
  
    with gr.Row():
        style_select = gr.Dropdown(
            choices=["ACS_1996", "dark_mode", "color_on_white", "minimal"],
            value="ACS_1996",
            label="🎨 结构图风格"
        )
        gen_struct_btn = gr.Button("生成结构图")
  
    struct_preview = gr.Image(label="结构图预览", type="filepath", height=400)
  
    with gr.Row():
        formula = gr.Textbox(label="分子式", interactive=False)
        mol_weight = gr.Textbox(label="分子量", interactive=False)
  
    # 事件绑定
    query_btn.click(
        fn=_on_query_compound,
        inputs=[compound_name],
        outputs=[smiles_input, formula, mol_weight],
    )
  
    gen_struct_btn.click(
        fn=_on_generate_structure,
        inputs=[smiles_input, style_select],
        outputs=[struct_preview],
    )


def _build_settings_tab():
    """构建设置选项卡"""
  
    with gr.Group():
        gr.Markdown("### 🔗 API 连接")
      
        api_provider = gr.Dropdown(
            choices=["sd_webui", "openai", "replicate", "mock"],
            value=settings.AI_IMAGE_PROVIDER,
            label="图像生成 API",
        )
        api_url = gr.Textbox(
            value=settings.SD_WEBUI_URL,
            label="API URL",
        )
        llm_provider = gr.Dropdown(
            choices=["claude", "openai", "local"],
            value=settings.LLM_PROVIDER,
            label="LLM API",
        )
        api_key = gr.Textbox(
            value="",
            label="API Key",
            type="password",
        )
  
    with gr.Group():
        gr.Markdown("### 🖼️ 图像设置")
      
        default_width = gr.Number(value=1920, label="默认宽度")
        default_height = gr.Number(value=1080, label="默认高度")
        default_style = gr.Dropdown(
            choices=["academic", "modern", "minimal"],
            value="academic",
            label="默认风格",
        )
  
    with gr.Group():
        gr.Markdown("### 📂 输出设置")
        output_dir = gr.Textbox(value=str(settings.OUTPUT_DIR), label="输出目录")
  
    save_settings_btn = gr.Button("💾 保存设置")
  
    # 事件
    save_settings_btn.click(
        fn=_on_save_settings,
        inputs=[api_provider, api_url, llm_provider, default_width, default_height, default_style, output_dir],
        outputs=[],
    )


# ─── 事件处理函数 ───────────────────────────────────────

def _on_generate(input_text: str, style: str, size: str, provider: str) -> tuple:
    """处理生成请求"""
    global _current_result
  
    if not input_text.strip():
        return None, None, "⚠️ 请输入描述内容", ""
  
    # 解析尺寸
    width, height = map(int, size.split("×"))
  
    # 更新状态
    status_msg = "🔄 正在解析化合物..."
  
    try:
        # 初始化编排器
        orchestrator = Orchestrator(llm_provider=settings.LLM_PROVIDER)
      
        # 执行生成（在实际环境中，这里会调用 LLM）
        # 目前用 mock 流程演示
        result = orchestrator.run(input_text)
      
        final_path = result.get("final_image", "")
      
        if final_path and Path(final_path).exists():
            return (
                final_path,                    # 结果图
                final_path,                    # 预览
                f"✅ 生成完成！输出: {final_path}",
                "100%",
            )
        else:
            return (
                None,
                None,
                "⚠️ 生成失败，请查看日志",
                "0%",
            )
  
    except Exception as e:
        logger.exception("生成失败")
        return (
            None,
            None,
            f"❌ 错误: {str(e)}",
            "",
        )


def _on_query_compound(name: str) -> tuple:
    """查询化合物信息"""
    try:
        gen = StructureGenerator()
        info = gen.resolve_compound(name)
        return (
            info.smiles,
            info.formula,
            f"{info.molecular_weight:.2f}",
        )
    except Exception as e:
        return (
            f"查询失败: {e}",
            "",
            "",
        )


def _on_generate_structure(smiles: str, style: str) -> str:
    """生成单个结构图"""
    try:
        gen = StructureGenerator(default_style=style)
        path, img = gen.generate_from_smiles(smiles)
        return str(path)
    except Exception as e:
        return None


def _on_save_settings(*args):
    """保存设置"""
    # 在实际项目中应写入 .env 文件
    api_provider, api_url, llm_provider, width, height, style, output_dir = args
    logger.info(f"设置已更新: {api_provider=}, {width=}x{height=}")
    return


# ─── 启动入口 ───────────────────────────────────────────

def launch(share: bool = False, port: int = 7860):
    """启动 GUI"""
    demo = build_interface()
    demo.launch(
        share=share,
        server_port=port,
        server_name="0.0.0.0",
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    launch()
```

### 创建 src/gui/__init__.py
```python
from .app import launch, build_interface
```

### 根目录启动脚本 run.py

在项目根目录创建 `run.py`：

```python
#!/usr/bin/env python3
"""启动 GUI 应用"""
import sys
from pathlib import Path

# 确保 src 在路径中
sys.path.insert(0, str(Path(__file__).parent))

from src.gui import launch

if __name__ == "__main__":
    import argparse
  
    parser = argparse.ArgumentParser(description="启动化学思维导图生成器")
    parser.add_argument("--port", type=int, default=7860, help="端口号")
    parser.add_argument("--share", action="store_true", help="创建公开链接")
  
    args = parser.parse_args()
  
    print("🧪 有机化学思维导图生成器")
    print(f"   启动中... 端口: {args.port}")
  
    launch(share=args.share, port=args.port)
```

### 验证
```bash
python run.py
# 浏览器打开 http://127.0.0.1:7860
# 应该看到完整的 GUI 界面
```

请提供所有文件的完整代码。
```

---

## Phase 7: 端到端联调

### 上下文恢复
> Phase 1-6 全部完成。所有模块和 GUI 就绪。
> 本 Phase 目标：整合测试，确保整个流水线可以运行。

### 📋 Agent 指令

```
项目: chem-mindmap/，所有 Phase 已完成。现在进行 Phase 7：端到端联调。

请创建 tests/test_pipeline.py，实现完整的端到端测试。

### 测试流程

```python
"""
端到端流水线测试
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import json
import logging
from src.structure_gen.generator import StructureGenerator
from src.compositor.basic import load_image, alpha_composite, resize_with_alpha
from src.compositor.lighting import composite_with_lighting
from src.compositor.pyramid import seamless_composite
from src.mindmap.layout import MindMapLayout, Node
from src.scene_gen.generator import SceneGenerator
from src.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_full_pipeline():
    """
    完整流水线测试：
    输入 → 解析化合物 → 结构图 → 思维导图布局 → 场景生成 → 合成 → 输出
    """
    logger.info("=" * 60)
    logger.info("开始端到端流水线测试")
    logger.info("=" * 60)
  
    # 1. 解析化合物
    logger.info("\n[Step 1] 解析化合物")
    gen = StructureGenerator()
  
    compounds = [
        {"name": "苯酚", "smiles": "c1ccc(cc1)O"},
        {"name": "苯甲酸", "smiles": "c1ccc(cc1)C(=O)O"},
        {"name": "水杨酸", "smiles": "c1ccc(c(c1)C(=O)O)O"},
        {"name": "乙酸", "smiles": "CC(=O)O"},
        {"name": "乙醇", "smiles": "CCO"},
    ]
  
    for c in compounds:
        try:
            info = gen.resolve_compound(c["name"])
            logger.info(f"  ✅ {c['name']}: {info.smiles}")
        except Exception as e:
            logger.error(f"  ❌ {c['name']}: {e}")
  
    # 2. 生成结构图
    logger.info("\n[Step 2] 生成结构图")
    struct_paths = {}
    for c in compounds:
        try:
            path, img = gen.generate_from_smiles(c["smiles"], style="ACS_1996")
            struct_paths[c["name"]] = {
                "path": str(path),
                "smiles": c["smiles"],
            }
            logger.info(f"  ✅ {c['name']}: {path}")
        except Exception as e:
            logger.error(f"  ❌ {c['name']}: {e}")
  
    # 3. 构建思维导图
    logger.info("\n[Step 3] 构建思维导图")
  
    # 构建树：有机化学 → 芳香族化合物 + 脂肪族化合物
    root = Node(id="root", label="有机化学")
  
    aromatic = Node(id="aromatic", label="芳香族化合物")
    aromatic.add_child(Node(id="phenol", label="苯酚", smiles="c1ccc(cc1)O"))
    aromatic.add_child(Node(id="benzoic", label="苯甲酸", smiles="c1ccc(cc1)C(=O)O"))
    aromatic.add_child(Node(id="salicylic", label="水杨酸", smiles="c1ccc(c(c1)C(=O)O)O"))
    root.add_child(aromatic)
  
    aliphatic = Node(id="aliphatic", label="脂肪族化合物")
    aliphatic.add_child(Node(id="acetic", label="乙酸", smiles="CC(=O)O"))
    aliphatic.add_child(Node(id="ethanol", label="乙醇", smiles="CCO"))
    root.add_child(aliphatic)
  
    layout = MindMapLayout(node_width=180, node_height=130)
    total_w, total_h = layout.layout_tree(root)
    logger.info(f"  布局尺寸: {total_w}×{total_h}")
  
    mindmap_img = layout.render(root)
    mindmap_path = str(settings.OUTPUT_DIR / "mindmaps" / "test_mindmap.png")
    cv2.imwrite(mindmap_path, mindmap_img)
    logger.info(f"  ✅ 思维导图已保存: {mindmap_path}")
  
    # 4. 生成场景（mock模式）
    logger.info("\n[Step 4] 生成场景")
    scene_gen = SceneGenerator(provider="mock")
    scene = scene_gen.generate(
        prompt="academic mind map for chemistry paper",
        width=1920,
        height=1080,
    )
    scene_path = str(settings.OUTPUT_DIR / "scenes" / "test_scene.png")
    scene.save(scene_path)
    logger.info(f"  ✅ 场景已保存（mock）: {scene_path}")
  
    # 5. 合成结构图到场景
    logger.info("\n[Step 5] 合成最终图像")
  
    bg = cv2.imread(scene_path)
    h_bg, w_bg = bg.shape[:2]
  
    # 定义每个结构图的位置（基于思维导图布局的坐标）
    positions = {
        "苯酚": (300, 300, 200, 150),
        "苯甲酸": (300, 500, 200, 150),
        "水杨酸": (300, 700, 200, 150),
        "乙酸": (1000, 400, 200, 150),
        "乙醇": (1000, 600, 200, 150),
    }
  
    for name, info in struct_paths.items():
        if name in positions:
            overlay = load_image(info["path"])
            x, y, w, h = positions[name]
            overlay = resize_with_alpha(overlay, w, h)
          
            bg = composite_with_lighting(
                bg, overlay, x, y, w, h,
                color_match='stats',
                texture_strength=0.08,
                shadow=True,
            )
            logger.info(f"  ✅ 合成 {name} 到 ({x}, {y})")
  
    final_path = str(settings.OUTPUT_DIR / "final" / "test_final.png")
    cv2.imwrite(final_path, bg)
    logger.info(f"  ✅ 最终图像已保存: {final_path}")
  
    # 6. 验证输出
    logger.info("\n[Step 6] 验证输出")
    assert Path(final_path).exists(), "最终图像不存在"
  
    final_img = cv2.imread(final_path)
    assert final_img.shape == (1080, 1920, 3), f"尺寸不正确: {final_img.shape}"
  
    logger.info(f"\n{'='*60}")
    logger.info("🎉 端到端测试通过！")
    logger.info(f"最终输出: {final_path}")
    logger.info(f"{'='*60}")
  
    return final_path


def test_compositor_quality():
    """合成质量测试"""
    logger.info("\n\n[质量测试] 合成效果评估")
  
    from src.compositor.pyramid import seamless_composite
  
    # 创建测试图像
    bg = cv2.imread(str(settings.OUTPUT_DIR / "scenes" / "test_scene.png"))
    struct = cv2.imread(
        str(settings.OUTPUT_DIR / "structures" / "aspirin_acs1996.png"),
        cv2.IMREAD_UNCHANGED,
    )
  
    if bg is None or struct is None:
        logger.warning("  测试图像不存在，跳过质量测试")
        return
  
    # 测试不同合成方法
    methods = ["direct", "lighting", "pyramid"]
    for method in methods:
        try:
            if method == "direct":
                from src.compositor.basic import alpha_composite, resize_with_alpha
                overlay_r = resize_with_alpha(struct, 300, 200)
                result = alpha_composite(bg, overlay_r, 100, 100)
            elif method == "lighting":
                result = composite_with_lighting(
                    bg, struct, 100, 100, 300, 200
                )
            elif method == "pyramid":
                result = seamless_composite(bg, struct, 100, 100, 300, 200)
          
            out_path = str(settings.OUTPUT_DIR / "final" / f"test_{method}.png")
            cv2.imwrite(out_path, result)
            logger.info(f"  ✅ {method}: {out_path}")
        except Exception as e:
            logger.error(f"  ❌ {method}: {e}")


if __name__ == "__main__":
    # 确保目录存在
    settings.ensure_dirs()
  
    # 运行流水线测试
    final_path = test_full_pipeline()
  
    # 运行质量测试
    test_compositor_quality()
  
    print(f"\n📁 所有输出保存在: {settings.OUTPUT_DIR}")
```

### 执行测试

```bash
cd chem-mindmap
python tests/test_pipeline.py
```

验证：
1. 所有步骤无报错
2. outputs/ 目录下生成所有中间文件和最终结果
3. 最终图像 visual 检查：
   - 结构图精确且清晰
   - 合成自然无 PS 感
   - 思维导图布局合理

请创建这个测试文件并运行。
```

---

## 快速参考

### 各 Phase 依赖关系图

```
Phase 0 ──▶ Phase 1 ──▶ Phase 2a ──▶ Phase 2b ──▶ Phase 2c
   │                                                     │
   │                                                     ▼
   │                                                Phase 3
   │                                                     │
   │                                                     ▼
   │                                                Phase 4
   │                                                     │
   │                                                     ▼
   │                                                Phase 5
   │                                                     │
   │                                                     ▼
   │                                                Phase 6
   │                                                     │
   └─────────────────────────────────────────────────────▼
                                                      Phase 7
```

### 每个 Phase 的输入与输出

| Phase | 依赖上游 | 输入 | 输出 |
|-------|---------|------|------|
| 0 | 无 | 无 | 项目骨架 + 依赖 |
| 1 | Phase 0 | SMILES/名称 | PNG 结构图 |
| 2a | Phase 1 | 结构图 + 场景 | 合成结果 |
| 2b | Phase 2a | 结构图 + 场景 | 光照匹配合成 |
| 2c | Phase 2b | 结构图 + 场景 | 金字塔融合 |
| 3 | Phase 1 | 树 JSON | 布局图像 |
| 4 | Phase 3 | 骨架图 | 美化场景 |
| 5 | Phase 1-4 | 用户输入 | 最终图像 |
| 6 | Phase 5 | 用户输入 | GUI 应用 |
| 7 | Phase 0-6 | 全部 | 验证结果 |

---

## 注意事项

1. **RDKit 安装**：`pip install rdkit` 可能失败，建议 `conda install -c conda-forge rdkit`
2. **API Keys**：OpenAI/Replicate/Claude 需要 API Key，存储在 `.env` 文件中
3. **SD WebUI**：如果使用本地 Stable Diffusion，需要先启动 WebUI 扩展（启用 API 和 ControlNet）
4. **内存管理**：大图合成时注意内存，建议先在 CPU 上调试通过再切换到 GPU
5. **调试模式**：所有模块都可以独立运行 + 命令行参数测试，方便定位问题

---

**祝你构建顺利！每个 Phase 完成后，请验收输出再进入下一个 Phase。**

`build.md` 文件内容如下，可直接复制到项目根目录使用：

---

# build.md — 有机化学思维导图 AI 图像生成器

> **项目目标**：构建一个 GUI 应用，用户输入大致想法（如"画一张关于阿司匹林合成路线的思维导图"），系统调用 AI 图像生成 API 生成场景图，用 RDKit 生成精确的化学结构图，自动合成输出可放入学术论文的高质量图片。

> **使用说明**：每个 Phase 在**独立对话**中执行。Agent 开始时先读本文件对应 Phase 的内容 + 上一 Phase 的输出摘要（`outputs/phase_N_summary.txt`），即可恢复上下文。200K 记忆完全够用，每个 Phase 可分配给不同 Agent。

> **Agent 启动指令模板**：
> ```
> 请先读取项目根目录的 build.md，找到 Phase N 章节。
> 然后读取 outputs/phase_N-1_summary.txt 了解上一阶段的产出。
> 按 Phase N 的「对话指令」开始执行。
> ```

---

## 项目总览

```
用户 Prompt
      │
      ▼
┌──────────────────────────────────────────────────┐
│           Phase 6: Agent 编排层 (LLM)             │
│   解析 prompt → 提取化合物列表 → 拆解子任务        │
└──┬───────────────┬───────────────┬───────────────┘
   │               │               │
   ▼               ▼               ▼
┌──────────┐ ┌──────────┐ ┌──────────────┐
│ Phase 1  │ │ Phase 3  │ │ Phase 4      │
│ 结构图   │ │ AI场景   │ │ 表面检测     │
│ 生成器   │ │ 生成器   │ │ 定位模块     │
│ (RDKit)  │ │ (SD API) │ │ (G-DINO+SAM) │
└────┬─────┘ └────┬─────┘ └──────┬───────┘
     │            │              │
     └────────────┼──────────────┘
                  ▼
         ┌────────────────┐
         │  Phase 2       │
         │  智能合成引擎   │
         │  贴图+透视+     │
         │  光照+金字塔    │
         └───────┬────────┘
                 ▼
         ┌────────────────┐
         │  Phase 7       │
         │  GUI 界面      │
         │  (Gradio)      │
         └───────┬────────┘
                 ▼
           最终输出图片
```

**技术栈**：Python 3.10+ / RDKit / OpenCV / Diffusers / Grounding DINO / SAM / Gradio

---

## 目录结构（完整形态）

```
chem-image-gen/
├── build.md                        # ← 本文件
├── pyproject.toml
├── .env.example
├── .env                            # 实际环境变量 (gitignore)
├── src/
│   ├── __init__.py
│   ├── config.py                   # Phase 0: 全局配置
│   ├── structure_gen/
│   │   ├── __init__.py
│   │   └── generator.py            # Phase 1: RDKit 结构图生成
│   ├── compositor/
│   │   ├── __init__.py
│   │   ├── basic.py                # Phase 2a: 直接贴图+透视
│   │   ├── lighting.py             # Phase 2b: 光照匹配+纹理
│   │   └── pyramid.py              # Phase 2c: 金字塔融合
│   ├── scene_gen/
│   │   ├── __init__.py
│   │   └── generator.py            # Phase 3: AI 场景生成
│   ├── detection/
│   │   ├── __init__.py
│   │   └── surface_detect.py       # Phase 4: 表面检测
│   ├── agent/
│   │   ├── __init__.py
│   │   ├── tools.py                # Phase 5a: Tool 定义
│   │   ├── orchestrator.py         # Phase 5b: 编排逻辑
│   │   └── prompts.py              # Phase 5c: 提示词
│   ├── pipeline.py                 # Phase 6: 端到端流水线
│   └── app.py                      # Phase 7: Gradio GUI
├── tests/
│   ├── __init__.py
│   ├── test_structure_gen.py
│   ├── test_compositor.py
│   ├── test_scene_gen.py
│   ├── test_detection.py
│   └── test_pipeline.py
├── outputs/
│   ├── structures/
│   ├── scenes/
│   ├── final/
│   └── phase_N_summary.txt         # 各 Phase 完成摘要
└── assets/
    └── test_images/
```

---

---

# Phase 0：环境搭建与项目骨架

## 上下文恢复

> 第一个 Phase，无需恢复任何上下文。从头开始。

## 目标

创建完整目录结构、安装全部依赖、配置环境变量、验证环境可用。

## 对话指令

```
你是一位资深 Python 工程师。请帮我从头搭建一个名为 "chem-image-gen" 的 Python 项目。

严格按照以下步骤执行：

### 步骤 1：创建目录结构

在用户当前工作目录下创建 chem-image-gen/ 并进入：

```bash
mkdir -p chem-image-gen
cd chem-image-gen
```

创建以下完整目录树（每个子目录都要有 __init__.py）：

```
chem-image-gen/
├── src/
│   ├── __init__.py
│   ├── structure_gen/
│   │   └── __init__.py
│   ├── compositor/
│   │   └── __init__.py
│   ├── scene_gen/
│   │   └── __init__.py
│   ├── detection/
│   │   └── __init__.py
│   └── agent/
│       └── __init__.py
├── tests/
│   └── __init__.py
├── outputs/
│   ├── structures/
│   ├── scenes/
│   └── final/
└── assets/
    └── test_images/
```

用 touch 命令创建所有空 __init__.py 文件（确保每个子目录都有）。

### 步骤 2：创建 pyproject.toml

```toml
[build-system]
requires = ["setuptools>=68.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "chem-image-gen"
version = "0.1.0"
description = "有机化学思维导图 AI 图像生成器"
requires-python = ">=3.10"
dependencies = [
    "opencv-python-headless>=4.8",
    "Pillow>=10.0",
    "numpy>=1.24",
    "scipy>=1.11",
    "pubchempy>=1.0.4",
    "pyyaml>=6.0",
    "python-dotenv>=1.0",
    "httpx>=0.25",
    "gradio>=4.0",
]

[project.optional-dependencies]
huggingface = [
    "torch>=2.0",
    "torchvision>=0.15",
    "diffusers>=0.25",
    "transformers>=4.36",
    "accelerate>=0.25",
    "safetensors>=0.4",
]
detection = [
    "groundingdino-py>=0.4.0",
    "segment-anything>=1.0",
]

[tool.setuptools.packages.find]
where = ["src"]
```

### 步骤 3：创建 .env.example

```
# Stable Diffusion
SD_MODEL_ID=stabilityai/stable-diffusion-xl-base-1.0
SD_USE_LOCAL=true
SD_API_URL=http://127.0.0.1:7860

# HuggingFace Token
HF_TOKEN=hf_your_token_here

# 输出目录
OUTPUT_DIR=./outputs

# 设备 (auto / cuda / mps / cpu)
DEVICE=auto

# 日志级别
LOG_LEVEL=INFO
```

### 步骤 4：创建 src/config.py

实现一个完整的 Config 单例类，要求：

- 用 `python-dotenv` 加载 `.env`（如不存在则加载 `.env.example`）
- `self.device`：自动检测，优先级 CUDA → MPS → CPU
- `self.output_dir`：Path 对象，不存在则自动创建
- `self.model_id`：从环境变量读取
- `self.hf_token`：从环境变量读取
- 支持属性式访问：`config.device`, `config.output_dir`
- 日志配置

代码示例骨架（请完善）：

```python
import os
import logging
from pathlib import Path
from dotenv import load_dotenv
import torch

load_dotenv(".env") or load_dotenv(".env.example")

class Config:
    _instance = None
  
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init()
        return cls._instance
  
    def _init(self):
        self.output_dir = Path(os.getenv("OUTPUT_DIR", "./outputs"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model_id = os.getenv("SD_MODEL_ID", "stabilityai/stable-diffusion-xl-base-1.0")
        self.hf_token = os.getenv("HF_TOKEN", "")
        self.log_level = os.getenv("LOG_LEVEL", "INFO")
      
        # 设备检测
        if os.getenv("DEVICE", "auto") != "auto":
            self.device = os.getenv("DEVICE")
        elif torch.cuda.is_available():
            self.device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
      
        logging.basicConfig(level=getattr(logging, self.log_level.upper()))
        logging.info(f"Config initialized: device={self.device}")

config = Config()
```

### 步骤 5：创建虚拟环境并安装依赖

```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
# 或 venv\Scripts\activate   # Windows

pip install --upgrade pip
pip install -e ".[huggingface,detection]"
```

### 步骤 6：特别处理 RDKit

RDKit 不能通过普通 pip 安装。请尝试以下顺序：

```bash
# 方案 1: conda（如果用户有 conda）
conda install -c conda-forge rdkit

# 方案 2: pip 非官方包
pip install rdkit-pypi

# 方案 3: 如果以上都失败，用 apt (仅 Linux)
sudo apt install python3-rdkit
```

### 步骤 7：环境验证

创建并运行测试代码，验证全部依赖：

```python
import sys

print("=== 环境验证 ===")

# 基础依赖
import cv2; print(f"✅ OpenCV {cv2.__version__}")
import numpy as np; print(f"✅ NumPy {np.__version__}")
from PIL import Image; print("✅ Pillow")
import scipy; print(f"✅ SciPy {scipy.__version__}")
import yaml; print("✅ PyYAML")
import httpx; print("✅ httpx")

# PyTorch 生态
import torch; print(f"✅ PyTorch {torch.__version__}")
print(f"   CUDA available: {torch.cuda.is_available()}")
import diffusers; print(f"✅ Diffusers {diffusers.__version__}")
import transformers; print(f"✅ Transformers {transformers.__version__}")

# 检测模型
try:
    import groundingdino; print("✅ Grounding DINO")
except ImportError:
    print("⚠️ Grounding DINO 未安装（Phase 4 需要）")

try:
    import segment_anything; print("✅ SAM")
except ImportError:
    print("⚠️ SAM 未安装（Phase 4 需要）")

# RDKit
try:
    from rdkit import Chem
    from rdkit.Chem import Draw
    mol = Chem.MolFromSmiles("c1ccccc1")
    assert mol is not None
    print("✅ RDKit: 苯环解析成功")
except ImportError:
    print("❌ RDKit 未安装！请手动执行: conda install -c conda-forge rdkit")
    print("   或: pip install rdkit-pypi")
    sys.exit(1)

# 配置
from src.config import config
print(f"✅ Config: device={config.device}, output_dir={config.output_dir}")

# Gradio
import gradio as gr; print(f"✅ Gradio {gr.__version__}")

print("\n🎉 所有依赖验证完成！Phase 0 完成。")
```

将此验证代码保存为 `tests/verify_env.py` 并运行。

### 步骤 8：生成摘要

创建 `outputs/phase_0_summary.txt`，内容包含：
- 项目根目录路径
- Python 版本
- 所有已安装包的名称和版本
- 设备类型 (CUDA/MPS/CPU)
- 任何遗留问题

完成后告知用户。
```

## 验收标准

- `python tests/verify_env.py` 全部 ✅
- `outputs/phase_0_summary.txt` 存在
- 项目目录结构符合上述规范

---

---

# Phase 1：化合物结构图生成器

## 上下文恢复

> 读取 `build.md` 了解项目总览。
> 读取 `outputs/phase_0_summary.txt` 了解环境状态。
> 本 Phase 创建 `src/structure_gen/generator.py`。

## 目标

封装 RDKit，将化合物名称或 SMILES 转换为高清、透明背景的结构图 PNG。支持多种学术风格。

## 对话指令

```
你正在构建 chem-image-gen 项目。Phase 0 已完成，环境就绪。

请读取 outputs/phase_0_summary.txt（如果存在）了解环境。

本 Phase 的目标：创建 src/structure_gen/generator.py，实现完整的 StructureGenerator 类。

## StructureGenerator 类规范

### 类属性：STYLE_PRESETS（字典常量）

```python
STYLE_PRESETS = {
    "ACS_1996": {
        "bond_line_width": 2.5,
        "font_size": 28,
        "background": (1, 1, 1, 0),    # RGBA，透明
        "atom_color": (0, 0, 0),
        "bond_color": (0, 0, 0),
        "padding_ratio": 0.08,
    },
    "dark_mode": {
        "bond_line_width": 3.0,
        "font_size": 30,
        "background": (0.08, 0.08, 0.10, 0),
        "atom_color": (0.92, 0.92, 0.92),
        "bond_color": (0.70, 0.75, 0.90),
        "padding_ratio": 0.08,
    },
    "color_on_white": {
        "bond_line_width": 2.0,
        "font_size": 26,
        "background": (1, 1, 1, 1),    # 不透明白底
        "atom_color": (0, 0, 0),
        "bond_color": (0.15, 0.15, 0.15),
        "padding_ratio": 0.06,
    },
    "minimal": {
        "bond_line_width": 1.5,
        "font_size": 22,
        "background": (1, 1, 1, 0),
        "atom_color": (0.12, 0.12, 0.12),
        "bond_color": (0.12, 0.12, 0.12),
        "padding_ratio": 0.04,
    },
}
```

### 构造方法

```python
def __init__(self, default_width=1200, default_height=800, default_style="ACS_1996"):
    self.default_width = default_width
    self.default_height = default_height
    self.default_style = default_style
    self.output_dir = Path("outputs/structures")
    self.output_dir.mkdir(parents=True, exist_ok=True)
```

### 方法 1：resolve(name: str) -> dict

通过 PubChem 解析化合物名称。

```python
def resolve(self, name: str) -> dict:
    """
    返回:
    {
        "name": "aspirin",
        "smiles": "CC(=O)Oc1ccccc1C(=O)O",
        "formula": "C9H8O4",
        "iupac_name": "2-acetyloxybenzoic acid",
        "molecular_weight": 180.16
    }
    """
```

实现逻辑：
1. 先尝试 `Chem.MolFromSmiles(name)` ——如果成功说明输入本身就是 SMILES
2. 否则调用 `pubchempy.get_compounds(name, 'name', limit=1)`
3. 如果 PubChem 也失败，抛出 `ValueError(f"无法解析化合物: {name}")`
4. 成功返回 dict

### 方法 2：generate_from_smiles(smiles, output_path=None, width=None, height=None, style=None, label=None) -> Path

核心生成方法。使用 `rdkit.Chem.Draw.MolDraw2DCairo`。

实现要点：
1. `mol = Chem.MolFromSmiles(smiles)`，若 None 抛出 ValueError
2. `AllChem.Compute2DCoords(mol)`
3. 创建 `MolDraw2DCairo(width, height)`
4. 设置 `drawOptions()` 的各项参数（bondLineWidth, fontSize 等）
5. `drawer.DrawMolecule(mol, legend=label or "")`
6. `drawer.FinishDrawing()`
7. 从 drawer.GetDrawingText() 获取 PNG bytes
8. 用 PIL 打开，转 RGBA
9. 如果 style 要求透明背景（background[3]==0）：白色像素替换为透明
   - 技巧：`np.array(img)` → 找到 RGB 都 > 240 的像素 → 将 alpha 设为 0
10. 保存到 output_path（None 则自动生成：`outputs/structures/{smiles[:20]}_{style}.png`）
11. 返回 Path 对象

### 方法 3：generate_from_name(name, output_path=None, ...)

先调用 `self.resolve(name)` 获取 SMILES，再调用 `generate_from_smiles`。

### 方法 4：generate_grid(compounds, cols=2, width=None, height=None, style=None, labels=None) -> Path

将多个化合物拼接成网格。

参数：
- compounds: SMILES 字符串列表
- cols: 每行列数
- labels: 可选标签列表

用 PIL 的 `Image.new` + `Image.paste` 拼接。

### 命令行入口

```python
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("compound", help="化合物名称或 SMILES")
    p.add_argument("--style", default="ACS_1996")
    p.add_argument("--output", "-o", default=None)
    p.add_argument("--width", type=int, default=1200)
    p.add_argument("--height", type=int, default=800)
    args = p.parse_args()
  
    gen = StructureGenerator(args.width, args.height, args.style)
    path = gen.generate_from_name(args.compound, args.output)
    print(f"✅ 结构图已生成: {path}")
```

## 同时创建 tests/test_structure_gen.py

测试以下内容（用 assert 语句，每个测试独立）：

1. **test_smiles_generation**：从 SMILES "CC(=O)Oc1ccccc1C(=O)O" 生成，验证文件存在
2. **test_name_generation**：从名称 "caffeine" 生成，验证文件存在
3. **test_all_styles**：四种风格各生成一张，全部成功
4. **test_transparent_bg**：验证 ACS_1996 风格生成的图片有 alpha 通道且部分像素透明
5. **test_invalid_smiles**：传入 "NOT_A_VALID_SMILES!!!"，确认抛出 ValueError
6. **test_grid**：generate_grid(["c1ccccc1", "CC(=O)Oc1ccccc1C(=O)O"], labels=["Benzene", "Aspirin"])，验证网格图存在

每项打印 ✅ 或 ❌，最后统计。

## 完成后执行

1. `python tests/test_structure_gen.py` ——确保全部通过
2. 手动生成三张示例图：
   ```bash
   python src/structure_gen/generator.py "aspirin" --style ACS_1996 -o outputs/structures/aspirin_ACS1996.png
   python src/structure_gen/generator.py "caffeine" --style dark_mode -o outputs/structures/caffeine_dark.png
   ```
3. 创建 `outputs/phase_1_summary.txt` 记录：
   - 创建了哪些文件
   - StructureGenerator 的关键接口
   - 测试结果 (N/N 通过)
   - 遗留问题

请开始实现，代码要完整可运行。
```

## 验收标准

- `python tests/test_structure_gen.py` 全部 ✅
- `outputs/structures/aspirin_ACS1996.png` 和 `outputs/structures/caffeine_dark.png` 存在
- 透明背景图在暗色浏览器/查看器中也能正常显示

---

---

# Phase 2：智能合成引擎（分 3 个子阶段执行）

## ⚠️ 重要

本 Phase 代码量大，**需要分 3 次独立对话**。每个子阶段独立可测试。

| 子阶段 | 产出文件 | 核心功能 | 预估对话数 |
|--------|---------|---------|-----------|
| 2a | `compositor/basic.py` | 直接贴图、透视变换、alpha 羽化 | 1 |
| 2b | `compositor/lighting.py` | 色彩匹配、纹理叠加、投影 | 1 |
| 2c | `compositor/pyramid.py` | 拉普拉斯金字塔多尺度融合 | 1 |

---

## Phase 2a：基础合成

### 上下文恢复

> Phase 1 完成。读取 `outputs/phase_1_summary.txt` 了解结构图生成器。
> 本子阶段创建 `src/compositor/basic.py`。

### 对话指令

```
Phase 1 完成。构建 Phase 2a：基础合成器。

请创建 src/compositor/basic.py，实现以下功能。

## 1. load_image_rgba(path) -> np.ndarray

```python
def load_image_rgba(path):
    """加载图片，返回 RGBA (H,W,4) uint8。
    cv2 加载为 BGR，需转为 RGB。
    无 alpha 通道时补全为 255。"""
```

## 2. alpha_composite(background, foreground, position, feather=2) -> np.ndarray

将前景（RGBA）合成到背景（RGB）上。

参数：
- background: (H, W, 3) numpy uint8
- foreground: (H_fg, W_fg, 4) numpy uint8 (RGBA)
- position: (x, y, w, h) — 目标区域在上图中的位置和大小
- feather: 羽化半径（对 alpha 做 GaussianBlur）

实现：
- 前景 resize 到 (w, h)
- 提取 alpha 通道，GaussianBlur 羽化
- `result = bg * (1-alpha) + fg_rgb * alpha`
- 注意处理 position 越界

## 3. perspective_composite(background, foreground, dst_corners, feather=2) -> np.ndarray

透视变换合成。

参数：
- dst_corners: (4,2) — 背景中目标四边形的四角坐标
  - 顺序：左上、右上、右下、左下

实现：
- 源角点 = [(0,0), (w_fg-1,0), (w_fg-1,h_fg-1), (0,h_fg-1)]
- `M = cv2.getPerspectiveTransform(src, dst)`
- `cv2.warpPerspective(fg, M, (bg_w, bg_h))`
- 同时变换 alpha 通道，羽化后合成

## 4. smart_composite(background, foreground, region, method="auto") -> np.ndarray

便捷函数，自动选择方法。

region 可以是：
- `(x, y, w, h)` 矩形 → 走 direct
- `[(x1,y1),(x2,y2),(x3,y3),(x4,y4)]` 四边形 → 走 perspective
- `(x, y, w, h, rotation_deg)` 带旋转 → 转四边形后走 perspective

## 测试

创建 tests/test_compositor.py（只测 Phase 2a 部分）：

```python
def test_direct_composite():
    """创建模拟白板背景 + 结构图 → 验证合成成功"""
    # 加载 Phase 1 生成的 aspirin_ACS1996.png
    # 创建 800x600 的白色背景，画灰色边框模拟白板
    # 调用 alpha_composite
    # 保存到 outputs/final/test_direct.png
    # 验证输出文件存在、尺寸正确
    pass

def test_perspective_composite():
    """测试透视变换合成"""
    # 同上，但用倾斜的四边形
    # 保存到 outputs/final/test_perspective.png
    pass
```

## 完成后

写入 outputs/phase_2a_summary.txt。

请开始实现，所有代码要完整可运行。
```

---

## Phase 2b：光照匹配与纹理融合

### 上下文恢复

> Phase 2a 完成，`compositor/basic.py` 可用。
> 本子阶段创建 `src/compositor/lighting.py`。

### 对话指令

```
Phase 2a 完成。构建 Phase 2b：光照匹配。

创建 src/compositor/lighting.py，实现以下函数。

## 1. color_transfer(src, target, mask=None) -> np.ndarray

将 src 的色彩分布匹配到 target。

算法（Reinhard 色彩迁移，简化版）：
- src_lab = cv2.cvtColor(src, cv2.COLOR_RGB2Lab).astype(float)
- target_lab = cv2.cvtColor(target, cv2.COLOR_RGB2Lab).astype(float)
- 如果 mask 不为 None，仅对 mask 内像素计算统计量
- 对每个通道：`result = (src - src_mean) * (target_std / src_std) + target_mean`
- 转回 RGB，clip [0,255]
- 返回 uint8

## 2. extract_texture(image, sigma=8) -> np.ndarray

提取图像表面纹理（高频成分）。

- `blurred = cv2.GaussianBlur(image, (0,0), sigma)`
- `texture = image.astype(float) - blurred.astype(float)`
- 归一化到 [-30, 30] 范围
- 返回 float 数组

## 3. apply_texture(struct, texture, strength=0.1) -> np.ndarray

- `result = struct.astype(float) + texture * strength`
- clip [0,255]，返回 uint8

## 4. generate_shadow(alpha, offset=(3,3), blur=5, opacity=0.25) -> np.ndarray

根据 alpha mask 生成投影。

- 创建全零画布（尺寸 = alpha 尺寸 + offset）
- 将 alpha 放到偏移位置
- GaussianBlur
- 乘以 opacity
- 返回 float（单通道）

## 5. apply_shadow(background, shadow, position) -> np.ndarray

将投影叠加到背景上。
- 提取背景对应区域
- `result = bg_region * (1 - shadow) + 0 * shadow`（投影 = 变暗）
- 返回修改后的背景

## 6. match_and_blend(background, foreground, position, 
                   color_match=True, texture_blend=0.06, 
                   shadow=True, feather=3) -> np.ndarray

一体化函数，组合以上所有步骤。这是外部调用的主要接口。

## 测试

在 tests/test_compositor.py 中添加：

```python
def test_lighting_match():
    """对比：不做光照匹配 vs 做光照匹配"""
    # 加载真实的白板照片（如果没有，用渐变模拟）
    # 加载结构图
    # 分别用 plain alpha_composite 和 match_and_blend
    # 保存对比图
    pass
```

完成后写入 outputs/phase_2b_summary.txt。
```

---

## Phase 2c：金字塔融合

### 上下文恢复

> Phase 2a、2b 完成。
> 本子阶段创建 `src/compositor/pyramid.py`。

### 对话指令

```
Phase 2a、2b 完成。构建 Phase 2c：金字塔融合。

创建 src/compositor/pyramid.py。

## 实现以下函数

### 1. gaussian_pyramid(image, levels=4) -> list

```python
def gaussian_pyramid(image, levels=4):
    """返回从原始尺寸到最粗糙级别的高斯金字塔。
    pyr[i+1] = cv2.pyrDown(pyr[i])"""
```

### 2. laplacian_pyramid(gp) -> list

```python
def laplacian_pyramid(gp):
    """从高斯金字塔构建拉普拉斯金字塔。
    lp[i] = gp[i] - cv2.pyrUp(gp[i+1])
    lp[-1] = gp[-1]  # 最顶层保持不变"""
```

### 3. pyramid_blend(bg, fg, mask, levels=4) -> np.ndarray

核心算法：

```
gp_bg = gaussian_pyramid(bg, levels)
gp_fg = gaussian_pyramid(fg, levels)
gp_mask = gaussian_pyramid(mask, levels)

lp_bg = laplacian_pyramid(gp_bg)
lp_fg = laplacian_pyramid(gp_fg)

# 每层融合
blended = []
for i in range(levels):
    m = gp_mask[i] / 255.0  # 归一化 mask
    m_3c = np.dstack([m]*3) if m.ndim==2 else m
    blended.append(lp_fg[i] * m_3c + lp_bg[i] * (1 - m_3c))

# 重建
result = blended[-1]
for i in range(levels-2, -1, -1):
    result = cv2.pyrUp(result, dstsize=(blended[i].shape[1], blended[i].shape[0]))
    result = result + blended[i]

return np.clip(result, 0, 255).astype(np.uint8)
```

### 4. seamless_composite(bg_path_or_array, fg_path, region) -> np.ndarray

最高层接口。内部调用：
1. 加载图像
2. 从 fg 提取 alpha 作为 mask
3. 调用 pyramid_blend
4. 返回合成结果

## 测试

在 tests/test_compositor.py 中添加：

```python
def test_pyramid_vs_direct():
    """对比 pyramid 融合 vs 直接 alpha 合成的效果"""
    pass
```

## Phase 2 总完成

创建 outputs/phase_2_summary.txt，汇总 Phase 2 全部三个子阶段：
- compositor/basic.py：直接合成 + 透视合成
- compositor/lighting.py：光照匹配 + 纹理 + 投影
- compositor/pyramid.py：金字塔融合
- 测试结果

请开始实现 Phase 2c。
```

---

---

# Phase 3：AI 场景生成器

## 上下文恢复

> Phase 1、Phase 2 完成。
> 读取 `outputs/phase_1_summary.txt` 和 `outputs/phase_2_summary.txt`。
> 本 Phase 创建 `src/scene_gen/generator.py`。

## 对话指令

```
Phase 1 和 2 完成。构建 Phase 3：AI 场景生成器。

创建 src/scene_gen/generator.py，实现 SceneGenerator 类。

## SceneGenerator 类

### 构造方法

```python
def __init__(self, model_id=None, device=None, use_local=True):
```

- model_id: None 则从 config 读取 SD_MODEL_ID（默认 stabilityai/stable-diffusion-xl-base-1.0）
- device: None 则从 config 读取
- use_local: True 用 diffusers 本地推理，False 用 HTTP API

### 方法 1：load_model()

使用 diffusers 加载 Stable Diffusion XL pipeline：

```python
def load_model(self):
    from diffusers import StableDiffusionXLPipeline
  
    pipe = StableDiffusionXLPipeline.from_pretrained(
        self.model_id,
        torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        use_safetensors=True,
        token=self.hf_token or True,
    )
  
    if self.device == "cuda":
        pipe.enable_model_cpu_offload()
    else:
        pipe = pipe.to(self.device)
  
    pipe.enable_vae_slicing()
    self.pipe = pipe
```

**重要**：如果模型下载失败或无 GPU，提供降级方案——返回纯色背景 + 提示文字，让后续流程能跑通。

### 方法 2：generate(prompt, negative_prompt=None, width=1024, height=1024, steps=30, guidance=7.5, seed=None) -> PIL.Image

默认 negative_prompt:
```
"blurry, low quality, distorted, warped, ugly, text, watermark, 
chemical structures, molecular diagrams, incorrect science"
```

### 方法 3：generate_scene_with_surface(scene_desc, surface="whiteboard", position="center", style="photorealistic", **kwargs) -> PIL.Image

**这是核心方法**。构建引导性 prompt 模板：

```
f"{scene_desc}, featuring a clearly visible rectangular {surface} with 
clean empty surface, positioned {position} in frame, {style}, 
straight-on angle, well-lit, sharp focus, 8k, professional photography"
```

surface 选项：whiteboard, computer_screen, paper_sheet, glass_board, chalkboard
position 选项：center, left, right, full_frame, upper_half
style 选项：photorealistic, scientific_illustration, warm_lab, clean_minimal

### 方法 4：generate_via_api(prompt, **kwargs) -> PIL.Image

通过 HTTP API 调用（用于远程 Stable Diffusion WebUI）：

```python
def generate_via_api(self, prompt, **kwargs):
    import httpx
    payload = {
        "prompt": prompt,
        "negative_prompt": kwargs.get("negative_prompt", ""),
        "width": kwargs.get("width", 1024),
        "height": kwargs.get("height", 1024),
        "steps": kwargs.get("steps", 30),
        "cfg_scale": kwargs.get("guidance", 7.5),
        "seed": kwargs.get("seed", -1),
    }
    resp = httpx.post(
        f"{self.api_url}/sdapi/v1/txt2img",
        json=payload,
        timeout=120
    )
    resp.raise_for_status()
    data = resp.json()
    # base64 decode
    import base64, io
    img_bytes = base64.b64decode(data["images"][0])
    return Image.open(io.BytesIO(img_bytes))
```

### 测试

创建 tests/test_scene_gen.py：

```python
def test_scene_generation():
    """测试场景生成"""
    gen = SceneGenerator(use_local=False)  # 如果无 GPU 先测试 API 模式
    img = gen.generate_scene_with_surface(
        "modern chemistry laboratory",
        surface="whiteboard",
    )
    img.save("outputs/scenes/test_lab_whiteboard.png")
    print("✅ 场景图生成成功")

def test_prompt_template():
    """验证 prompt 模板正确生成"""
    gen = SceneGenerator(use_local=False)
    prompt = gen._build_guided_prompt(
        "a scientist's desk",
        surface="whiteboard",
        position="center",
    )
    assert "whiteboard" in prompt
    assert "rectangular" in prompt.lower()
    print(f"✅ Prompt 模板: {prompt[:100]}...")
```

## 完成后

写入 outputs/phase_3_summary.txt。

**特别说明**：如果用户无 GPU，标注清楚并提供 mock 模式。告知用户可以通过 Replicate API 或本地 WebUI 使用 SD。
```

---

---

# Phase 4：表面检测与定位

## 上下文恢复

> Phase 1-3 完成。读取各自的 summary。
> 本 Phase 创建 `src/detection/surface_detect.py`。

## 对话指令

```
Phase 1-3 完成。构建 Phase 4：表面检测模块。

创建 src/detection/surface_detect.py。

## SurfaceDetector 类

目标：在场景图中自动检测可放置结构图的矩形表面（白板、屏幕、纸张）。

### 构造方法

```python
def __init__(self, device=None, use_sam=True):
```

加载 Grounding DINO 和 SAM（如果 use_sam=True）。

```python
# Grounding DINO
from groundingdino.util.inference import Model as GDModel
self.gd_model = GDModel(
    model_config_path="GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
    model_checkpoint_path="groundingdino_swint_ogc.pth",
    device=self.device,
)

# SAM
from segment_anything import sam_model_registry, SamPredictor
sam = sam_model_registry["vit_h"](
    checkpoint="sam_vit_h_4b8939.pth"
)
sam.to(self.device)
self.sam_predictor = SamPredictor(sam)
```

**重要**：如果模型未下载，提供清晰的下载指令和 fallback 方案。

### 方法 1：detect_surfaces(image, text_prompt=None) -> list[dict]

```python
def detect_surfaces(self, image, text_prompt=None):
    """
    检测所有矩形表面。
  
    text_prompt 默认: "whiteboard . screen . paper . rectangle . board"
  
    返回列表，每项:
    {
        "label": "whiteboard",
        "bbox": (x, y, w, h),
        "confidence": 0.92,
        "corners": [(x1,y1), (x2,y2), (x3,y3), (x4,y4)],  # 四角点
        "mask": np.ndarray,        # SAM 分割 mask (H,W) bool
        "area_ratio": 0.35,        # 占画面比例
    }
    """
```

实现流程：
1. Grounding DINO 检测目标（box_threshold=0.35, text_threshold=0.25）
2. 对每个检测到的 box：
   - SAM 分割得到精细 mask
   - 从 mask 提取轮廓：`cv2.findContours` + `cv2.approxPolyDP`
   - 如果是四边形 → 取四角点
   - 否则 → 取最小外接矩形的四角点
3. 按面积降序排序
4. 返回

### 方法 2：detect_best_surface(image, preferred="whiteboard") -> dict | None

返回最适合放置结构图的表面（面积最大、最方正、置信度最高的匹配项）。

### 方法 3：get_placement_region(surface_dict, padding=0.05) -> dict

```python
def get_placement_region(self, surface_dict, padding=0.05):
    """
    返回放置区域信息：
    {
        "type": "perspective",  # "direct" 或 "perspective"
        "region": (x, y, w, h),
        "corners": [(x1,y1), ...],  # 内缩 padding 后的角点
        "transform_matrix": M,      # 3x3 透视变换矩阵
    }
    """
```

### Fallback 方案

如果 Grounding DINO/SAM 不可用：

```python
def _fallback_detect(self, image):
    """用传统 CV 方法检测矩形"""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100)
    # 合并线段找四边形
    # 返回最可能的矩形区域
```

### 测试

创建 tests/test_detection.py：

```python
def test_surface_detection():
    """对 Phase 3 生成的场景图检测白板"""
    detector = SurfaceDetector(use_sam=False)  # 先用 fallback
    img = Image.open("outputs/scenes/test_lab_whiteboard.png")
    surfaces = detector.detect_surfaces(np.array(img))
  
    if surfaces:
        # 可视化
        import cv2
        vis = np.array(img).copy()
        for s in surfaces:
            cv2.rectangle(vis, (s["bbox"][0], s["bbox"][1]),
                         (s["bbox"][0]+s["bbox"][2], s["bbox"][1]+s["bbox"][3]),
                         (0,255,0), 3)
        cv2.imwrite("outputs/final/detection_vis.png", cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
        print(f"✅ 检测到 {len(surfaces)} 个表面")
    else:
        print("⚠️ 未检测到表面，fallback 使用全图中央区域")
```

## 完成后

写入 outputs/phase_4_summary.txt。
```

---

---

# Phase 5：Agent 编排层（分 2 个子阶段）

---

## Phase 5a：Tool 定义

### 上下文恢复

> Phase 1-4 完成。
> 本子阶段创建 `src/agent/tools.py` 和 `src/agent/prompts.py`。

### 对话指令

```
Phase 1-4 完成。构建 Phase 5a：Tool 定义和提示词。

## 任务 1：创建 src/agent/prompts.py

```python
# src/agent/prompts.py

SYSTEM_PROMPT = """你是一个有机化学图像生成专家。

用户会描述他们想要的化学思维导图或包含化学结构的场景图。你的工作是：

1. 仔细解析用户输入，识别所有提到的有机化合物名称
2. 理解用户想要的场景（白板/纸张/屏幕/海报等）
3. 规划布局：哪些化合物放在场景的什么位置
4. 按顺序调用工具完成生成

## 可用工具
- resolve_compound: 通过化合物名称获取精确的 SMILES 和分子信息
- generate_structure: 生成化合物的精确 2D 结构图（PNG，透明背景）
- generate_scene: 生成场景主图（含白板/纸张等可书写表面）
- detect_surface: 在场景图中检测可放置结构图的矩形表面
- composite_image: 将结构图自然合成到场景图中的指定位置

## 工作流程
1. 首先调用 resolve_compound 解析所有化合物（可并行调用）
2. 然后并行调用 generate_structure（所有化合物）和 generate_scene（场景）
3. 接着调用 detect_surface 定位放置面
4. 最后对每个化合物调用 composite_image 合成

## 重要规则
- 化学结构必须精确！永远不要用 AI 生成结构式
- 始终使用工具生成精确结构图，然后合成到场景中
- 如果用户提到的化合物名称不明确，先确认再操作
- 输出图像的风格应符合学术论文要求
"""

# 预设示例
EXAMPLES = [
    "实验室白板上画着阿司匹林的结构式，旁边放着一杯咖啡",
    "一张海报展示咖啡因和茶碱的化学结构对比",
    "化学家笔记本上画着布洛芬合成路线的思维导图",
    "现代实验室里，玻璃板上用马克笔写着紫杉醇的分子结构",
]
```

## 任务 2：创建 src/agent/tools.py

实现所有 Tool 的定义和执行函数。

### Tool 定义（OpenAI function-calling 格式）

```python
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "resolve_compound",
            "description": "解析有机化合物名称，返回 SMILES、分子式、IUPAC 名称等信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "compound_name": {
                        "type": "string",
                        "description": "化合物名称，如 'aspirin', 'caffeine', '2,4-dinitrotoluene'"
                    }
                },
                "required": ["compound_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "generate_structure",
            "description": "生成化合物的精确 2D 化学结构图",
            "parameters": {
                "type": "object",
                "properties": {
                    "smiles": {"type": "string", "description": "化合物的 SMILES 字符串"},
                    "style": {
                        "type": "string",
                        "enum": ["ACS_1996", "dark_mode", "color_on_white", "minimal"],
                        "default": "ACS_1996"
                    },
                    "label": {"type": "string", "description": "可选：在结构图下方显示的标签"},
                    "output_path": {"type": "string", "description": "可选：输出文件路径"}
                },
                "required": ["smiles"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "generate_scene",
            "description": "生成场景主图（专业摄影风格，含可书写表面）",
            "parameters": {
                "type": "object",
                "properties": {
                    "scene_description": {"type": "string"},
                    "surface_type": {
                        "type": "string",
                        "enum": ["whiteboard", "screen", "paper", "glass_board", "chalkboard"],
                        "default": "whiteboard"
                    },
                    "style": {
                        "type": "string", 
                        "enum": ["photorealistic", "scientific_illustration", "warm_lab"],
                        "default": "photorealistic"
                    }
                },
                "required": ["scene_description"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "detect_surface",
            "description": "在场景图中检测可放置结构图的矩形表面（白板/屏幕/纸张）",
            "parameters": {
                "type": "object",
                "properties": {
                    "image_path": {"type": "string", "description": "场景图的路径"},
                    "preferred_surface": {
                        "type": "string",
                        "default": "whiteboard"
                    }
                },
                "required": ["image_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "composite_image",
            "description": "将化学结构图合成到场景图中的指定位置",
            "parameters": {
                "type": "object",
                "properties": {
                    "background_path": {"type": "string"},
                    "structure_path": {"type": "string"},
                    "region": {
                        "type": "object",
                        "description": "放置区域，可包含 x, y, w, h 或 corners 数组",
                        "properties": {
                            "x": {"type": "number"},
                            "y": {"type": "number"},
                            "w": {"type": "number"},
                            "h": {"type": "number"},
                            "corners": {"type": "array"}
                        }
                    },
                    "method": {
                        "type": "string",
                        "enum": ["auto", "direct", "perspective", "pyramid"],
                        "default": "pyramid"
                    },
                    "color_match": {"type": "boolean", "default": True}
                },
                "required": ["background_path", "structure_path", "region"]
            }
        }
    },
]
```

### Tool 执行函数

```python
from src.structure_gen.generator import StructureGenerator
from src.scene_gen.generator import SceneGenerator
from src.detection.surface_detect import SurfaceDetector
from src.compositor.pyramid import seamless_composite
from src.compositor.lighting import match_and_blend

structure_gen = StructureGenerator()
scene_gen = SceneGenerator()
surface_detector = SurfaceDetector()

def execute_resolve_compound(args):
    info = structure_gen.resolve(args["compound_name"])
    return {"success": True, "data": info}

def execute_generate_structure(args):
    path = structure_gen.generate_from_smiles(
        args["smiles"],
        output_path=args.get("output_path"),
        style=args.get("style", "ACS_1996"),
        label=args.get("label"),
    )
    return {"success": True, "path": str(path)}

def execute_generate_scene(args):
    path = scene_gen.generate_scene_with_surface(
        args["scene_description"],
        surface=args.get("surface_type", "whiteboard"),
        style=args.get("style", "photorealistic"),
    )
    return {"success": True, "path": str(path)}

def execute_detect_surface(args):
    from PIL import Image
    img = Image.open(args["image_path"])
    best = surface_detector.detect_best_surface(
        np.array(img),
        preferred=args.get("preferred_surface", "whiteboard")
    )
    if best:
        return {"success": True, "surface": best}
    else:
        # fallback: 使用图像中央 60% 区域
        w, h = img.size
        return {
            "success": True,
            "surface": {
                "label": "fallback_center",
                "bbox": (int(w*0.2), int(h*0.15), int(w*0.6), int(h*0.6)),
                "corners": None,
            }
        }

def execute_composite_image(args):
    region = args["region"]
    method = args.get("method", "pyramid")
  
    if "corners" in region and region["corners"]:
        from src.compositor.basic import perspective_composite
        bg = load_image_rgba(args["background_path"])
        fg = load_image_rgba(args["structure_path"])
        result = perspective_composite(bg[:,:,:3], fg, region["corners"])
        # 后续加光照匹配
    else:
        x, y, w, h = region["x"], region["y"], region["w"], region["h"]
        result = match_and_blend(
            args["background_path"],
            args["structure_path"],
            (x, y, w, h),
            color_match=args.get("color_match", True),
        )
  
    output_path = args.get("output_path", "outputs/final/result.png")
    save_image(result, output_path)
    return {"success": True, "path": output_path}

# 执行器映射表
TOOL_EXECUTORS = {
    "resolve_compound": execute_resolve_compound,
    "generate_structure": execute_generate_structure,
    "generate_scene": execute_generate_scene,
    "detect_surface": execute_detect_surface,
    "composite_image": execute_composite_image,
}
```

完成后写入 outputs/phase_5a_summary.txt。
```

---

## Phase 5b：编排器

### 上下文恢复

> Phase 5a 完成。读取 `outputs/phase_5a_summary.txt`。
> 本子阶段创建 `src/agent/orchestrator.py`。

### 对话指令

```
Phase 5a 完成。构建 Phase 5b：Agent 编排器。

创建 src/agent/orchestrator.py。

## AgentOrchestrator 类

```python
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from PIL import Image

@dataclass
class GenerationResult:
    final_image: Image.Image = None
    final_path: Path = None
    compounds: list = field(default_factory=list)
    workflow_log: list = field(default_factory=list)
    elapsed_time: float = 0.0
    success: bool = False
    error: str = ""

class AgentOrchestrator:
    """
    编排整个生成流程。
  
    通过 LLM 的 function-calling 能力自动决定调用哪些工具。
    支持 OpenAI 和 Anthropic 两种 LLM 后端。
  
    使用方式：
        orch = AgentOrchestrator(llm_backend="openai", api_key="sk-...")
        result = orch.run("画一张阿司匹林合成路线的思维导图")
        result.final_image.save("output.png")
    """
  
    def __init__(self, llm_backend="openai", api_key=None, model=None):
        self.backend = llm_backend  # "openai" 或 "anthropic"
        self.api_key = api_key
        self.model = model or ("gpt-4o" if llm_backend == "openai" else "claude-sonnet-4-20250514")
        self.max_iterations = 12
        self.tools = TOOLS
        self.executors = TOOL_EXECUTORS
  
    def run(self, user_prompt: str) -> GenerationResult:
        start_time = time.time()
        result = GenerationResult()
      
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]
      
        try:
            for iteration in range(self.max_iterations):
                # 调用 LLM
                response = self._call_llm(messages)
              
                if self._has_tool_calls(response):
                    # 执行工具调用
                    for tc in self._get_tool_calls(response):
                        tool_name = tc["name"]
                        tool_args = tc["args"]
                      
                        log_entry = {"tool": tool_name, "args": tool_args}
                      
                        if tool_name in self.executors:
                            tool_result = self.executors[tool_name](tool_args)
                            log_entry["result"] = tool_result
                            result.workflow_log.append(log_entry)
                          
                            # 将结果反馈给 LLM
                            messages.append(self._assistant_message(response))
                            messages.append(self._tool_result_message(tc["id"], tool_result))
                        else:
                            log_entry["result"] = {"success": False, "error": f"未知工具: {tool_name}"}
                            result.workflow_log.append(log_entry)
                else:
                    # LLM 不再调用工具，流程结束
                    result.workflow_log.append({
                        "type": "llm_response",
                        "content": self._get_text_content(response)
                    })
                    break
          
            # 从日志提取最终图片路径
            self._extract_result(result)
          
        except Exception as e:
            result.success = False
            result.error = str(e)
      
        result.elapsed_time = time.time() - start_time
        return result
  
    def _call_llm(self, messages):
        """调用 LLM API，适配 OpenAI / Anthropic"""
        # 实现细节由 Agent 根据所选后端补充
        pass
  
    def _has_tool_calls(self, response) -> bool:
        """检查响应是否包含 tool calls"""
        pass
  
    def _get_tool_calls(self, response) -> list[dict]:
        """提取 tool calls"""
        pass
  
    def _extract_result(self, result: GenerationResult):
        """从 workflow_log 中提取最终图片"""
        for log in reversed(result.workflow_log):
            if log.get("tool") == "composite_image":
                path = log.get("result", {}).get("path")
                if path and Path(path).exists():
                    result.final_path = Path(path)
                    result.final_image = Image.open(path)
                    result.success = True
                    break
```

## 关键实现细节

### OpenAI 适配

```python
def _call_llm_openai(self, messages):
    from openai import OpenAI
    client = OpenAI(api_key=self.api_key)
    response = client.chat.completions.create(
        model=self.model,
        messages=messages,
        tools=self.tools,
        tool_choice="auto",
    )
    return response.choices[0]
```

### Anthropic 适配

```python
def _call_llm_anthropic(self, messages):
    import anthropic
    client = anthropic.Anthropic(api_key=self.api_key)
  
    # 转换消息格式
    system_msg = [m for m in messages if m["role"] == "system"]
    chat_msgs = [m for m in messages if m["role"] != "system"]
  
    # 转换 tools 格式
    anthropic_tools = self._convert_tools_to_anthropic()
  
    response = client.messages.create(
        model=self.model,
        system=system_msg[0]["content"] if system_msg else "",
        messages=chat_msgs,
        tools=anthropic_tools,
        max_tokens=4096,
    )
    return response
```

## 测试

```python
# 简易测试：直接调用（不经过 LLM）
def test_direct_orchestration():
    orch = AgentOrchestrator()
    # 手动模拟 tool calls 序列
    # 1. resolve "aspirin"
    r1 = execute_resolve_compound({"compound_name": "aspirin"})
    assert r1["success"]
  
    # 2. generate_structure
    r2 = execute_generate_structure({"smiles": r1["data"]["smiles"]})
    assert r2["success"]
  
    # 3. generate_scene
    r3 = execute_generate_scene({"scene_description": "lab whiteboard"})
    assert r3["success"]
  
    # 4. detect_surface
    r4 = execute_detect_surface({"image_path": r3["path"]})
    assert r4["success"]
  
    # 5. composite
    surf = r4["surface"]
    r5 = execute_composite_image({
        "background_path": r3["path"],
        "structure_path": r2["path"],
        "region": {
            "x": surf["bbox"][0], "y": surf["bbox"][1],
            "w": surf["bbox"][2], "h": surf["bbox"][3],
        }
    })
    assert r5["success"]
    print("✅ 直接编排测试通过")
```

完成后写入 outputs/phase_5_summary.txt。
```

---

---

# Phase 6：端到端流水线

## 上下文恢复

> Phase 1-5 全部完成。读取各 Phase summary。
> 本 Phase 创建 `src/pipeline.py`。

## 对话指令

```
Phase 1-5 完成。构建 Phase 6：端到端流水线。

创建 src/pipeline.py，实现 ChemicalImagePipeline 类。

## 目标

提供最高层的一行代码 API：

```python
pipeline = ChemicalImagePipeline()
result = pipeline.generate("实验室白板上画着阿司匹林的结构式")
result.save("output.png")
```

## ChemicalImagePipeline 类

```python
@dataclass
class PipelineResult:
    final_image: Image.Image
    final_path: Path
    compounds: list[dict]
    scene_path: Path
    structure_paths: list[Path]
    workflow_log: list[dict]
    elapsed_time: float
  
    def save(self, path):
        self.final_image.save(path)
  
    def summary(self) -> str:
        lines = [
            f"✅ 生成完成！",
            f"化合物: {len(self.compounds)} 个",
            f"耗时: {self.elapsed_time:.1f} 秒",
            f"输出: {self.final_path}",
        ]
        return "\n".join(lines)

class ChemicalImagePipeline:
    def __init__(self, structure_style="ACS_1996", scene_style="photorealistic",
                 composite_method="pyramid", llm_backend=None, api_key=None):
        self.structure_style = structure_style
        self.scene_style = scene_style
        self.composite_method = composite_method
      
        self.structure_gen = StructureGenerator(default_style=structure_style)
        self.scene_gen = SceneGenerator()
        self.detector = SurfaceDetector()
      
        # 尝试初始化 Agent，如果不可用则用简化流程
        if llm_backend:
            self.orchestrator = AgentOrchestrator(
                llm_backend=llm_backend, api_key=api_key
            )
        else:
            self.orchestrator = None
  
    def generate(self, prompt: str) -> PipelineResult:
        start = time.time()
      
        if self.orchestrator:
            # 使用 Agent 编排
            return self._generate_with_agent(prompt, start)
        else:
            # 简化流程：正则提取化合物 + 默认布局
            return self._generate_simple(prompt, start)
  
    def _generate_with_agent(self, prompt, start) -> PipelineResult:
        """使用 LLM Agent 编排完整流程"""
        agent_result = self.orchestrator.run(prompt)
      
        return PipelineResult(
            final_image=agent_result.final_image,
            final_path=agent_result.final_path,
            compounds=agent_result.compounds,
            scene_path=Path(
                agent_result.workflow_log中提取的scene_path
            ),
            structure_paths=[
                Path(log["result"]["path"])
                for log in agent_result.workflow_log
                if log.get("tool") == "generate_structure"
            ],
            workflow_log=agent_result.workflow_log,
            elapsed_time=time.time() - start,
        )
  
    def _generate_simple(self, prompt, start) -> PipelineResult:
        """简化流程：不依赖 LLM，用正则表达式提取化合物名称"""
        import re
      
        # 1. 简单提取化合物名称（基于大写字母开头的单词模式）
        # 实际实现中需要更复杂的 NER 或化学词典匹配
        compounds = self._extract_compound_names(prompt)
      
        # 2. 解析并生成结构图
        structure_paths = []
        compounds_info = []
        for name in compounds:
            try:
                info = self.structure_gen.resolve(name)
                compounds_info.append(info)
                path = self.structure_gen.generate_from_smiles(
                    info["smiles"], style=self.structure_style
                )
                structure_paths.append(path)
            except Exception as e:
                logging.warning(f"跳过 {name}: {e}")
      
        # 3. 生成场景
        scene_path = self.scene_gen.generate_scene_with_surface(
            "modern chemistry laboratory",
            style=self.scene_style,
        )
      
        # 4. 检测表面
        scene_img = np.array(Image.open(scene_path))
        best_surface = self.detector.detect_best_surface(scene_img)
      
        # 5. 合成
        final = scene_path
        n = len(structure_paths)
        for i, sp in enumerate(structure_paths):
            if best_surface:
                # 将表面等分为 n 份
                x, y, w, h = best_surface["bbox"]
                region_w = w // n
                region = (x + i * region_w, y, region_w, h)
            else:
                # 默认居中排列
                region = (200 + i * 400, 200, 350, 300)
          
            output = f"outputs/final/temp_{i}.png"
            from src.compositor.lighting import match_and_blend
            match_and_blend(
                str(final), str(sp), region,
                color_match=True,
            )
            final = output
      
        return PipelineResult(
            final_image=Image.open(final),
            final_path=Path(final),
            compounds=compounds_info,
            scene_path=scene_path,
            structure_paths=structure_paths,
            workflow_log=[],
            elapsed_time=time.time() - start,
        )
  
    def _extract_compound_names(self, prompt: str) -> list[str]:
        """简单提取化合物名称（不依赖 LLM 的 fallback）"""
        # 常见化合物词典
        common_compounds = [
            "aspirin", "caffeine", "ibuprofen", "paracetamol", "morphine",
            "penicillin", "testosterone", "cholesterol", "dopamine",
            "serotonin", "adrenaline", "nicotine", "ethanol", "glucose",
            "sucrose", "benzene", "toluene", "phenol", "aniline",
        ]
        found = []
        prompt_lower = prompt.lower()
        for c in common_compounds:
            if c in prompt_lower:
                found.append(c)
        return found
```

## 命令行入口

```python
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="化学图像生成流水线")
    p.add_argument("prompt", help="描述你想要的化学图像")
    p.add_argument("--style", default="ACS_1996")
    p.add_argument("--scene", default="photorealistic")
    p.add_argument("--method", default="pyramid")
    p.add_argument("--output", "-o", default="outputs/final/result.png")
    args = p.parse_args()
  
    pipeline = ChemicalImagePipeline(
        structure_style=args.style,
        scene_style=args.scene,
        composite_method=args.method,
    )
    result = pipeline.generate(args.prompt)
    result.save(args.output)
    print(result.summary())
```

## 测试

创建 tests/test_pipeline.py：

```python
def test_simple_pipeline():
    pipeline = ChemicalImagePipeline()
    result = pipeline.generate(
        "实验室白板上画着阿司匹林和咖啡因的结构式"
    )
    assert result.final_image is not None
    assert result.final_path.exists()
    assert len(result.compounds) >= 1  # 至少解析到 1 个
    print(f"✅ 端到端测试通过")
    print(result.summary())
```

完成后写入 outputs/phase_6_summary.txt。
```

---

---

# Phase 7：Gradio GUI 界面

## 上下文恢复

> Phase 1-6 全部完成。`ChemicalImagePipeline` 可正常工作。
> 本 Phase 创建 `src/app.py`。

## 对话指令

```
Phase 1-6 全部完成。构建 Phase 7：GUI 界面。

创建 src/app.py，使用 Gradio 搭建 Web 界面。

## 界面要求

```
┌─────────────────────────────────────────────────────┐
│  🧪 有机化学思维导图 AI 图像生成器                     │
│  输入你的想法，AI 自动生成包含精确化学结构的学术级图片  │
├─────────────────────────────────────────────────────┤
│                                                      │
│  描述你想要的内容：                                   │
│  ┌──────────────────────────────────────────────┐    │
│  │ 实验室白板上画着阿司匹林、咖啡因的结构式，     │    │
│  │ 旁边放着一杯咖啡，学术摄影风格                 │    │
│  └──────────────────────────────────────────────┘    │
│                                                      │
│  结构图风格：[ACS_1996 ▾]    场景风格：[photorealistic ▾] │
│  合成方法：[pyramid ▾]        图片尺寸：[1024×1024 ▾]    │
│                                                      │
│  [🚀 生成图片]    [🗑️ 清空]                           │
│                                                      │
├─────────────────────────────────────────────────────┤
│  生成结果：                                          │
│  ┌──────────────────────────────────────────────┐    │
│  │                                              │    │
│  │           [最终生成的图片预览]                 │    │
│  │                                              │    │
│  └──────────────────────────────────────────────┘    │
│                                                      │
│  📋 生成日志：                                       │
│  ┌──────────────────────────────────────────────┐    │
│  │ ✅ 解析到 3 个化合物                           │    │
│  │ ✅ 结构图生成完成 (3/3)                        │    │
│  │ ✅ 场景图生成完成                              │    │
│  │ ⏱️ 总耗时: 45.2 秒                             │    │
│  └──────────────────────────────────────────────┘    │
│                                                      │
│  [📥 下载图片]                                        │
└─────────────────────────────────────────────────────┘
```

## 完整代码框架

```python
# src/app.py
import gradio as gr
from pathlib import Path
from src.pipeline import ChemicalImagePipeline
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 初始化流水线
pipeline = ChemicalImagePipeline()

def generate_image(prompt, structure_style, scene_style, composite_method, size_str):
    """生成图片的主回调函数"""
    if not prompt or not prompt.strip():
        yield None, "⚠️ 请输入描述内容"
        return
  
    log = []
    log.append("🚀 开始生成...")
    yield None, "\n".join(log)
  
    try:
        # 更新配置
        pipeline.structure_style = structure_style
        pipeline.scene_style = scene_style
        pipeline.composite_method = composite_method
      
        log.append("📝 正在解析 prompt...")
        yield None, "\n".join(log)
      
        # 执行生成
        result = pipeline.generate(prompt)
      
        log.append(f"✅ 解析到 {len(result.compounds)} 个化合物")
        for c in result.compounds:
            log.append(f"   • {c.get('name', c.get('smiles', 'unknown'))}")
      
        log.append(f"✅ 结构图生成完成 ({len(result.structure_paths)} 个)")
        log.append("✅ 场景图生成完成")
        log.append("✅ 合成完成")
        log.append(f"⏱️ 总耗时: {result.elapsed_time:.1f} 秒")
        log.append(f"📁 输出: {result.final_path}")
      
        yield result.final_image, "\n".join(log)
      
    except Exception as e:
        log.append(f"❌ 错误: {str(e)}")
        logger.exception("生成失败")
        yield None, "\n".join(log)

# ── 构建 Gradio 界面 ──

with gr.Blocks(
    title="化学思维导图 AI 生成器",
    theme=gr.themes.Soft(),
    css="footer {visibility: hidden}"
) as app:
  
    gr.Markdown("""
    # 🧪 有机化学思维导图 AI 图像生成器
    输入你的想法，AI 自动生成包含**精确化学结构**的学术级图片。
    结构式由 RDKit 精确渲染，场景由 Stable Diffusion 生成，最终自然合成。
    """)
  
    with gr.Row():
        # ── 左侧：输入区 ──
        with gr.Column(scale=2):
            prompt_input = gr.Textbox(
                label="📝 描述你想要的内容",
                placeholder=(
                    "例如：实验室白板上画着阿司匹林（aspirin）和咖啡因（caffeine）"
                    "的结构式对比，旁边放着一杯咖啡，学术摄影风格"
                ),
                lines=4,
            )
          
            with gr.Row():
                style_dd = gr.Dropdown(
                    choices=["ACS_1996", "dark_mode", "color_on_white", "minimal"],
                    value="ACS_1996",
                    label="结构图风格",
                )
                scene_dd = gr.Dropdown(
                    choices=["photorealistic", "scientific_illustration", "warm_lab"],
                    value="photorealistic",
                    label="场景风格",
                )
          
            with gr.Row():
                method_dd = gr.Dropdown(
                    choices=["pyramid", "direct", "perspective"],
                    value="pyramid",
                    label="合成方法",
                )
                size_dd = gr.Dropdown(
                    choices=["1024×1024", "1280×720", "720×1280", "1792×1024"],
                    value="1024×1024",
                    label="图片尺寸",
                )
          
            with gr.Row():
                gen_btn = gr.Button("🚀 生成图片", variant="primary", size="lg")
                clear_btn = gr.Button("🗑️ 清空")
      
        # ── 右侧：输出区 ──
        with gr.Column(scale=3):
            image_output = gr.Image(
                label="生成结果",
                type="pil",
                height=450,
                show_download_button=True,
            )
            log_output = gr.Textbox(
                label="📋 生成日志",
                lines=10,
                interactive=False,
                max_lines=15,
            )
  
    # ── 事件绑定 ──
    gen_btn.click(
        fn=generate_image,
        inputs=[prompt_input, style_dd, scene_dd, method_dd, size_dd],
        outputs=[image_output, log_output],
    )
  
    clear_btn.click(
        fn=lambda: ("", None, ""),
        outputs=[prompt_input, image_output, log_output],
    )
  
    # ── 示例区 ──
    gr.Markdown("### 💡 试试这些示例（点击即可填入）")
    gr.Examples(
        examples=[
            ["实验室白板上画着阿司匹林（aspirin）的结构式，学术摄影风格，8k"],
            ["一张海报展示咖啡因（caffeine）和茶碱（theophylline）的化学结构对比"],
            ["化学家笔记本上画着布洛芬（ibuprofen）合成路线的思维导图"],
            ["现代实验室玻璃板上用马克笔写着紫杉醇（paclitaxel）的分子结构"],
            ["黑板上用粉笔画着苯环（benzene）的共振结构，有学术氛围"],
        ],
        inputs=[prompt_input],
    )

# ── 启动 ──
if __name__ == "__main__":
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )
```

## 启动方法

```bash
cd chem-image-gen
source venv/bin/activate
python src/app.py
# 打开浏览器访问 http://localhost:7860
```

## 完成后

写入 outputs/phase_7_summary.txt。
```

---

---

# 附录 A：常见问题与解决方案

## RDKit 安装失败

```bash
# 方案 1: conda
conda install -c conda-forge rdkit

# 方案 2: pip 非官方包
pip install rdkit-pypi

# 方案 3: Ubuntu 系统包
sudo apt install python3-rdkit
```

## GPU 显存不足（OOM）

在 `src/scene_gen/generator.py` 的 `load_model()` 中：

```python
pipe.enable_model_cpu_offload()
pipe.enable_vae_slicing()
pipe.enable_vae_tiling()  # 如果还 OOM
```

或者使用更小的模型：`stabilityai/stable-diffusion-2-1-base`

## Grounding DINO / SAM 模型下载慢

手动下载模型文件放到 `~/.cache/huggingface/hub/` 下，或使用 HuggingFace 镜像：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

## 无 GPU 环境

- 设置 `DEVICE=cpu`（在 `.env` 中）
- SD 生成会很慢（每张 2-5 分钟），但可运行
- 推荐使用远程 API：`SD_USE_LOCAL=false` + 配置 `SD_API_URL`

---

# 附录 B：依赖关系速查

| 模块                          | 依赖                                           |
| ----------------------------- | ---------------------------------------------- |
| `structure_gen/generator.py`  | rdkit, pubchempy, Pillow, numpy                |
| `compositor/basic.py`         | opencv, numpy                                  |
| `compositor/lighting.py`      | opencv, numpy, scipy                           |
| `compositor/pyramid.py`       | opencv, numpy                                  |
| `scene_gen/generator.py`      | torch, diffusers, transformers, httpx          |
| `detection/surface_detect.py` | torch, groundingdino, segment-anything, opencv |
| `agent/orchestrator.py`       | httpx（或 openai/anthropic SDK）               |
| `pipeline.py`                 | 以上全部                                       |
| `app.py`                      | gradio, pipeline.py                            |

---

# 附录 C：Phase 摘要模板

每个 Phase 结束时 Agent 应在 `outputs/` 下生成摘要文件：

```
Phase N 完成摘要
==================
时间: 2024-xx-xx xx:xx
Agent 对话编号: #N

创建的文件:
  - src/xxx/xxx.py
  - tests/test_xxx.py

关键接口:
  - ClassName.method(params) -> return_type

测试结果: N/N 通过

遗留问题:
  - (如无则写"无")

下次对话需读取:
  - build.md (Phase N+1 章节)
  - outputs/phase_N_summary.txt
```

---

# 开始构建

**从 Phase 0 开始。** 将对应章节的「对话指令」内容复制给 Claude，即可开始执行。每个 Phase 完成后会自动生成摘要文件，下一个 Agent 读取摘要文件即可恢复上下文。

整个项目预计需要 **7-9 次独立对话**完成。200K 记忆完全足够处理每个独立的 Phase。

`write` 工具不可用，我直接输出完整的 `build.md` 内容。你可以复制保存为文件。

---

```markdown
# build.md — 有机化学思维导图 AI 图像生成器

> **使用方式**：每个 Phase 在**独立对话**中执行。Agent 在对话开头读取本文件对应 Phase 章节，即可恢复上下文。每个 Phase 产生独立可用的代码工件。

---

## 项目总览

```
用户输入（GUI）
    │
    ▼
┌──────────────────────────────────────────────────┐
│  Phase 5: Agent 编排层 (LLM)                      │
│  · 解析用户意图 → 提取化合物列表 + 场景描述        │
│  · 决策调用哪些工具，以什么顺序                    │
└──────┬───────────────────────┬───────────────────┘
       │                       │
       ▼                       ▼
┌──────────────┐     ┌──────────────────┐
│ Phase 1      │     │ Phase 3          │
│ 结构图生成器  │     │ AI 场景生成器     │
│ (RDKit)      │     │ (SD/DALL-E API)  │
│ → 精确PNG    │     │ → 场景主图        │
└──────┬───────┘     └────────┬─────────┘
       │                      │
       └──────────┬───────────┘
                  ▼
       ┌──────────────────┐
       │ Phase 2          │
       │ 智能合成引擎      │
       │ (光照+透视+金字塔) │
       └────────┬─────────┘
                │
                ▼
       ┌──────────────────┐
       │ Phase 4          │
       │ 表面检测定位      │
       │ (可选,提升自动化) │
       └────────┬─────────┘
                │
                ▼
       ┌──────────────────┐
       │ Phase 6          │
       │ GUI 界面         │
       │ (Gradio)         │
       └──────────────────┘
```

**技术栈**：Python 3.10+ / RDKit / OpenCV / Pillow / NumPy / PubChemPy / Gradio

---

## 目录结构（最终形态）

```
chem-image-gen/
├── build.md                    # 本文件
├── requirements.txt
├── .env.example
├── config.yaml
├── src/
│   ├── __init__.py
│   ├── config.py               # 全局配置加载
│   ├── structure_gen.py        # Phase 1: RDKit 结构图生成
│   ├── compositor.py           # Phase 2: 智能合成引擎
│   ├── scene_gen.py            # Phase 3: AI 场景生成
│   ├── surface_detector.py     # Phase 4: 表面检测（可选）
│   ├── agent.py                # Phase 5: Agent 编排
│   ├── pipeline.py             # Phase 6a: 端到端流水线
│   └── gui.py                  # Phase 6b: Gradio GUI
├── output/
│   ├── structures/             # 中间产物：结构图 PNG
│   ├── scenes/                 # 中间产物：场景图
│   └── final/                  # 最终合成图
└── tests/
    ├── test_structure_gen.py
    ├── test_compositor.py
    └── test_pipeline.py
```

---

---

## Phase 1：环境搭建 + 结构图生成器

### 上下文恢复

> 这是项目的第一个 Phase。你需要从零开始创建项目骨架、安装依赖、并实现 RDKit 结构图生成器。
> 项目根目录为 `~/chem-image-gen/`。
> 阅读本文件"项目总览"和"目录结构"章节了解全局。

### 目标

1. 创建项目目录结构
2. 编写 `requirements.txt` 并安装所有依赖
3. 创建 `config.yaml` 和 `src/config.py`
4. 实现 `src/structure_gen.py`：通过化合物名称或 SMILES 生成精确的化学结构图
5. 编写测试脚本验证

### 任务清单

- [ ] 创建完整目录结构
- [ ] 编写 `requirements.txt`
- [ ] `pip install -r requirements.txt`（RDKit 可能需要 `conda install -c conda-forge rdkit`）
- [ ] 创建 `config.yaml`
- [ ] 创建 `src/config.py`
- [ ] 创建所有 `__init__.py`
- [ ] 实现 `src/structure_gen.py`
- [ ] 编写 `tests/test_structure_gen.py`
- [ ] 运行测试，确认通过

### 关键文件

#### `requirements.txt`

```
rdkit>=2023.09
opencv-python-headless>=4.8.0
Pillow>=10.0.0
numpy>=1.24.0
pubchempy>=1.0.4
pyyaml>=6.0
scipy>=1.11.0
httpx>=0.25.0
python-dotenv>=1.0.0
gradio>=4.0.0
```

#### `config.yaml`

```yaml
output:
  structures_dir: "output/structures"
  scenes_dir: "output/scenes"
  final_dir: "output/final"

structure_gen:
  default_width: 1200
  default_height: 800
  default_style: "ACS_1996"
  transparent_bg: true

image_gen:
  provider: "openai"
  default_size: [1024, 1024]
  default_quality: "hd"

openai:
  api_key: "${OPENAI_API_KEY}"

stability:
  api_key: "${STABILITY_API_KEY}"

local_sd:
  api_url: "http://127.0.0.1:7860"

compositor:
  feather_radius: 3
  shadow_opacity: 0.25
  color_match: true
  texture_blend: 0.08
  pyramid_levels: 3
```

#### `src/config.py`

```python
"""全局配置加载"""
import os
import yaml
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
PROJECT_ROOT = Path(__file__).parent.parent

def load_config(config_path=None):
    if config_path is None:
        config_path = PROJECT_ROOT / "config.yaml"
    with open(config_path) as f:
        raw = f.read()
    for var in os.environ:
        raw = raw.replace(f"${{{var}}}", os.environ[var])
    config = yaml.safe_load(raw)
    for d in ["structures_dir", "scenes_dir", "final_dir"]:
        (PROJECT_ROOT / config["output"][d]).mkdir(parents=True, exist_ok=True)
    return config
```

#### `src/structure_gen.py`（骨架）

> 你需要实现完整的 `StructureGenerator` 类。以下是接口规范和关键细节。

```python
"""
化合物结构图生成器 - 基于 RDKit + PubChemPy
"""
from __future__ import annotations
import io, hashlib, logging
from pathlib import Path
from typing import Optional, Literal
from dataclasses import dataclass

import pubchempy as pcp
from rdkit import Chem
from rdkit.Chem import Draw, AllChem, Descriptors
from rdkit.Chem.Draw import rdMolDraw2D
from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)
StyleName = Literal["ACS_1996", "dark_mode", "color_on_white", "minimal"]

STYLE_PRESETS = {
    "ACS_1996": {
        "bond_line_width": 2.5, "atom_label_font_size": 28,
        "background_color": (1, 1, 1, 0), "atom_color": (0, 0, 0),
        "bond_color": (0, 0, 0), "double_bond_offset": 0.25, "padding": 0.05,
    },
    "dark_mode": {
        "bond_line_width": 3.0, "atom_label_font_size": 32,
        "background_color": (0.1, 0.1, 0.1, 0), "atom_color": (0.9, 0.9, 0.9),
        "bond_color": (0.9, 0.9, 0.9), "double_bond_offset": 0.3, "padding": 0.08,
    },
    "color_on_white": {
        "bond_line_width": 2.0, "atom_label_font_size": 26,
        "background_color": (1, 1, 1, 1), "atom_color": (0.2, 0.2, 0.6),
        "bond_color": (0.3, 0.3, 0.3), "padding": 0.06,
    },
    "minimal": {
        "bond_line_width": 1.5, "atom_label_font_size": 22,
        "background_color": (1, 1, 1, 0), "atom_color": (0.15, 0.15, 0.15),
        "bond_color": (0.15, 0.15, 0.15), "padding": 0.03,
    },
}

@dataclass
class CompoundInfo:
    name: str
    smiles: str
    formula: str
    iupac_name: str
    molecular_weight: float

class StructureGenerator:
    """化学结构图生成器。"""

    ATOM_HIGHLIGHT_COLORS = {
        "O": (0.8,0.2,0.2), "N": (0.2,0.2,0.8), "S": (0.8,0.8,0.2),
        "P": (0.8,0.5,0.2), "F": (0.2,0.8,0.2), "Cl": (0.2,0.8,0.2),
        "Br": (0.6,0.3,0.1), "I": (0.5,0.2,0.5),
    }

    def __init__(self, default_width=1200, default_height=800,
                 default_style="ACS_1996", output_dir="output/structures",
                 transparent_bg=True):
        self.default_width = default_width
        self.default_height = default_height
        self.default_style = default_style
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.transparent_bg = transparent_bg

    def resolve_compound(self, query: str) -> CompoundInfo:
        """通过名称/CAS号/SMILES 解析化合物。
        策略：1) 尝试直接 SMILES  2) PubChem 查询  3) 失败抛 ValueError"""
        # ... 由你实现 ...

    def generate_from_smiles(self, smiles, output_path=None, width=None,
                             height=None, style=None, label=None) -> Path:
        """通过 SMILES 生成结构图 PNG。
        关键步骤：
        1. Chem.MolFromSmiles → 检查 None
        2. AllChem.Compute2DCoords
        3. rdMolDraw2D.MolDraw2DCairo(w, h)
        4. 配置 drawOptions()（bondLineWidth, fontSize, padding 等）
        5. drawer.DrawMolecule(mol, legend=label)
        6. drawer.FinishDrawing() → GetDrawingText() → PIL Image
        7. transparent_bg 处理：白色→透明
        8. 保存 PNG"""
        # ... 由你实现 ...

    def generate_from_name(self, compound_name, output_path=None, **kwargs) -> Path:
        """通过名称生成结构图。先 resolve_compound 再 generate_from_smiles。"""
        # ... 由你实现 ...

    def generate_batch(self, compounds: list, style=None) -> list[Path]:
        """批量生成，compounds 每项可以是名称或 SMILES。"""
        # ... 由你实现 ...
```

**实现要点**：
1. 必须使用 `rdMolDraw2D.MolDraw2DCairo`（不是 Agg 后端），Cairo 支持更好的渲染
2. 透明背景：Cairo 默认不透明，需在 PIL 中将白色像素 alpha 设为 0
3. `color_on_white` 风格需高亮杂原子（O、N、S 等非 C/H 原子）
4. 自动生成文件名时使用 SMILES 的 MD5 前 12 位

#### `tests/test_structure_gen.py`

```python
"""Phase 1 测试"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.structure_gen import StructureGenerator, STYLE_PRESETS
from PIL import Image
import numpy as np

def test_smiles(): ...
def test_name(): ...
def test_all_styles(): ...
def test_transparent(): ...
def test_invalid(): ...
def test_batch(): ...

if __name__ == "__main__":
    # 运行所有测试
    print("🎉 全部通过！Phase 1 完成。")
```

### 验收标准

```bash
cd ~/chem-image-gen
python tests/test_structure_gen.py
# 输出: 🎉 全部通过！Phase 1 完成。
ls output/structures/
```

---

---

## Phase 2：智能合成引擎

### 上下文恢复

> Phase 1 已完成。`src/structure_gen.py` 可生成透明背景 PNG 结构图。
> 本 Phase 目标：实现 `src/compositor.py`，将结构图自然合成到场景主图中。
> 核心挑战：消除"PS 感"——光照一致、边缘无缝、透视正确。
> 项目根目录：`~/chem-image-gen/`

### 目标

实现 `SmartCompositor` 类，三种合成模式：

| 模式          | 适用场景      | 核心算法                           |
| ------------- | ------------- | ---------------------------------- |
| `direct`      | 正面白板/屏幕 | Alpha合成 + 色彩匹配 + 羽化 + 投影 |
| `perspective` | 倾斜表面      | 透视变换 + 色彩匹配                |
| `pyramid`     | 最自然效果    | 拉普拉斯金字塔多尺度融合           |

### 核心代码骨架

```python
"""
智能合成引擎 - 将结构图无缝融入场景主图
"""
from __future__ import annotations
import logging
from pathlib import Path
from typing import Optional, Literal
from dataclasses import dataclass
import cv2
import numpy as np

logger = logging.getLogger(__name__)
FusionMethod = Literal["direct", "perspective", "pyramid"]

@dataclass
class CompositorConfig:
    feather_radius: int = 3
    shadow_offset: tuple = (3, 3)
    shadow_blur: int = 5
    shadow_opacity: float = 0.25
    color_match: bool = True
    texture_blend: float = 0.08
    pyramid_levels: int = 3

class SmartCompositor:
    def __init__(self, config=None):
        self.config = config or CompositorConfig()

    def composite(self, background, structure, target_region,
                  dst_points=None, method="pyramid", output=None) -> np.ndarray:
        """主入口。加载图像 → 按方法分发 → 保存 → 返回 BGR 数组"""
        # ... 由你实现 ...

    def _direct(self, bg, struct, region) -> np.ndarray:
        """方案 A：直接 Alpha 合成。
        步骤：缩放结构图 → 提取 RGBA → 色彩匹配 → 纹理叠加
        → 羽化 alpha → 添加投影 → Alpha 混合"""
        # ... 由你实现 ...

    def _perspective(self, bg, struct, region, dst_points) -> np.ndarray:
        """方案 B：透视变换合成。
        cv2.getPerspectiveTransform + cv2.warpPerspective + Alpha 混合"""
        # ... 由你实现 ...

    def _pyramid(self, bg, struct, region, dst_points) -> np.ndarray:
        """方案 C：拉普拉斯金字塔多尺度融合。
        构建 3 层高斯金字塔 → 拉普拉斯金字塔 → 每层 Alpha 融合 → 重建"""
        # ... 由你实现 ...

    # 辅助方法
    def _match_colors(self, src, target):
        """色彩匹配：src 的 mean/std 对齐到 target。核心抗PS感技术"""
        # src_f = src.astype(np.float32)
        # s_mean, s_std = src_f.mean/std
        # t_mean, t_std = target.mean/std
        # matched = (src_f - s_mean) * (t_std/s_std) + t_mean
        # ... 由你实现 ...

    def _add_texture(self, struct, roi, strength):
        """ROI 高频纹理微量叠加到结构图 → 融入表面"""
        # texture = roi - cv2.GaussianBlur(roi, (21,21), 0)
        # ... 由你实现 ...

    def _drop_shadow(self, bg, alpha, region, offset, blur, opacity):
        """结构图下方添加微妙投影"""
        # ... 由你实现 ...

    def _load(self, src, mode):
        """加载图像，自动处理 BGR/BGRA"""
        # ... 由你实现 ...
```

**关键实现细节**：
1. 金字塔融合的核心：`cv2.pyrDown` 构建高斯金字塔，然后 `lap = g[i] - pyrUp(g[i+1])` 获取拉普拉斯层
2. 每层用对应尺度的 mask 做 Alpha 混合，最后从最粗层逐级 `pyrUp + lap` 重建
3. 色彩匹配只对有效像素区域（alpha > 0.1）
4. 投影 opacity 通常 0.15-0.3，太重会不自然

### 验收标准

```bash
cd ~/chem-image-gen
python tests/test_compositor.py
# 输出: 🎉 全部通过！Phase 2 完成。
ls output/final/
```

---

---

## Phase 3：AI 场景生成器

### 上下文恢复

> Phase 1-2 已完成。本 Phase 实现 AI 场景图生成。
> 项目根目录：`~/chem-image-gen/`
> 配置在 `config.yaml` 中，API Key 通过 `.env` 管理。

### 目标

实现 `SceneGenerator` 类，支持 OpenAI DALL-E 3 / Stability API / 本地 SD WebUI 三种后端。

### 核心设计

**引导性 Prompt 构建**是关键——要让 AI 生成易于合成的场景：

```python
@dataclass
class ScenePrompt:
    scene_desc: str
    surface_type: str      # whiteboard | screen | paper | glass_board
    surface_position: str  # center | left | right | full
    style: str             # photorealistic | scientific | casual

    def build(self) -> str:
        """构建完整 prompt，引导 AI 生成包含清晰矩形放置面的场景"""
        # 关键：加入 "straight-on angle, minimal perspective distortion,
        #         the writing surface should be clearly rectangular"
        # ... 由你实现 ...
```

### 核心代码骨架

```python
"""
AI 场景生成接口
"""
from __future__ import annotations
import logging, time
from pathlib import Path
from typing import Optional, Literal
from dataclasses import dataclass
from io import BytesIO
import yaml, httpx
from PIL import Image

logger = logging.getLogger(__name__)

@dataclass
class ScenePrompt:
    scene_desc: str
    surface_type: str = "whiteboard"
    surface_position: str = "center"
    style: str = "photorealistic"
    def build(self) -> str: ...

class SceneGenerator:
    def __init__(self, config_path="config.yaml"): ...

    def generate(self, prompt, surface_type="whiteboard",
                 surface_position="center", style="photorealistic",
                 size=None, output=None) -> Path:
        """主入口：构建 prompt → 调用后端 → 保存 PNG → 返回路径"""
        # ... 由你实现 ...

    def _gen_openai(self, prompt, size) -> Image.Image:
        """DALL-E 3: openai.images.generate(model="dall-e-3", ...)"""
        # ... 由你实现 ...

    def _gen_stability(self, prompt, size) -> Image.Image:
        """Stability API v2: /v2beta/stable-image/generate/core"""
        # ... 由你实现 ...

    def _gen_local_sd(self, prompt, size) -> Image.Image:
        """本地 SD WebUI: POST /sdapi/v1/txt2img"""
        # ... 由你实现 ...
```

**注意**：
- `.env` 中设置 `OPENAI_API_KEY` 或 `STABILITY_API_KEY` 或本地 SD 地址
- DALL-E 3 只支持三种尺寸：1024x1024、1792x1024、1024x1792

### 验收标准

```bash
export OPENAI_API_KEY=sk-xxxx
python -c "
from src.scene_gen import SceneGenerator
gen = SceneGenerator()
path = gen.generate('a chemistry lab whiteboard', output='output/scenes/test.png')
print(path)
"
```

---

---

## Phase 4：表面检测定位（可选）

### 上下文恢复

> Phase 1-3 已完成。本 Phase 为可选——实现自动检测场景图中的可放置表面区域。
> 如果时间有限，可跳过，在 GUI 中让用户手动框选或使用 Phase 5 Agent 估算的默认坐标。

### 目标

实现 `SurfaceDetector`：基于 Canny 边缘检测 + 轮廓查找定位矩形区域。

### 核心骨架

```python
"""
表面检测定位
"""
from dataclasses import dataclass
import cv2, numpy as np

@dataclass
class SurfaceRegion:
    bbox: tuple[int,int,int,int]    # (x, y, w, h)
    corners: np.ndarray              # (4,2) 四角点
    confidence: float
    surface_type: str

class SurfaceDetector:
    def __init__(self, method="edge"): ...

    def detect(self, image_path) -> list[SurfaceRegion]:
        """Canny 边缘检测 → 查找轮廓 → 筛选四边形 → 排序返回"""
        # 1. 灰度 → 高斯模糊 → Canny(50,150) → 膨胀
        # 2. findContours → 按面积过滤（>5% 图像面积）
        # 3. approxPolyDP → 找四条边的
        # 4. _order_corners（左上→右上→右下→左下）
        # ... 由你实现 ...

    def _order_corners(self, pts): ...
```

---

---

## Phase 5：Agent 编排层

### 上下文恢复

> Phase 1-3 已完成。本 Phase 实现 LLM Agent，自动解析用户意图并编排工具调用。
> 项目根目录：`~/chem-image-gen/`

### 目标

实现 `ChemistryImageAgent`：
- 将工具清单（System Prompt）发给 LLM
- LLM 返回 JSON 执行计划
- Agent 按计划调用 `StructureGenerator` → `SceneGenerator` → `SmartCompositor`
- 返回最终图片

### 核心设计

**Agent 不做推理**，而是让 Claude/GPT-4 做推理。Agent 只负责：1) 发 Prompt + 工具清单，2) 解析 JSON，3) 执行工具调用。

### 核心代码骨架

```python
"""
Agent 编排层
"""
import json, logging
from src.structure_gen import StructureGenerator
from src.compositor import SmartCompositor
from src.scene_gen import SceneGenerator

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """你是一个化学图像生成专家 Agent。

可用工具:
1. resolve_compound(name) → {smiles, formula, iupac_name}
2. generate_structure(smiles, style, label) → file_path
3. generate_scene(description, surface_type) → file_path
4. composite(background, structure, x, y, w, h, method) → file_path

请输出 JSON 执行计划:
{
    "scene": {"description": "...", "surface_type": "whiteboard", "style": "photorealistic"},
    "compounds": [
        {"name": "...", "location": {"x":100,"y":150,"width":500,"height":350}, "style":"ACS_1996"}
    ],
    "composite_method": "pyramid"
}

位置估算（1024x1024 场景）:
- 1 个化合物: 居中 (212, 287, 600, 450)
- 2 个化合物: 左右并排: 左 (50,287,430,450), 右 (544,287,430,450)
- 3 个化合物: 三角布局: 顶 (312,50,400,300), 左下 (50,400,430,400), 右下 (544,400,430,400)
- 4+ 个化合物: 2x2 网格
"""

class ChemistryImageAgent:
    def __init__(self, config_path="config.yaml", llm=None):
        self.structure_gen = StructureGenerator(...)
        self.scene_gen = SceneGenerator(...)
        self.compositor = SmartCompositor(...)
        self.llm = llm  # 由调用方注入

    def _tool_resolve_compound(self, name): ...
    def _tool_generate_structure(self, smiles, style, label): ...
    def _tool_generate_scene(self, description, surface_type): ...
    def _tool_composite(self, bg, struct, x, y, w, h, method): ...

    def _call_llm(self, user_prompt) -> dict:
        """调用 LLM 获取执行计划。需要自己实现。"""
        # 发送 SYSTEM_PROMPT + user_prompt
        # 要求只输出 JSON
        # 解析并返回 dict
        raise NotImplementedError("请实现 _call_llm")

    def run(self, user_input) -> dict:
        """端到端：LLM 解析 → 生成场景 → 生成结构图 → 逐个合成"""
        plan = self._call_llm(user_input)
        scene_path = self._tool_generate_scene(...)
        for comp in plan["compounds"]:
            info = self._tool_resolve_compound(comp["name"])
            struct_path = self._tool_generate_structure(...)
            scene_path = self._tool_composite(scene_path, struct_path, ...)
        return {"final_image": scene_path, "plan": plan}


def create_claude_agent(api_key=None, model="claude-sonnet-4-20250514"):
    """工厂函数：创建使用 Claude 作为后端的 Agent"""
    import anthropic, os
    api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    client = anthropic.Anthropic(api_key=api_key)
    agent = ChemistryImageAgent()

    def call_claude(user_prompt):
        resp = client.messages.create(
            model=model, max_tokens=2048, system=SYSTEM_PROMPT,
            messages=[{"role":"user","content":user_prompt+"\n\n请只输出JSON。"}]
        )
        text = resp.content[0].text
        # 提取 JSON（处理 ```json 包裹）
        if "```json" in text: text = text.split("```json")[1].split("```")[0]
        elif "```" in text: text = text.split("```")[1].split("```")[0]
        return json.loads(text.strip())

    agent._call_llm = call_claude
    return agent
```

---

---

## Phase 6：GUI 界面 + 端到端联调

### 上下文恢复

> 所有核心模块已完成。本 Phase 用 Gradio 构建 GUI 并串联所有组件。
> 项目根目录：`~/chem-image-gen/`

### 目标

1. `src/pipeline.py`：一行代码调用的端到端流水线
2. `src/gui.py`：Gradio Web GUI
3. 完整集成测试

### 核心代码骨架

#### `src/pipeline.py`

```python
"""端到端流水线"""
from src.agent import create_claude_agent

class Pipeline:
    def __init__(self, api_key=None):
        self.agent = create_claude_agent(api_key=api_key)

    def run(self, user_input) -> dict:
        return self.agent.run(user_input)
```

#### `src/gui.py`（关键部分）

```python
"""Gradio GUI"""
import os, gradio as gr
from PIL import Image
from src.pipeline import Pipeline

_pipeline = None

def get_pipeline():
    global _pipeline
    if _pipeline is None:
        _pipeline = Pipeline(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    return _pipeline

def generate_image(user_prompt, surface_type, image_style, structure_style, progress=gr.Progress()):
    if not user_prompt.strip():
        return [None, None, None, "❌ 请输入描述"]
    pipe = get_pipeline()
    progress(0.1, desc="🔍 解析意图...")
    enhanced = f"{user_prompt}。场景为{surface_type}，{image_style}风格，结构式{structure_style}风格。"
    try:
        result = pipe.run(enhanced)
    except Exception as e:
        return [None, None, None, f"❌ 失败: {e}"]
    progress(0.9, desc="✅ 完成")
    final = Image.open(result["final_image"]) if result.get("final_image") else None
    compounds_info = result.get("compounds", [])
    info = "\n".join([f"- **{c['name']}**: `{c.get('smiles','')}`" for c in compounds_info])
    return [final, None, None, info]

def create_ui():
    with gr.Blocks(title="有机化学图像生成器", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🧪 有机化学思维导图生成器\n输入想法，AI 生成学术级化学图像")
        with gr.Row():
            with gr.Column(scale=1):
                user_prompt = gr.Textbox(label="描述", placeholder="例如：画一张实验室白板，上面有阿司匹林和咖啡因的结构式...", lines=4)
                surface_type = gr.Dropdown(["whiteboard","screen","paper","glass_board"], value="whiteboard", label="场景")
                image_style = gr.Dropdown(["photorealistic","scientific","casual"], value="photorealistic", label="图片风格")
                structure_style = gr.Dropdown(["ACS_1996","dark_mode","color_on_white","minimal"], value="ACS_1996", label="结构式风格")
                btn = gr.Button("🚀 生成", variant="primary")
                info = gr.Markdown("")
            with gr.Column(scale=2):
                final_image = gr.Image(label="最终结果", type="pil")
        examples = gr.Examples([
            ["画一张实验室白板，上面有阿司匹林的结构式", "whiteboard","photorealistic","ACS_1996"],
            ["平板屏幕显示咖啡因和布洛芬结构", "screen","photorealistic","color_on_white"],
        ], inputs=[user_prompt, surface_type, image_style, structure_style])
        btn.click(generate_image, [user_prompt,surface_type,image_style,structure_style], [final_image,None,None,info])
    return demo

if __name__ == "__main__":
    create_ui().launch(server_name="0.0.0.0", server_port=7860)
```

### 验收标准

```bash
export ANTHROPIC_API_KEY=your_key
python src/gui.py
# 浏览器打开 http://localhost:7860
# 输入测试 prompt → 点击生成 → 检查结果
```

---

---

## 附录 A：快速启动

```bash
cd ~/chem-image-gen
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
# 如果 RDKit 失败: conda install -c conda-forge rdkit

cp .env.example .env   # 填入 API Key
python tests/test_structure_gen.py
python tests/test_compositor.py
python src/gui.py      # 打开 http://localhost:7860
```

## 附录 B：对话记忆管理

每个 Phase 独立对话。开头模板：

```
请先读取 ~/chem-image-gen/build.md 中 Phase N 章节。
Phase N-1 已完成，产出物接口如下：
  [列出前序 Phase 的关键类和接口]
现在请实现 Phase N，按照 build.md 中的骨架代码完成。
```

## 附录 C：常见问题

| 问题           | 方案                                     |
| -------------- | ---------------------------------------- |
| RDKit 安装失败 | `conda install -c conda-forge rdkit`     |
| 合成有"PS感"   | 启用 `color_match=True` + `pyramid` 方法 |
| 结构位置不准   | 跳过 Phase 4，GUI 中手动框选或用默认坐标 |
| API 调用失败   | 检查 `.env` 中的 Key，确认网络可达       |
```

---

这个 `build.md` 可以直接保存到项目根目录使用。每个 Phase 都是自包含的，Agent 在每次对话开头读对应章节即可恢复上下文。Phase 1-3 是核心必做，Phase 4 可选，Phase 5-6 是集成和 GUI。