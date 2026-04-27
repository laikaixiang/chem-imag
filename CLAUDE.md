# chem-mindmap

AI驱动的有机化学思维导图生成器，输出包含精确化学结构的学术论文级图像。

## Environment

```bash
conda activate chem-image  # Python 3.11
cd D:\PycharmProjects\chem-image
```

## Commands

```bash
# 安装依赖
pip install -e .

# 安装 RDKit（Phase 1 必需，用 conda）
conda install -c conda-forge rdkit

# 安装 PyTorch（可选，AI 功能）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 验证环境
python tests/verify_env_simple.py

# 运行测试
python tests/test_structure_gen.py
python tests/test_compositor.py
python tests/test_pipeline.py
```

## Architecture

```
src/
├── config.py          # Settings 单例，加载 .env，自动检测 CUDA/MPS/CPU
├── structure_gen/     # Phase 1: RDKit 精确结构图生成
├── compositor/        # Phase 2: 图像合成（basic/lighting/pyramid）
├── mindmap/           # Phase 3: 思维导图布局引擎
├── scene_gen/         # Phase 4: AI 背景场景生成
├── agent/             # Phase 5: LLM 编排层 + Tool 系统
└── gui/               # Phase 6: Gradio 界面
```

每个 Phase 在独立对话中实现，通过 `build.md` 对应章节恢复上下文。

## Key Files

- `build.md` — 完整构建指南，每个 Phase 的 Agent 指令在此
- `src/config.py` — `settings` 单例，所有模块通过它获取路径和设备信息
- `.env.example` — 复制为 `.env` 并填入 API Key
- `logs/phase_N_log.md` — 每个 Phase 的完成记录

## Configuration

`.env` 关键变量：

```
AI_IMAGE_PROVIDER=sd_webui   # sd_webui | openai | replicate
ANTHROPIC_API_KEY=           # Claude API Key
OUTPUT_DIR=./outputs
DEVICE=cuda                  # cuda | mps | cpu（config.py 会自动检测覆盖）
```

## Gotchas

- **RDKit 必须用 conda 安装**，pip 版本在 Windows 上不稳定：`conda install -c conda-forge rdkit`
- **NumPy 版本冲突**：OpenCV 4.12+ 需要 NumPy 2.x，但 Anaconda 基础环境部分包依赖 1.x。始终在 `chem-image` conda 环境中工作，不要在 base 环境运行
- **gradio 导入时**若出现 pyarrow/pandas NumPy 警告可忽略，不影响功能
- `settings.ensure_dirs()` 在 `config.py` 模块加载时自动调用，无需手动创建 `outputs/` 子目录
- 输出目录在 `.gitignore` 中，不提交到 git

## Build Phases

| Phase | 模块 | 状态 |
|-------|------|------|
| 0 | 环境搭建 | ✅ 完成 |
| 1 | `structure_gen` — RDKit 结构图 | 待实现 |
| 2 | `compositor` — 图像合成 | 待实现 |
| 3 | `mindmap` — 布局引擎 | 待实现 |
| 4 | `scene_gen` — AI 场景 | 待实现 |
| 5 | `agent` — 编排层 | 待实现 |
| 6 | `gui` — Gradio 界面 | 待实现 |
| 7 | 端到端联调 | 待实现 |
