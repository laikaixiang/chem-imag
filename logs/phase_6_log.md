# Phase 6: GUI 界面 — Completion Log

**Date**: 2026-04-28
**Author**: lkx
**Status**: ✅ Complete

## Files Created

| File | Description |
|------|-------------|
| `src/gui/__init__.py` | 延迟导入，避免 gradio/pandas/numpy 依赖冲突 |
| `src/gui/app_tkinter.py` | Python 原生 GUI（tkinter），3 面板布局，后台线程生成 |
| `src/gui/app_gradio.py` | Web 界面（Gradio Blocks），3 Tab，事件驱动 |
| `run.py` | 启动入口，`--mode tkinter/gradio --port --share` |

## 启动方式

```bash
# Python 原生 GUI（默认，无需浏览器）
python run.py

# Web 界面
python run.py --mode gradio --port 7860

# Web 界面 + 公网分享
python run.py --mode gradio --share
```

## tkinter 版本要点

- `ChemMindmapApp` 类，3 个 LabelFrame 面板
- 后台 `threading.Thread` 执行生成，`queue.Queue` + `root.after()` 回主线程更新 UI
- PIL ImageTk 嵌入显示图片
- 结构图预览：下拉预设 + 自定义输入 + SMILES 自动解析
- 设置面板：API URL、LLM 选择、默认尺寸、输出目录

## Gradio 版本要点

- `gr.Blocks(theme=gr.themes.Soft())` + 3 个 TabItem
- 事件驱动：`.click(fn=..., inputs=[...], outputs=[...])`
- 调用 `AgentOrchestrator.run()` 执行默认管线
- 结构图预览 Tab 独立，调用 `StructureGenerator` 直接生成

## 已知问题

- Gradio 在 base 环境有 NumPy 1.x/2.x 冲突 → 需在 `chem-image` conda 环境运行
- CLAUDE.md Gotchas 中已有记录此问题
