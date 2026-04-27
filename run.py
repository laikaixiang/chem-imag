#!/usr/bin/env python3
"""chem-mindmap 启动入口 — 支持 tkinter 和 Gradio 两种 GUI 模式."""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main():
    parser = argparse.ArgumentParser(description="启动有机化学思维导图生成器")
    parser.add_argument("--mode", choices=["tkinter", "gradio"], default="tkinter",
                        help="GUI 模式: tkinter (原生) 或 gradio (Web)")
    parser.add_argument("--port", type=int, default=7860, help="Gradio 端口号")
    parser.add_argument("--share", action="store_true", help="Gradio 创建公开链接")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    print("🧪 有机化学思维导图生成器")
    print(f"   GUI 模式: {args.mode}")

    if args.mode == "gradio":
        from src.gui.app_gradio import launch_gradio
        print(f"   端口: {args.port}")
        print(f"   打开浏览器访问: http://127.0.0.1:{args.port}")
        launch_gradio(share=args.share, port=args.port)
    else:
        from src.gui.app_tkinter import launch_tkinter
        launch_tkinter()


if __name__ == "__main__":
    main()
