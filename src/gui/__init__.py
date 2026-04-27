"""GUI module for chem-mindmap — tkinter and Gradio interfaces.

Each backend is lazily imported to avoid dependency conflicts
(gradio → pandas → pyarrow can fail with NumPy 2.x).
"""

__all__ = ['launch_tkinter', 'launch_gradio']


def launch_tkinter():
    from .app_tkinter import launch_tkinter as _launch
    _launch()


def launch_gradio(share: bool = False, port: int = 7860):
    from .app_gradio import launch_gradio as _launch
    _launch(share=share, port=port)
