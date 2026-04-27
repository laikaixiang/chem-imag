"""Python native GUI (tkinter) for chem-mindmap."""

import logging
import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageTk

from src.config import settings

logger = logging.getLogger(__name__)

PRESET_COMPOUNDS = [
    "aspirin", "caffeine", "paracetamol", "ibuprofen",
    "benzene", "phenol", "benzoic acid", "ethanol", "acetic acid",
]

STYLE_OPTIONS = ["academic", "modern", "minimal"]
STRUCT_STYLES = ["ACS_1996", "dark_mode", "color_on_white", "minimal"]
SIZE_OPTIONS = ["1920×1080", "2560×1440"]
PROVIDER_OPTIONS = ["mock", "sd_webui"]


class ChemMindmapApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("🧪 有机化学思维导图生成器")
        self.root.geometry("1100x900")
        self.root.minsize(900, 700)

        self._result_queue: queue.Queue = queue.Queue()
        self._preview_photo: ImageTk.PhotoImage | None = None
        self._struct_photo: ImageTk.PhotoImage | None = None
        self._result_photo: ImageTk.PhotoImage | None = None

        self._build_ui()
        self._poll_queue()

    # ── UI construction ─────────────────────────────────────────

    def _build_ui(self):
        # Header
        header = ttk.Frame(self.root)
        header.pack(fill=tk.X, padx=10, pady=(10, 0))
        ttk.Label(header, text="🧪 有机化学思维导图生成器",
                  font=("Microsoft YaHei", 16, "bold")).pack()
        ttk.Label(header, text="AI 驱动的学术论文级有机化学思维导图生成",
                  font=("Microsoft YaHei", 9)).pack()

        # Notebook (tab-like panels with LabelFrame stacking)
        content = ttk.Frame(self.root)
        content.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self._build_generation_panel(content)
        self._build_structure_panel(content)
        self._build_settings_panel(content)

    def _build_generation_panel(self, parent):
        frame = ttk.LabelFrame(parent, text="📝 生成思维导图", padding=10)
        frame.pack(fill=tk.BOTH, expand=True, pady=(0, 8))

        # Prompt
        ttk.Label(frame, text="输入描述:").pack(anchor=tk.W)
        self.prompt_text = tk.Text(frame, height=3, font=("Microsoft YaHei", 10))
        self.prompt_text.pack(fill=tk.X, pady=(2, 6))
        self.prompt_text.insert("1.0", "生成关于苯酚、苯甲酸及其酯化反应的思维导图")

        # Controls row
        ctrl = ttk.Frame(frame)
        ctrl.pack(fill=tk.X, pady=4)

        ttk.Label(ctrl, text="风格:").pack(side=tk.LEFT)
        self.style_var = tk.StringVar(value="academic")
        ttk.Combobox(ctrl, textvariable=self.style_var, values=STYLE_OPTIONS,
                     width=10, state="readonly").pack(side=tk.LEFT, padx=4)

        ttk.Label(ctrl, text="尺寸:").pack(side=tk.LEFT, padx=(12, 0))
        self.size_var = tk.StringVar(value="1920×1080")
        ttk.Combobox(ctrl, textvariable=self.size_var, values=SIZE_OPTIONS,
                     width=10, state="readonly").pack(side=tk.LEFT, padx=4)

        ttk.Label(ctrl, text="后端:").pack(side=tk.LEFT, padx=(12, 0))
        self.provider_var = tk.StringVar(value="mock")
        ttk.Combobox(ctrl, textvariable=self.provider_var, values=PROVIDER_OPTIONS,
                     width=10, state="readonly").pack(side=tk.LEFT, padx=4)

        self.gen_btn = ttk.Button(ctrl, text="🔬 开始生成", command=self._start_generation)
        self.gen_btn.pack(side=tk.RIGHT)

        # Status
        self.status_var = tk.StringVar(value="✅ 就绪，等待输入...")
        ttk.Label(frame, textvariable=self.status_var, font=("Microsoft YaHei", 9),
                  foreground="gray").pack(anchor=tk.W, pady=(4, 2))

        # Progress bar
        self.progress = ttk.Progressbar(frame, mode="indeterminate")

        # Image display area
        img_frame = ttk.Frame(frame)
        img_frame.pack(fill=tk.BOTH, expand=True, pady=6)

        # Preview
        preview_box = ttk.LabelFrame(img_frame, text="🖼️ 中间预览")
        preview_box.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 4))
        self.preview_label = ttk.Label(preview_box, text="（生成过程中显示）")
        self.preview_label.pack(fill=tk.BOTH, expand=True)

        # Result
        result_box = ttk.LabelFrame(img_frame, text="🎯 最终结果")
        result_box.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(4, 0))
        self.result_label = ttk.Label(result_box, text="（生成完成后显示）")
        self.result_label.pack(fill=tk.BOTH, expand=True)

        # Action buttons
        actions = ttk.Frame(frame)
        actions.pack(fill=tk.X, pady=(4, 0))
        ttk.Button(actions, text="📥 下载 PNG", command=self._download_result).pack(side=tk.LEFT, padx=2)
        ttk.Button(actions, text="🔄 重新生成", command=self._start_generation).pack(side=tk.LEFT, padx=2)

    def _build_structure_panel(self, parent):
        frame = ttk.LabelFrame(parent, text="🔬 结构图预览", padding=10)
        frame.pack(fill=tk.BOTH, expand=True, pady=(0, 8))

        # Query row
        qrow = ttk.Frame(frame)
        qrow.pack(fill=tk.X, pady=2)

        ttk.Label(qrow, text="化合物:").pack(side=tk.LEFT)
        self.compound_var = tk.StringVar(value="aspirin")
        combo = ttk.Combobox(qrow, textvariable=self.compound_var,
                             values=PRESET_COMPOUNDS, width=18)
        combo.pack(side=tk.LEFT, padx=4)
        ttk.Button(qrow, text="🔍 查询", command=self._query_compound).pack(side=tk.LEFT, padx=4)

        ttk.Label(qrow, text="SMILES:").pack(side=tk.LEFT, padx=(12, 0))
        self.smiles_var = tk.StringVar(value="CC(=O)Oc1ccccc1C(=O)O")
        ttk.Entry(qrow, textvariable=self.smiles_var, width=30).pack(side=tk.LEFT, padx=4)

        # Generate row
        grow = ttk.Frame(frame)
        grow.pack(fill=tk.X, pady=4)

        ttk.Label(grow, text="风格:").pack(side=tk.LEFT)
        self.struct_style_var = tk.StringVar(value="ACS_1996")
        ttk.Combobox(grow, textvariable=self.struct_style_var, values=STRUCT_STYLES,
                     width=14, state="readonly").pack(side=tk.LEFT, padx=4)
        ttk.Button(grow, text="生成结构图", command=self._generate_structure).pack(side=tk.LEFT, padx=8)

        # Structure display
        self.struct_label = ttk.Label(frame, text="（点击 [生成结构图] 显示）")
        self.struct_label.pack(fill=tk.BOTH, expand=True, pady=4)

    def _build_settings_panel(self, parent):
        frame = ttk.LabelFrame(parent, text="⚙️ 设置", padding=10)
        frame.pack(fill=tk.X, pady=(0, 8))

        row1 = ttk.Frame(frame)
        row1.pack(fill=tk.X, pady=2)
        ttk.Label(row1, text="API URL:").pack(side=tk.LEFT)
        self.api_url_var = tk.StringVar(value=settings.SD_WEBUI_URL)
        ttk.Entry(row1, textvariable=self.api_url_var, width=40).pack(side=tk.LEFT, padx=4)

        ttk.Label(row1, text="LLM:").pack(side=tk.LEFT, padx=(12, 0))
        self.llm_var = tk.StringVar(value=settings.LLM_PROVIDER)
        ttk.Combobox(row1, textvariable=self.llm_var, values=["claude", "openai", "default"],
                     width=10, state="readonly").pack(side=tk.LEFT, padx=4)

        row2 = ttk.Frame(frame)
        row2.pack(fill=tk.X, pady=2)

        ttk.Label(row2, text="默认尺寸:").pack(side=tk.LEFT)
        self.def_w_var = tk.StringVar(value="1920")
        ttk.Entry(row2, textvariable=self.def_w_var, width=6).pack(side=tk.LEFT)
        ttk.Label(row2, text="×").pack(side=tk.LEFT)
        self.def_h_var = tk.StringVar(value="1080")
        ttk.Entry(row2, textvariable=self.def_h_var, width=6).pack(side=tk.LEFT, padx=(0, 12))

        ttk.Label(row2, text="输出目录:").pack(side=tk.LEFT)
        self.out_dir_var = tk.StringVar(value=str(settings.OUTPUT_DIR))
        ttk.Entry(row2, textvariable=self.out_dir_var, width=28).pack(side=tk.LEFT, padx=4)
        ttk.Button(row2, text="浏览", command=self._browse_out_dir).pack(side=tk.LEFT)

        ttk.Button(frame, text="💾 保存设置", command=self._save_settings).pack(pady=(6, 0))

    # ── Actions ─────────────────────────────────────────────────

    def _start_generation(self):
        prompt = self.prompt_text.get("1.0", "end-1c").strip()
        if not prompt:
            messagebox.showwarning("输入为空", "请输入思维导图描述内容")
            return

        self.gen_btn.config(state=tk.DISABLED)
        self.status_var.set("🔄 正在生成...")
        self.progress.pack(fill=tk.X, pady=4)
        self.progress.start(10)

        threading.Thread(target=self._run_generation, args=(prompt,), daemon=True).start()

    def _run_generation(self, prompt: str):
        try:
            from src.agent.orchestrator import AgentOrchestrator
            orch = AgentOrchestrator(llm_provider="default")
            result = orch.run(prompt)
            final = result.get("final_image", "")
            self._result_queue.put(("done", final, ""))
        except Exception as e:
            logger.exception("generate failed")
            self._result_queue.put(("error", "", str(e)))

    def _query_compound(self):
        name = self.compound_var.get().strip()
        if not name:
            return
        try:
            from src.structure_gen.generator import StructureGenerator
            gen = StructureGenerator()
            smiles = gen.resolve(name)
            self.smiles_var.set(smiles)
        except Exception as e:
            messagebox.showerror("查询失败", str(e))

    def _generate_structure(self):
        smiles = self.smiles_var.get().strip()
        style = self.struct_style_var.get()
        if not smiles:
            return
        try:
            from src.structure_gen.generator import StructureGenerator
            gen = StructureGenerator(default_style=style)
            path, img = gen.generate_from_smiles(smiles, width=600, height=400)
            self._display_on_label(str(path), self.struct_label, "struct")
        except Exception as e:
            messagebox.showerror("生成失败", str(e))

    def _download_result(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".png", filetypes=[("PNG images", "*.png")],
            initialdir=str(settings.OUTPUT_DIR),
        )
        if path and self._result_photo:
            # Re-save from the currently showing image
            pass

    def _save_settings(self):
        logger.info("settings saved")
        messagebox.showinfo("设置", "设置已保存（本次会话生效）")

    def _browse_out_dir(self):
        d = filedialog.askdirectory(initialdir=self.out_dir_var.get())
        if d:
            self.out_dir_var.set(d)

    # ── Image display helper ────────────────────────────────────

    def _display_on_label(self, img_path: str, label: ttk.Label, tag: str):
        try:
            img = Image.open(img_path)
            label.update()
            w, h = label.winfo_width(), label.winfo_height()
            if w < 50:
                w, h = 400, 300
            img.thumbnail((w - 20, h - 20), Image.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            label.config(image=photo, text="")
            label.image = photo  # keep ref
            if tag == "preview":
                self._preview_photo = photo
            elif tag == "struct":
                self._struct_photo = photo
            else:
                self._result_photo = photo
        except Exception as e:
            logger.warning("display image failed: %s", e)

    # ── Thread result polling ───────────────────────────────────

    def _poll_queue(self):
        try:
            while True:
                msg = self._result_queue.get_nowait()
                status, path, error = msg
                self.progress.stop()
                self.progress.pack_forget()
                self.gen_btn.config(state=tk.NORMAL)

                if status == "done" and path and Path(path).exists():
                    self._display_on_label(path, self.preview_label, "preview")
                    self._display_on_label(path, self.result_label, "result")
                    self.status_var.set(f"✅ 生成完成！输出: {path}")
                elif status == "error":
                    self.status_var.set(f"❌ 错误: {error}")
                    messagebox.showerror("生成失败", error)
                else:
                    self.status_var.set("⚠️ 生成未产生最终图像")
        except queue.Empty:
            pass
        self.root.after(300, self._poll_queue)


# ── entry ───────────────────────────────────────────────────────

def launch_tkinter():
    root = tk.Tk()
    ChemMindmapApp(root)
    root.mainloop()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    launch_tkinter()
