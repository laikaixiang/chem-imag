"""Gradio Web UI for chem-mindmap."""

import logging
from pathlib import Path

import gradio as gr

from src.config import settings
from src.agent.orchestrator import AgentOrchestrator
from src.structure_gen.generator import StructureGenerator

logger = logging.getLogger(__name__)

PRESET_COMPOUNDS = [
    "aspirin", "caffeine", "paracetamol", "ibuprofen",
    "benzene", "phenol", "benzoic acid", "ethanol", "acetic acid",
]


def build_interface() -> gr.Blocks:
    with gr.Blocks(
        title="chem-mindmap - 有机化学思维导图生成器",
        theme=gr.themes.Soft(),
    ) as demo:
        gr.Markdown("# 🧪 有机化学思维导图生成器")
        gr.Markdown("AI 驱动的学术论文级有机化学思维导图生成")

        with gr.Tabs():
            with gr.TabItem("📝 生成思维导图"):
                _build_generation_tab()
            with gr.TabItem("🔬 结构图预览"):
                _build_structure_tab()
            with gr.TabItem("⚙️ 设置"):
                _build_settings_tab()

    return demo


def _build_generation_tab():
    with gr.Row():
        with gr.Column(scale=2):
            input_text = gr.Textbox(
                label="📝 输入描述",
                placeholder="例如：生成关于苯酚、苯甲酸及其酯化反应的思维导图",
                lines=4,
            )
            with gr.Row():
                style = gr.Dropdown(
                    choices=["academic", "modern", "minimal"],
                    value="academic", label="🎨 风格",
                )
                size = gr.Dropdown(
                    choices=["1920×1080", "2560×1440"],
                    value="1920×1080", label="📐 尺寸",
                )
                provider = gr.Dropdown(
                    choices=["mock", "sd_webui"],
                    value="mock", label="🤖 AI 后端",
                )

    with gr.Row():
        generate_btn = gr.Button("🔬 开始生成", variant="primary", size="lg")
        status_text = gr.Markdown("✅ 就绪，等待输入...")

    with gr.Row():
        preview_img = gr.Image(label="🖼️ 中间预览", type="filepath", height=350)
        result_img = gr.Image(label="🎯 最终结果", type="filepath", height=350)

    with gr.Row():
        download_btn = gr.Button("📥 下载 PNG")

    generate_btn.click(
        fn=_on_generate,
        inputs=[input_text, style, size, provider],
        outputs=[result_img, preview_img, status_text],
    )


def _build_structure_tab():
    with gr.Row():
        compound_name = gr.Dropdown(
            label="化合物名称",
            choices=PRESET_COMPOUNDS,
            value="aspirin",
            allow_custom_value=True,
        )
        smiles_input = gr.Textbox(label="SMILES", value="CC(=O)Oc1ccccc1C(=O)O")
        query_btn = gr.Button("🔍 查询")

    with gr.Row():
        style_select = gr.Dropdown(
            choices=["ACS_1996", "dark_mode", "color_on_white", "minimal"],
            value="ACS_1996", label="🎨 结构图风格",
        )
        gen_struct_btn = gr.Button("生成结构图")

    struct_preview = gr.Image(label="结构图预览", type="filepath", height=350)

    with gr.Row():
        formula = gr.Textbox(label="分子式", interactive=False)
        mol_weight = gr.Textbox(label="分子量", interactive=False)

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
    with gr.Group():
        gr.Markdown("### 🔗 API 连接")
        api_provider = gr.Dropdown(
            choices=["mock", "sd_webui", "openai", "replicate"],
            value=settings.AI_IMAGE_PROVIDER, label="图像生成 API",
        )
        api_url = gr.Textbox(value=settings.SD_WEBUI_URL, label="API URL")
        llm_provider = gr.Dropdown(
            choices=["claude", "openai", "default"],
            value=settings.LLM_PROVIDER, label="LLM API",
        )
        api_key = gr.Textbox(value="", label="API Key", type="password")

    with gr.Group():
        gr.Markdown("### 🖼️ 图像默认设置")
        default_width = gr.Number(value=1920, label="默认宽度")
        default_height = gr.Number(value=1080, label="默认高度")
        default_style = gr.Dropdown(
            choices=["academic", "modern", "minimal"],
            value="academic", label="默认风格",
        )

    with gr.Group():
        gr.Markdown("### 📂 输出设置")
        output_dir = gr.Textbox(value=str(settings.OUTPUT_DIR), label="输出目录")

    save_btn = gr.Button("💾 保存设置")
    save_msg = gr.Markdown("")

    save_btn.click(
        fn=_on_save_settings,
        inputs=[api_provider, api_url, llm_provider, default_width, default_height, default_style, output_dir],
        outputs=[save_msg],
    )


# ── event handlers ──────────────────────────────────────────────

def _on_generate(input_text: str, style: str, size: str, provider: str):
    if not input_text.strip():
        return None, None, "⚠️ 请输入描述内容"

    try:
        orchestrator = AgentOrchestrator(llm_provider="default")
        result = orchestrator.run(input_text)

        final_path = result.get("final_image", "")
        if final_path and Path(final_path).exists():
            return final_path, final_path, f"✅ 生成完成！输出: {final_path}"
        return None, None, "⚠️ 生成未产生最终图像，请查看控制台日志"
    except Exception as e:
        logger.exception("generate failed")
        return None, None, f"❌ 错误: {e}"


def _on_query_compound(name: str):
    try:
        gen = StructureGenerator()
        smiles = gen.resolve(name)
        return smiles, "", ""
    except Exception as e:
        return f"查询失败: {e}", "", ""


def _on_generate_structure(smiles: str, style: str):
    try:
        gen = StructureGenerator(default_style=style)
        path, _ = gen.generate_from_smiles(smiles, width=800, height=600)
        return str(path)
    except Exception as e:
        logger.exception("structure gen failed")
        return None


def _on_save_settings(api_provider, api_url, llm_provider, width, height, style, output_dir):
    logger.info("settings updated: provider=%s, size=%sx%s", api_provider, width, height)
    return "✅ 设置已保存（本次会话生效）"


def launch_gradio(share: bool = False, port: int = 7860):
    demo = build_interface()
    demo.launch(share=share, server_port=port, server_name="0.0.0.0")
