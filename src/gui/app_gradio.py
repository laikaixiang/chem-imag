"""Gradio Web UI for chem-mindmap."""

import logging
from pathlib import Path

import gradio as gr

from src.config import settings, api_config
from src.pipeline import ChemicalImagePipeline
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
                    choices=["1024×768", "1920×1080"],
                    value="1024×768", label="📐 尺寸",
                )
                providers = api_config.available_figure_providers
                default_provider = providers[0] if providers else "mock"
                provider = gr.Dropdown(
                    choices=providers,
                    value=default_provider, label="🤖 AI 图像后端",
                )

    with gr.Row():
        generate_btn = gr.Button("🔬 开始生成", variant="primary", size="lg")
        status_text = gr.Markdown("✅ 就绪，等待输入...")

    with gr.Row():
        scene_img = gr.Image(label="🖼️ 场景图", type="filepath", height=400)
        result_img = gr.Image(label="🎯 最终结果", type="filepath", height=400)

    generate_btn.click(
        fn=_on_generate,
        inputs=[input_text, style, size, provider],
        outputs=[result_img, scene_img, status_text],
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
    talk_providers = api_config.available_talk_providers
    figure_providers = api_config.available_figure_providers

    with gr.Group():
        gr.Markdown("### 🔗 API 连接")
        figure_provider = gr.Dropdown(
            choices=figure_providers,
            value=figure_providers[0] if figure_providers else "mock",
            label="图像生成 API",
        )
        api_url = gr.Textbox(value=api_config.figure_url, label="API URL")
        talk_provider = gr.Dropdown(
            choices=talk_providers,
            value=talk_providers[0] if talk_providers else "",
            label="LLM API（对话/提取）",
        )
        api_key = gr.Textbox(value="", label="API Key", type="password")

    with gr.Group():
        gr.Markdown("### 🖼️ 图像默认设置")
        default_width = gr.Number(value=1024, label="默认宽度")
        default_height = gr.Number(value=768, label="默认高度")
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
        inputs=[figure_provider, api_url, talk_provider, default_width, default_height, default_style, output_dir],
        outputs=[save_msg],
    )


# ── event handlers ──────────────────────────────────────────────

def _on_generate(input_text: str, style: str, size: str, provider: str):
    if not input_text.strip():
        return None, None, "⚠️ 请输入描述内容"

    try:
        w_str, h_str = size.split("×")
        width, height = int(w_str), int(h_str)

        pipe = ChemicalImagePipeline(image_provider=provider)
        result = pipe.generate(
            user_input=input_text,
            style=style,
            width=width,
            height=height,
        )

        final_path = result.get("final_image", "")
        scene_path = result.get("scene_path", "")
        scene_out = scene_path if scene_path and Path(scene_path).exists() else None
        final_out = final_path if final_path and Path(final_path).exists() else None

        if final_out:
            compounds_str = ", ".join(result.get("compounds", []))
            return final_out, scene_out, f"✅ 生成完成！化合物: {compounds_str}\n输出: {final_path}"
        return None, scene_out, "⚠️ 生成未产生最终图像，请查看控制台日志"
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
