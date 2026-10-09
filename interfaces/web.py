"""交互层：Gradio 网页 Demo。

只负责界面与输入校验，业务逻辑全部委托给 pipeline.analyze_image()。
"""
import os
import sys
from collections import Counter

import gradio as gr

# 兼容 `python interfaces/web.py` 直接启动：把项目根目录加入模块搜索路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import settings
from pipeline.analyze import analyze_image

# 当前运行模式（决定报告来源），用于界面提示
MODE = "云端大模型（多模态）" if settings.LLM_API_KEY else "离线模板（未配置 API Key）"

# 示例图：位于项目 assets 目录（首次启动前确保存在）
EXAMPLE_DIR = os.path.join(os.path.dirname(__file__), "..", "assets")
_EXAMPLE_FILES = [
    os.path.join(EXAMPLE_DIR, "bus.jpg"),
    os.path.join(EXAMPLE_DIR, "zidane.jpg"),
]
EXAMPLES = [p for p in _EXAMPLE_FILES if os.path.exists(p)]

# 系统架构图（存在时在"使用说明"页展示）
ARCH_IMG = os.path.join(EXAMPLE_DIR, "architecture.png")

HEADER = f"""# 🛡️ 城市安全之眼
<div style="font-size:16px;color:#64748b;margin-top:6px">
基于 <b>YOLOv8 轻量检测</b> + <b>云端多模态大模型</b> 的智慧城市安全分析 Demo ｜ 当前模式：<b>{MODE}</b>
</div>
"""

USAGE = f"""## 使用说明

### 快速上手
1. 左侧上传一张**街景 / 监控图片**，或点击示例图一键试用。
2. 点击 **🔍 开始分析**，右侧展示**带检测框的结果图**，并自动生成《城市安全事件分析报告》。

### 配置云端大模型（获得真正的智能报告）
当前为 `{MODE}`。把 `.env.example` 复制为 `.env`，填入移动云 / 九天大模型的
`LLM_BASE_URL / LLM_API_KEY / LLM_MODEL` 后重启服务即可，无需改动任何代码。

### 技术架构（分层解耦）
- **配置层** `config/`：模型、API、阈值、中文标签映射
- **检测层** `core/detector.py`：YOLOv8n 本地轻量推理
- **理解层** `core/reporter.py` + **兜底层** `core/template.py`：多模态报告 / 离线模板
- **工具层** `utils/`：画框、图片读写
- **编排层** `pipeline/`：检测→理解→画框串联
- **交互层** `interfaces/`：本网页

### 检测能力
可识别：行人、自行车/非机动车、摩托车、小汽车、公交车、卡车、交通信号灯、消防栓、停止标志、停车计费器、公共座椅等，并对中文标签做映射。
"""

CSS = """
.gradio-container { max-width: 1080px !important; margin: auto !important; }
.gr-button-primary { font-weight: 600; }
footer { visibility: hidden; }
#header { margin-bottom: 4px; }
"""


def build_summary(detections):
    """把检测结果汇总成概览 Markdown。"""
    if not detections:
        return "🚫 本次未检测到目标。"
    cnt = Counter(d["label_zh"] for d in detections)
    return "\n".join(f"- **{k}**：{v} 个" for k, v in cnt.items())


def analyze(image):
    if image is None:
        return None, "请先上传图片，或点击示例图试用。", "等待输入…"
    try:
        out_path, report, detections = analyze_image(image)
    except Exception as e:  # noqa: BLE001
        return None, f"分析出错：{e}", ""
    return out_path, report, build_summary(detections)


theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="cyan",
    neutral_hue="slate",
)

with gr.Blocks(title="城市安全之眼") as demo:
    gr.Markdown(HEADER, elem_id="header")
    with gr.Tabs():
        with gr.Tab("📷 图片安全分析"):
            with gr.Row(equal_height=False):
                with gr.Column(scale=1):
                    img = gr.Image(label="上传街景 / 监控图片", type="numpy", height=360)
                    with gr.Row():
                        btn = gr.Button("🔍 开始分析", variant="primary", scale=3)
                        clear_btn = gr.ClearButton(img, scale=1)
                    if EXAMPLES:
                        gr.Examples(examples=EXAMPLES, inputs=img, label="或一键试用示例")
                with gr.Column(scale=1):
                    out_img = gr.Image(label="检测结果（带检测框）", type="filepath", height=360)
                    gr.Markdown("### 📊 检测概览")
                    summary = gr.Markdown("等待分析…")
            gr.Markdown("### 📝 城市安全事件分析报告")
            report = gr.Markdown("上传图片并点击「开始分析」后，将在此生成报告。")
            btn.click(analyze, [img], [out_img, report, summary])

        with gr.Tab("📖 使用说明"):
            if os.path.exists(ARCH_IMG):
                gr.Image(value=ARCH_IMG, label="系统分层架构", interactive=False, height=420)
            gr.Markdown(USAGE)


if __name__ == "__main__":
    # 启动前预热检测模型，避免首次请求冷启动延迟（缓解公网隧道超时）
    try:
        from core.detector import get_model
        get_model()
    except Exception:  # noqa: BLE001
        pass

    # server_name 绑定 0.0.0.0 才能在云服务器上通过公网/内网 IP 访问；
    # 需要临时外网演示可设置 GRADIO_SHARE=true（会生成 gradio.live 临时公网链接，
    # 走的是出向隧道，不受云服务器安全组/防火墙限制）。
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("GRADIO_PORT", "7860")),
        share=os.getenv("GRADIO_SHARE", "false").lower() == "true",
        theme=theme,
        css=CSS,
    )
