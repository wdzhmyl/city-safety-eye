"""交互层：Gradio 网页 Demo（演示智慧工地安全巡检 · 端云协同）。"""

import os
import sys
from collections import Counter

import gradio as gr

# 兼容 `python interfaces/web.py` 直接启动：把项目根目录加入模块搜索路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import settings
from pipeline.collab import analyze_image

# 当前运行模式（决定报告来源），用于界面提示
MODE = "云端大模型研判（多模态）" if settings.LLM_API_KEY else "本地离线兜底（未配置 API Key）"

# 示例图：位于项目 assets 目录（首次启动前确保存在）
EXAMPLE_DIR = os.path.join(os.path.dirname(__file__), "..", "assets")
_EXAMPLE_FILES = [
    os.path.join(EXAMPLE_DIR, "bus.jpg"),
    os.path.join(EXAMPLE_DIR, "zidane.jpg"),
]
EXAMPLES = [p for p in _EXAMPLE_FILES if os.path.exists(p)]

# 系统架构图（存在时在"使用说明"页展示）
ARCH_IMG = os.path.join(EXAMPLE_DIR, "architecture.png")

HEADER = f"""# 🚧 智慧工地安全巡检平台（端云协同）
<div style="font-size:16px;color:#64748b;margin-top:6px">
基于 <b>边缘端 YOLOv8 轻量推理</b> + <b>云端多模态大模型理解</b> 的跨域协同施工安全巡检 Demo（智算方向）｜ 当前模式：<b>{MODE}</b>
</div>
"""

USAGE = f"""## 使用说明

### 快速上手
1. 左侧上传一张**工地现场图片**，或点击示例图一键试用。
2. 点击 **🔍 开始巡检**，右侧展示**带检测框的结果图**（危险作业区以橙框标出，闯入目标标红），并自动生成《施工安全巡检报告》。

### 危险作业区规则
画面右侧默认圈定一块"危险作业区"，**作业人员 / 工程车辆**等进入该区域即被记为闯入违规（标红预警）。
区域可在 `config/settings.py` 的 `DANGER_ZONE` 中按你的监控画面调整，或设 `ENABLE_DANGER_ZONE=False` 关闭。

### 配置云端大模型（获得真正的智能研判）
当前为 `{MODE}`。把 `.env.example` 复制为 `.env`，填入移动云 / 九天大模型的
`LLM_BASE_URL / LLM_API_KEY / LLM_MODEL` 后重启服务即可，无需改动任何代码。

### 技术架构（分层解耦 · 端云协同）
- **配置层** `config/`：场景、模型、API、阈值、标签与危险区规则
- **边缘推理层** `core/edge_infer.py`：YOLOv8n 本地轻量推理 + 危险区闯入判定（CPU）
- **云端理解层** `core/cloud_reason.py` + **离线兜底层** `core/fallback_report.py`：施工安全报告 / 离线兜底
- **工具层** `utils/`：可视化、图片读写
- **编排层** `pipeline/collab.py`：边缘推理→危险区规则→云端理解→可视化 串联
- **交互层** `interfaces/`：本网页

### 巡检能力（工地语境）
边缘端可识别并映射为：作业人员、非机动车、摩托车、通勤车辆、通勤班车、工程运输车、现场信号灯、消防栓、警示标识等，
并对"危险作业区闯入"做实时规则化预警。
"""

CSS = """
.gradio-container { max-width: 1080px !important; margin: auto !important; }
.gr-button-primary { font-weight: 600; }
footer { visibility: hidden; }
#header { margin-bottom: 4px; }
"""


def build_summary(detections, intrusions=None):
    """把推理结果汇总成概览 Markdown（含危险区闯入预警）。"""
    if not detections:
        return "🚫 本次未检测到目标。"
    cnt = Counter(d["label_zh"] for d in detections)
    lines = "\n".join(f"- **{k}**：{v} 个" for k, v in cnt.items())
    if intrusions:
        lines += (f"\n\n⚠️ **危险作业区闯入 {len(intrusions)} 起**（"
                  + "、".join(d["label_zh"] for d in intrusions) + "），请立即核查！")
    return lines


def analyze(image):
    if image is None:
        return None, "请先上传图片，或点击示例图试用。", "等待输入…"
    try:
        out_path, report, detections, intrusions = analyze_image(image)
    except Exception as e:  # noqa: BLE001
        return None, f"推理出错：{e}", ""
    return out_path, report, build_summary(detections, intrusions)


theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="cyan",
    neutral_hue="slate",
)

with gr.Blocks(title="智慧工地安全巡检平台") as demo:
    gr.Markdown(HEADER, elem_id="header")
    with gr.Tabs():
        with gr.Tab("📷 工地图片巡检"):
            with gr.Row(equal_height=False):
                with gr.Column(scale=1):
                    img = gr.Image(label="上传工地图片（边缘端实时检测）", type="numpy", height=360)
                    with gr.Row():
                        btn = gr.Button("🔍 开始巡检", variant="primary", scale=3)
                        clear_btn = gr.ClearButton(img, scale=1)
                    if EXAMPLES:
                        gr.Examples(examples=EXAMPLES, inputs=img, label="或一键试用示例")
                with gr.Column(scale=1):
                    out_img = gr.Image(label="巡检结果（检测框 / 危险区）", type="filepath", height=360)
                    gr.Markdown("### 📊 巡检概览")
                    summary = gr.Markdown("等待巡检…")
            gr.Markdown("### 📝 施工安全巡检报告")
            report = gr.Markdown("上传图片并点击「开始巡检」后，将在此生成报告。")
            btn.click(analyze, [img], [out_img, report, summary])

        with gr.Tab("📖 使用说明"):
            if os.path.exists(ARCH_IMG):
                gr.Image(value=ARCH_IMG, label="端云协同施工安全巡检架构", interactive=False, height=420)
            gr.Markdown(USAGE)


if __name__ == "__main__":
    # 启动前预热边缘推理模型，避免首次请求冷启动延迟（缓解公网隧道超时）
    try:
        from core.edge_infer import get_model
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
