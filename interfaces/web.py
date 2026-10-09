"""交互层：Gradio 网页 Demo（演示智慧工地安全巡检 · 端云协同）。

含两个页面：
- 「工地图片巡检」：单张图片上传分析（云端/兜底报告）。
- 「🎥 实时监控」：**解耦的摄像头监控**——前端可搜索/选择摄像头（本地 / 手机 / 网络 RTSP），
  服务端逐帧边缘检测 + 危险区预警，支持**主动截屏**与**异常自动截屏存证**。
"""

import os
import sys
import tempfile
import threading
import time
from collections import Counter

import cv2
import gradio as gr

# 兼容 `python interfaces/web.py` 直接启动：把项目根目录加入模块搜索路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import settings
from pipeline.collab import analyze_image
from core.edge_infer import infer, check_danger_zone
from core.cloud_reason import generate_report
from utils.visualize import draw_detections
from utils.camera import CameraManager

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

### 快速上手（单图分析）
1. 左侧上传一张**工地现场图片**，或点击示例图一键试用。
2. 点击 **🔍 开始巡检**，右侧展示**带检测框的结果图**（危险作业区以橙框标出，闯入目标标红），并自动生成《施工安全巡检报告》。

### 🎥 实时监控（解耦摄像头 + 截屏存证）
切换到「🎥 实时监控」页：
1. 点 **🔍 搜索摄像头** 自动发现本机摄像头；
2. 手机的摄像头请用 **IP 摄像头 App**（如 IP Webcam / DroidCam）把手机变成网络视频流，
   在"添加网络摄像头地址"框粘贴地址（如 `http://手机IP:8080/video` 或 `rtsp://...`），点 ➕ 添加；
3. 在下拉框**选择摄像头** → 点 **▶ 开启监控**，即开始边缘端逐帧实时检测；
4. 检测到**危险作业区闯入**时会**自动截屏存证**（标记问题），也可随时点 **📸 主动截屏**；
5. 右侧画廊与记录可回看所有截图，便于专业管理。

### 危险作业区规则
画面右侧默认圈定一块"危险作业区"，**作业人员 / 工程车辆**等进入该区域即被记为闯入违规（标红预警）。
区域可在 `config/settings.py` 的 `DANGER_ZONE` 中按你的监控画面调整，或设 `ENABLE_DANGER_ZONE=False` 关闭。

### 配置云端大模型（获得真正的智能研判）
当前为 `{MODE}`。把 `.env.example` 复制为 `.env`，填入移动云 / 九天大模型的
`LLM_BASE_URL / LLM_API_KEY / LLM_MODEL` 后重启服务即可，无需改动任何代码。

### 技术架构（分层解耦 · 端云协同）
- **配置层** `config/`：场景、模型、API、阈值、标签与危险区规则
- **边缘推理层** `core/edge_infer.py`：YOLOv8n 本地轻量推理 + 危险区闯入判定（CPU）
- **工具层** `utils/camera.py`：与 UI 解耦的摄像头管理（本地发现 / 网络流）
- **云端理解层** `core/cloud_reason.py` + **离线兜底层** `core/fallback_report.py`：施工安全报告 / 离线兜底
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

# ===== 实时监控：解耦摄像头 + 状态（线程安全）=====
_MONITOR_DIR = tempfile.mkdtemp(prefix="eci_monitor_")
_FRAME_PATH = os.path.join(_MONITOR_DIR, "frame.jpg")   # 每帧覆盖写入（RGB，便于可视化与云端看图）
_OUT_PATH = os.path.join(_MONITOR_DIR, "out.jpg")
SCREENSHOT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "screenshots")
os.makedirs(SCREENSHOT_DIR, exist_ok=True)

cam_mgr = CameraManager()
_lock = threading.Lock()
_state_rt = {                       # 实时监控共享状态
    "monitoring": False,
    "annotated": None,              # 当前标注帧（RGB numpy），用于前端推流
    "raw": None,                    # 当前原始帧（BGR numpy），用于截屏
    "alert": "未开启监控",
    "detections": [],
    "intrusions": [],
}
SOURCE_CHOICES = []                 # [(value, label), ...] 摄像头源列表
NET_SOURCES = []                    # 用户添加的网络/手机流地址
shot_log = []                       # (path, tag, ts, note)


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


def build_shot_md():
    if not shot_log:
        return "#### 截图记录\n暂无截图"
    lines = ["#### 截图记录"] + [
        f"- [{tag}] {ts} · {note}" for _, tag, ts, note in shot_log[-12:][::-1]
    ]
    return "\n".join(lines)


def do_discover():
    """搜索本机摄像头 + 合并已添加的网络源，刷新下拉列表。"""
    idxs = cam_mgr.discover()
    local = [(f"local:{i}", f"本地摄像头 {i}") for i in idxs]
    net = [(f"net:{u}", f"网络摄像头 {u}") for u in NET_SOURCES]
    SOURCE_CHOICES.clear()
    SOURCE_CHOICES.extend(local + net)
    choices = [label for _, label in SOURCE_CHOICES]
    value = SOURCE_CHOICES[0][0] if SOURCE_CHOICES else None
    if not SOURCE_CHOICES:
        return gr.update(choices=["（未找到可用摄像头）"], value=None)
    return source_dd.update(choices=choices, value=value)


def add_network(url):
    """添加网络/手机摄像头流地址（RTSP 或 http 视频流）。"""
    url = (url or "").strip()
    if not url:
        return gr.update(), "请输入摄像头地址（如 http://手机IP:8080/video 或 rtsp://...）"
    if url not in NET_SOURCES:
        NET_SOURCES.append(url)
        local = [(v, l) for v, l in SOURCE_CHOICES if v.startswith("local:")]
        net = [(f"net:{u}", f"网络摄像头 {u}") for u in NET_SOURCES]
        SOURCE_CHOICES.clear()
        SOURCE_CHOICES.extend(local + net)
    return gr.update(choices=[l for _, l in SOURCE_CHOICES], value=f"net:{url}"), f"✅ 已添加网络摄像头：{url}"


def start_monitor(value):
    """打开所选摄像头并开始监控。"""
    if not value:
        return "请先搜索并选择摄像头"
    kind, _, arg = value.partition(":")
    ok = cam_mgr.open(int(arg) if kind == "local" else arg)
    if not ok:
        return f"⚠️ 打开失败：{value}（请确认设备或地址可用，手机需先开启 IP 摄像头 App）"
    with _lock:
        _state_rt["monitoring"] = True
    return f"✅ 已开启监控：{value}"


def stop_monitor():
    with _lock:
        _state_rt["monitoring"] = False
    cam_mgr.release()
    return "⏹ 已停止监控"


def save_screenshot(mark=True, auto=False):
    """保存当前帧为截图。mark=True 时叠加检测框与问题标记（危险区/闯入标红）。"""
    with _lock:
        raw = _state_rt["raw"]
        dets = _state_rt["detections"]
        intr = _state_rt["intrusions"]
        monitoring = _state_rt["monitoring"]
    if raw is None or not monitoring:
        return None
    ts = time.strftime("%Y%m%d_%H%M%S")
    tag = "自动" if auto else "主动"
    note = "危险区闯入" if intr else "常规巡检"
    path = os.path.join(SCREENSHOT_DIR, f"{tag}_{ts}.jpg")
    if mark:
        tmp = os.path.join(_MONITOR_DIR, "shot_raw.jpg")
        cv2.imwrite(tmp, cv2.cvtColor(raw, cv2.COLOR_BGR2RGB))
        draw_detections(tmp, dets, path)
    else:
        cv2.imwrite(path, cv2.cvtColor(raw, cv2.COLOR_BGR2RGB))
    shot_log.append((path, tag, ts, note))
    return path


def manual_shot():
    p = save_screenshot(mark=True, auto=False)
    if p is None:
        return "请先开启监控再截屏", build_shot_md()
    return f"📸 已主动截屏：{os.path.basename(p)}", build_shot_md()


def make_report_rt():
    """基于当前画面生成云端/兜底研判报告（端云协同：边缘盯防、云端按需想）。"""
    with _lock:
        dets = _state_rt["detections"]
        intr = _state_rt["intrusions"]
        monitoring = _state_rt["monitoring"]
    if not monitoring:
        return "请先开启监控"
    return generate_report(dets, _FRAME_PATH, intrusions=intr)


def refresh_ui():
    """定时刷新告警、截图画廊与记录（由 gr.Timer 调用）。"""
    with _lock:
        alert = _state_rt["alert"]
    gallery = [p for p, _, _, _ in shot_log][-12:][::-1]
    return alert, gallery, build_shot_md()


def stream_gen():
    """持续向前端推流当前标注帧（未开启监控时推送 None）。"""
    while True:
        with _lock:
            monitoring = _state_rt["monitoring"]
            annotated = _state_rt["annotated"]
        yield annotated if (monitoring and annotated is not None) else None
        time.sleep(0.03)


def processing_loop():
    """后台采集线程：逐帧边缘检测 + 危险区规则 + 自动截屏（解耦，无需前端干预）。"""
    last_auto = 0.0
    while True:
        with _lock:
            monitoring = _state_rt["monitoring"]
        if not monitoring:
            time.sleep(0.1)
            continue
        frame = cam_mgr.read()
        if frame is None:
            time.sleep(0.05)
            continue
        # 检测（直接传 numpy 帧，免落盘）
        detections, _ = infer(frame)
        h, w = frame.shape[:2]
        intrusions, _ = check_danger_zone(detections, (w, h))
        # 保存 RGB 帧用于可视化/云端看图，并绘制标注
        cv2.imwrite(_FRAME_PATH, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw_detections(_FRAME_PATH, detections, _OUT_PATH)
        annotated = cv2.cvtColor(cv2.imread(_OUT_PATH), cv2.COLOR_BGR2RGB)
        alert = build_summary(detections, intrusions)
        with _lock:
            _state_rt["annotated"] = annotated
            _state_rt["raw"] = frame.copy()
            _state_rt["detections"] = detections
            _state_rt["intrusions"] = intrusions
            _state_rt["alert"] = alert
        # 危险区闯入时自动截屏存证（3 秒内最多一次，避免刷屏）
        now = time.time()
        if intrusions and now - last_auto > 3:
            save_screenshot(mark=True, auto=True)
            last_auto = now
        time.sleep(0.03)


# 启动后台采集线程（守护线程，随进程退出）
threading.Thread(target=processing_loop, daemon=True).start()


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

        with gr.Tab("🎥 实时监控"):
            with gr.Row(equal_height=False):
                with gr.Column(scale=1):
                    with gr.Row():
                        discover_btn = gr.Button("🔍 搜索摄像头")
                        add_url = gr.Textbox(
                            label="添加网络/手机摄像头地址",
                            placeholder="如 http://手机IP:8080/video 或 rtsp://...",
                            scale=3,
                        )
                        add_btn = gr.Button("➕ 添加", scale=1)
                    source_dd = gr.Dropdown(
                        label="选择摄像头（本地 / 手机 / 网络）",
                        choices=[], value=None, interactive=True,
                    )
                    with gr.Row():
                        start_btn = gr.Button("▶ 开启监控", variant="primary", scale=2)
                        stop_btn = gr.Button("⏹ 停止监控", scale=1)
                    shot_btn = gr.Button("📸 主动截屏（标记问题）")
                    report_btn = gr.Button("📝 生成云端研判报告")
                    status_md = gr.Markdown("点击「搜索摄像头」先发现本机设备；手机请先用 IP 摄像头 App 再粘贴地址添加。")
                with gr.Column(scale=1):
                    live_img = gr.Image(
                        label="实时画面（边缘端逐帧检测 / 危险区预警）", type="numpy", height=360,
                    )
                    alert_md = gr.Markdown("未开启监控")
                    gr.Markdown("### 📸 截图与告警记录（自动/主动存证）")
                    gallery = gr.Gallery(label="截图回看", columns=3, height=240)
                    shot_md = gr.Markdown(build_shot_md())
                    report_md = gr.Markdown(
                        "点「生成云端研判报告」获取当前画面研判（检测到危险区闯入时建议立即生成）。")
            # 摄像头搜索 / 添加 / 开关
            discover_btn.click(do_discover, None, [source_dd])
            add_btn.click(add_network, [add_url], [source_dd, status_md])
            start_btn.click(start_monitor, [source_dd], [status_md])
            stop_btn.click(stop_monitor, None, [status_md])
            # 截屏 / 云端报告
            shot_btn.click(manual_shot, None, [status_md, shot_md])
            report_btn.click(make_report_rt, None, [report_md])
            # 前端推流 + 定时刷新（告警/画廊/记录）
            demo.load(stream_gen, None, [live_img])
            if hasattr(gr, "Timer"):
                timer = gr.Timer(1.0)
                timer.tick(refresh_ui, None, [alert_md, gallery, shot_md])

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
