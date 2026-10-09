"""生成系统分层架构图 assets/architecture.png（可复现）。

不依赖任何外部数据，纯本地绘制。运行：python tools/generate_architecture.py
"""
import os

import matplotlib
from matplotlib import font_manager

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSET_DIR = os.path.join(ROOT, "assets")
OUT = os.path.join(ASSET_DIR, "architecture.png")
os.makedirs(ASSET_DIR, exist_ok=True)

# 显式注册中文字体（环境已装 Noto Serif CJK SC，但 matplotlib 需手动加载 .ttc）
_CJK_PATH = "/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc"
if os.path.exists(_CJK_PATH):
    font_manager.fontManager.addfont(_CJK_PATH)
    _cjk_name = font_manager.FontProperties(fname=_CJK_PATH).get_name()
    plt.rcParams["font.family"] = [_cjk_name, "serif"]
plt.rcParams["axes.unicode_minus"] = False

fig, ax = plt.subplots(figsize=(10, 7.6))
ax.set_xlim(0, 10)
ax.set_ylim(0, 11)
ax.axis("off")

P = {
    "inter": "#2563eb",
    "pipe": "#4f46e5",
    "detect": "#0891b2",
    "understand": "#7c3aed",
    "fallback": "#db2777",
    "util": "#0d9488",
    "config": "#64748b",
}


def box(x, y, w, h, title, sub, fc):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                 boxstyle="round,pad=0.02,rounding_size=0.1",
                 linewidth=0, facecolor=fc, zorder=2))
    ax.text(x + w / 2, y + h / 2 + 0.18, title, ha="center", va="center",
            color="white", fontsize=13, fontweight="bold", zorder=3)
    ax.text(x + w / 2, y + h / 2 - 0.22, sub, ha="center", va="center",
            color="#e2e8f0", fontsize=8.5, zorder=3)


def arrow(x1, y1, x2, y2, color="#334155"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                 mutation_scale=14, color=color, lw=1.8, zorder=1))


# 各层方框
box(1.0, 8.6, 8.0, 0.9, "交互层", "interfaces/ · Gradio 网页（可扩展 API / 监测大屏）", P["inter"])
box(1.0, 7.3, 8.0, 0.9, "编排层", "pipeline.analyze · 串联 检测 → 理解 → 画框", P["pipe"])
box(1.0, 5.5, 3.7, 1.3, "检测层", "core/detector · YOLOv8n 本地轻量推理（CPU）", P["detect"])
box(5.3, 6.0, 3.7, 0.9, "理解层", "core/reporter · 云端多模态大模型", P["understand"])
box(5.3, 5.0, 3.7, 0.8, "兜底层", "core/template · 离线模板报告", P["fallback"])
box(1.0, 3.5, 8.0, 0.9, "工具层", "utils/ · 画框 / 图片读写 / 临时文件", P["util"])
box(1.0, 2.2, 8.0, 0.9, "配置层", "config/ · 模型 / API / 阈值 / 中文标签映射", P["config"])

# 数据流箭头
arrow(5.0, 8.6, 5.0, 8.2)        # 交互 → 编排
arrow(3.0, 7.3, 2.7, 6.8)        # 编排 → 检测
arrow(7.0, 7.3, 7.15, 6.9)       # 编排 → 理解
arrow(7.15, 6.0, 7.15, 5.8)      # 理解 → 兜底
arrow(2.7, 5.5, 4.5, 4.4)        # 检测 → 工具
arrow(7.15, 5.0, 6.5, 4.4)       # 兜底 → 工具
arrow(5.0, 3.5, 5.0, 3.1)        # 工具 → 配置

# 标题与说明
ax.text(5.0, 10.55, "城市安全之眼 · 系统分层架构", ha="center", va="center",
        fontsize=15, fontweight="bold", color="#0f172a")
ax.text(5.0, 10.15, "检测与理解解耦：YOLO 精确计数画框，多模态大模型负责语义报告",
        ha="center", va="center", fontsize=9.5, color="#475569")
ax.text(0.3, 1.45, "数据流：上传图片 → 检测 →（理解 / 兜底）→ 画框 → 报告", ha="left",
        va="center", fontsize=8.5, color="#475569")

plt.tight_layout()
plt.savefig(OUT, dpi=160, bbox_inches="tight", facecolor="white")
print("saved:", OUT)
