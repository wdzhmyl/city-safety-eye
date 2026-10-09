"""配置层：集中管理模型、API 与推理参数（智慧工地安全巡检场景）。

所有可调项集中在此，其余各层只依赖本模块，便于统一修改与环境隔离。
可通过同目录 .env 文件或系统环境变量覆盖（推荐环境变量，避免泄露密钥）。
"""
import os
from dotenv import load_dotenv

load_dotenv()

# ===== 场景定位 =====
SCENARIO = "智慧工地安全巡检"

# ===== 云端大模型配置（OpenAI 兼容接口）=====
# 移动云 / 九天大模型通常提供 OpenAI 兼容的 /v1 端点。
# 在移动云大赛账号后台拿到 base_url / api_key / model 后，
# 可写入同目录 .env 文件，或设置系统环境变量（推荐，避免泄露密钥）。
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://your-mobilecloud-llm-endpoint/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")   # 留空 -> 自动回退本地模板报告，demo 仍可跑
LLM_MODEL = os.getenv("LLM_MODEL", "your-model-name")
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "800"))

# ===== 本地检测模型（轻量，CPU 可跑）=====
DETECTION_MODEL = os.getenv("DETECTION_MODEL", "yolov8n.pt")   # 首次运行自动下载
DETECTION_CONF = float(os.getenv("DETECTION_CONF", "0.35"))    # 置信度阈值

# ===== 中文标签映射：把通用检测类别重映射到工地语境 =====
LABEL_ZH = {
    "person": "作业人员",
    "bicycle": "非机动车",
    "motorcycle": "摩托车",
    "car": "通勤车辆",
    "bus": "通勤班车",
    "truck": "工程运输车",
    "traffic light": "现场信号灯",
    "fire hydrant": "消防栓",
    "stop sign": "警示标识",
    "parking meter": "停车计费器",
    "bench": "临时设施",
}

# ===== 危险作业区（禁入 / 闯入预警规则）=====
# 以归一化坐标 (x1, y1, x2, y2) 表示，范围 0~1，对应图像左上到右下。
# 仅用于演示"规则化安全监管"：受监管的目标进入该区域即记为闯入违规。
# 可按你的监控画面在 .env 或此处调整，或设 ENABLE_DANGER_ZONE=False 关闭。
ENABLE_DANGER_ZONE = os.getenv("ENABLE_DANGER_ZONE", "true").lower() == "true"
DANGER_ZONE = (0.62, 0.12, 0.98, 0.62)   # 归一化矩形：默认圈定画面右侧为危险作业区
DANGER_ZONE_LABEL = "危险作业区"
# 哪些类别进入危险区记为违规（其余类别即使进入也不预警）
ZONE_WATCH_CLASSES = {"person", "car", "truck", "bus", "motorcycle", "bicycle"}
