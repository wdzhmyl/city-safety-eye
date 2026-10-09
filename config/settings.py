"""配置层：集中管理模型、API 与检测参数。

所有可调项集中在此，其余各层只依赖本模块，便于统一修改与环境隔离。
可通过同目录 .env 文件或系统环境变量覆盖（推荐环境变量，避免泄露密钥）。
"""
import os
from dotenv import load_dotenv

load_dotenv()

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

# ===== 中文标签映射：聚焦城市治理相关类别（COCO 子集）=====
LABEL_ZH = {
    "person": "行人",
    "bicycle": "自行车/非机动车",
    "motorcycle": "摩托车",
    "car": "小汽车",
    "bus": "公交车",
    "truck": "卡车",
    "traffic light": "交通信号灯",
    "fire hydrant": "消防栓",
    "stop sign": "停止标志",
    "parking meter": "停车计费器",
    "bench": "公共座椅",
}
