"""理解层：调用云端多模态大模型生成《城市安全事件分析报告》。

无 API Key 或调用失败时，回退到本地模板报告（core.template），保证 demo 始终可运行。
"""
import base64

from openai import OpenAI

from config import settings
from core import template

SYSTEM_PROMPT = """你是一名智慧城市安全分析师。基于图像目标检测结果（类别/置信度）以及原始图像，
输出一份结构化的《城市安全事件分析报告》，严格使用以下 Markdown 结构：
# 城市安全事件分析报告
## 一、检测事件清单
- 用列表列出主要目标及数量
## 二、风险研判
- 评估交通、人群聚集、设施占用等风险等级（低/中/高）
## 三、处置建议
- 给出具体、可操作的处置建议
## 四、总体结论
- 一句话总结当前区域安全态势
若图像中存在明显烟火、积水、违停等异常，请务必指出。语言专业、简洁、中文。"""


def _encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def generate_report(detections, image_path):
    """生成安全事件分析报告。

    优先调用云端多模态大模型；无 Key 或调用异常时自动降级为本地模板报告。

    Args:
        detections: 检测结果列表。
        image_path: 原图路径（用于多模态输入）。
    Returns:
        Markdown 报告文本。
    """
    if not settings.LLM_API_KEY:
        return template.build_template_report(detections)

    client = OpenAI(base_url=settings.LLM_BASE_URL, api_key=settings.LLM_API_KEY)
    summary = "; ".join(f'{d["label_zh"]}({d["conf"]:.2f})' for d in detections) or "未检测到目标"
    try:
        resp = client.chat.completions.create(
            model=settings.LLM_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": [
                    {"type": "text", "text": f"目标检测结果：{summary}。请结合图像生成报告。"},
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/jpeg;base64,{_encode_image(image_path)}"}},
                ]},
            ],
            max_tokens=settings.LLM_MAX_TOKENS,
        )
        return resp.choices[0].message.content
    except Exception as e:
        note = f"\n\n（云端模型调用失败，已回退本地模板：{e}）"
        return template.build_template_report(detections, note=note)
