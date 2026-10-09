"""离线兜底层：无云端 Key 或调用失败时使用的本地模板报告，保证推理服务高可用。

体现端云协同推理的可靠性：云侧不可用时自动降级到本地，demo 永不崩。
"""
from collections import Counter


def build_fallback_report(detections, note=""):
    """基于推理结果生成离线兜底报告。

    Args:
        detections: 推理结果列表（来自 core.edge_infer.infer）。
        note: 附加说明（如失败原因），追加在报告末尾。
    Returns:
        Markdown 格式的报告文本。
    """
    cnt = Counter(d["label_zh"] for d in detections)
    items = "\n".join(f"- {k}：{v} 个" for k, v in cnt.items()) or "- 未检测到目标"
    return f"""# 推理分析报告（本地模板）
## 一、推理对象清单
{items}
## 二、推理研判
- 基于目标数量的初步研判；接入云端模型后可获得精细分析。
## 三、处置建议
- 请依据实际场景派单处置。
## 四、总体结论
- 当前为本地离线兜底报告，配置云端 API 后可自动生成专业推理分析。{note}"""
