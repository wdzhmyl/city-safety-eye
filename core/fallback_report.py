"""离线兜底层：无云端 Key 或调用失败时使用的本地模板报告，保证推理服务高可用。

体现端云协同推理的可靠性：云侧不可用时自动降级到本地，demo 永不崩。
"""
from collections import Counter

from config import settings


def build_fallback_report(detections, intrusions=None, note=""):
    """基于推理结果生成离线兜底报告（施工安全巡检语境）。

    Args:
        detections: 推理结果列表（来自 core.edge_infer.infer）。
        intrusions: 危险作业区闯入目标列表（可选）。
        note: 附加说明（如失败原因），追加在报告末尾。
    Returns:
        Markdown 格式的报告文本。
    """
    cnt = Counter(d["label_zh"] for d in detections)
    items = "\n".join(f"- {k}：{v} 个" for k, v in cnt.items()) or "- 未检测到目标"
    zone_line = ""
    if settings.ENABLE_DANGER_ZONE:
        n = len(intrusions) if intrusions else 0
        if n:
            names = "、".join(d["label_zh"] for d in intrusions)
            zone_line = (f"\n- ⚠️ **危险作业区闯入 {n} 起**，请立即核查现场并驱离/整改"
                         f"（涉及：{names}）。")
        else:
            zone_line = "\n- 危险作业区内暂未检测到违规闯入。"
    return f"""# 施工安全巡检报告（本地模板）
## 一、现场概况
{items}
## 二、安全隐患研判
- 基于目标分布与危险作业区规则的初步研判；接入云端模型后可获得精细分析。{zone_line}
## 三、整改建议
- 请安全员依据现场派单处置，重点核查危险作业区闯入情况。
## 四、处置结论
- 当前为本地离线兜底报告，配置云端 API 后可自动生成专业施工安全巡检分析。{note}"""
