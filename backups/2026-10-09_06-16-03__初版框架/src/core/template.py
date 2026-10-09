"""兜底层：无云端 Key 或调用失败时使用的本地模板报告。

保证 demo 在任何环境下都可运行、都有输出。
"""
from collections import Counter


def build_template_report(detections, note=""):
    """基于检测结果生成离线模板报告。

    Args:
        detections: 检测结果列表（来自 core.detector.detect）。
        note: 附加说明（如失败原因），追加在报告末尾。
    Returns:
        Markdown 格式的报告文本。
    """
    cnt = Counter(d["label_zh"] for d in detections)
    items = "\n".join(f"- {k}：{v} 个" for k, v in cnt.items()) or "- 未检测到目标"
    return f"""# 城市安全事件分析报告（本地模板）
## 一、检测事件清单
{items}
## 二、风险研判
- 基于目标数量的初步研判；接入云端模型后可获得精细分析。
## 三、处置建议
- 请依据实际场景派单处置。
## 四、总体结论
- 当前为本地离线模板报告，配置云端 API 后可自动生成专业分析。{note}"""
