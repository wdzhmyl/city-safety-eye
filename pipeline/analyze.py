"""编排层：把检测层、理解层、工具层串联为一次完整的"安全事件分析"。

交互层只需调用 analyze_image()，无需关心内部三层如何协作。
后续扩展视频按帧分析、批处理、缓存等，也在此层统一编排。
"""
from core.detector import detect
from core.reporter import generate_report
from utils.draw import draw_detections
from utils.imageio import new_tempfile, numpy_to_tempfile


def analyze_image(image):
    """执行一次完整的图片安全分析。

    Args:
        image: numpy 数组（网页上传）或图片路径。
    Returns:
        (annotated_path, report, detections)
        annotated_path: 带检测框的标注图路径
        report: Markdown 安全事件分析报告
        detections: 检测结果明细（供前端做"检测概览"）
    """
    in_path = image if isinstance(image, str) else numpy_to_tempfile(image)

    detections, _ = detect(in_path)

    out_path = new_tempfile(suffix=".jpg")
    draw_detections(in_path, detections, out_path)

    report = generate_report(detections, in_path)
    return out_path, report, detections
