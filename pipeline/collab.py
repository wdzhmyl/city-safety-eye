"""编排层：把边缘推理层、云端理解层、工具层串联为一次完整的"端云协同推理"。

交互层只需调用 analyze_image()，无需关心内部各层如何协作。
后续扩展视频按帧推理、批处理、缓存等，也在此层统一编排。
"""
from core.edge_infer import infer
from core.cloud_reason import generate_report
from utils.visualize import draw_detections
from utils.imgio import new_tempfile, numpy_to_tempfile


def analyze_image(image):
    """执行一次完整的端云协同图片推理。

    Args:
        image: numpy 数组（网页上传）或图片路径。
    Returns:
        (annotated_path, report, detections)
        annotated_path: 带推理框的标注图路径
        report: Markdown 推理分析报告
        detections: 推理结果明细（供前端做"推理概览"）
    """
    in_path = image if isinstance(image, str) else numpy_to_tempfile(image)

    detections, _ = infer(in_path)

    out_path = new_tempfile(suffix=".jpg")
    draw_detections(in_path, detections, out_path)

    report = generate_report(detections, in_path)
    return out_path, report, detections
