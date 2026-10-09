"""编排层：把边缘推理、危险区规则、云端理解、可视化串联为一次完整"端云协同工地巡检"。

交互层只需调用 analyze_image()，无需关心内部各层如何协作。
后续扩展视频按帧推理、批处理、缓存等，也在此层统一编排。
"""
from PIL import Image

from core.edge_infer import infer, check_danger_zone
from core.cloud_reason import generate_report
from utils.visualize import draw_detections
from utils.imgio import new_tempfile, numpy_to_tempfile


def analyze_image(image):
    """执行一次完整的端云协同工地巡检。

    Args:
        image: numpy 数组（网页上传）或图片路径。
    Returns:
        (annotated_path, report, detections, intrusions)
        annotated_path: 带检测框与危险区的标注图路径
        report: Markdown 施工安全巡检报告
        detections: 推理结果明细（供前端做"巡检概览"）
        intrusions: 危险作业区闯入目标列表
    """
    in_path = image if isinstance(image, str) else numpy_to_tempfile(image)

    detections, _ = infer(in_path)

    img = Image.open(in_path).convert("RGB")
    intrusions, _zone = check_danger_zone(detections, img.size)

    out_path = new_tempfile(suffix=".jpg")
    draw_detections(in_path, detections, out_path)

    report = generate_report(detections, in_path, intrusions=intrusions)
    return out_path, report, detections, intrusions
