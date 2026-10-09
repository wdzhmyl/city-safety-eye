"""边缘推理层：YOLOv8n 本地轻量目标推理（CPU 可跑，非重模型），体现端侧实时推理能力。"""
from ultralytics import YOLO

from config import settings

_model = None


def get_model():
    """懒加载 + 单例，避免每次请求重复加载权重。"""
    global _model
    if _model is None:
        _model = YOLO(settings.DETECTION_MODEL)
    return _model


def infer(image_path):
    """对图片执行目标推理（边缘端轻量模型）。

    Args:
        image_path: 图片路径。
    Returns:
        (detections, raw_results)
        detections: [{"label_en", "label_zh", "conf", "box": [x1, y1, x2, y2]}]
    """
    model = get_model()
    results = model(image_path, conf=settings.DETECTION_CONF, verbose=False)[0]
    detections = []
    for box in results.boxes:
        cls_id = int(box.cls[0])
        label_en = model.names[cls_id]
        detections.append({
            "label_en": label_en,
            "label_zh": settings.LABEL_ZH.get(label_en, label_en),
            "conf": float(box.conf[0]),
            "box": [int(v) for v in box.xyxy[0].tolist()],
        })
    return detections, results
