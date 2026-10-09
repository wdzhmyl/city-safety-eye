"""边缘推理层：YOLOv8n 本地轻量目标推理（CPU 可跑），并叠加工地安全规则（危险区闯入判定）。"""
from ultralytics import YOLO

import numpy as np

from config import settings

_model = None


def get_model():
    """懒加载 + 单例，避免每次请求重复加载权重。"""
    global _model
    if _model is None:
        _model = YOLO(settings.DETECTION_MODEL)
    return _model


def infer(image):
    """对图片执行目标推理（边缘端轻量模型）。

    Args:
        image: 图片路径(str) 或 numpy 数组(BGR)。实时摄像头场景直接传帧，
               避免每帧反复落盘，降低端侧延迟。
    Returns:
        (detections, raw_results)
        detections: [{"label_en", "label_zh", "conf", "box": [x1, y1, x2, y2]}]
    """
    model = get_model()
    src = image if isinstance(image, np.ndarray) else image
    results = model(src, conf=settings.DETECTION_CONF, verbose=False)[0]
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


def check_danger_zone(detections, image_size):
    """根据配置的危险作业区，判定哪些目标闯入（工地安全规则）。

    Args:
        detections: infer() 返回的推理结果。
        image_size: (width, height)，用于将归一化区域换算成像素坐标。
    Returns:
        (intrusions, zone_px)
        intrusions: 闯入危险区、且属于监管类别的目标列表（detections 会带 in_zone 标记）
        zone_px: 危险区像素坐标 (x1, y1, x2, y2)，未启用时返回 None
    """
    for d in detections:
        d.setdefault("in_zone", False)
    if not settings.ENABLE_DANGER_ZONE:
        return [], None

    w, h = image_size
    x1, y1, x2, y2 = settings.DANGER_ZONE
    zone_px = (int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h))

    intrusions = []
    for d in detections:
        bx1, by1, bx2, by2 = d["box"]
        cx, cy = (bx1 + bx2) / 2, (by1 + by2) / 2
        inside = zone_px[0] <= cx <= zone_px[2] and zone_px[1] <= cy <= zone_px[3]
        watched = d["label_en"] in settings.ZONE_WATCH_CLASSES
        d["in_zone"] = bool(inside and watched)
        if d["in_zone"]:
            intrusions.append(d)
    return intrusions, zone_px
