"""工具层：在图像上绘制检测框与中文标签。"""
from PIL import Image, ImageDraw


def draw_detections(image_path, detections, out_path):
    """在图片上绘制检测框与标签，输出到 out_path。

    Args:
        image_path: 原图路径。
        detections: 检测结果列表（需含 "box" 与 "label_zh"、"conf"）。
        out_path: 标注图输出路径。
    Returns:
        out_path
    """
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    for d in detections:
        x1, y1, x2, y2 = d["box"]
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        draw.text((x1, max(0, y1 - 12)), f'{d["label_zh"]} {d["conf"]:.2f}', fill="red")
    img.save(out_path)
    return out_path
