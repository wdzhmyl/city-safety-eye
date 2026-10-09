"""冒烟测试：验证各层可导入、兜底报告可正常生成。

不依赖云端 Key，可直接运行：python -m pytest tests/ -v
"""
from core import template
from utils import draw
from config import settings


def test_settings_loaded():
    """配置层应提供检测模型与置信度阈值。"""
    assert settings.DETECTION_MODEL
    assert 0.0 < settings.DETECTION_CONF <= 1.0


def test_template_report_with_detections():
    """兜底层应能对检测结果生成结构化报告。"""
    detections = [
        {"label_en": "car", "label_zh": "小汽车", "conf": 0.9, "box": [0, 0, 10, 10]},
        {"label_en": "person", "label_zh": "行人", "conf": 0.8, "box": [0, 0, 5, 5]},
        {"label_en": "car", "label_zh": "小汽车", "conf": 0.7, "box": [0, 0, 20, 20]},
    ]
    text = template.build_template_report(detections)
    assert "城市安全事件分析报告" in text
    assert "小汽车：2 个" in text
    assert "行人：1 个" in text


def test_template_report_empty():
    """无检测结果时不应报错。"""
    text = template.build_template_report([])
    assert "未检测到目标" in text


def test_draw_detections(tmp_path):
    """工具层应能生成标注图文件。"""
    from PIL import Image

    src = tmp_path / "src.jpg"
    Image.new("RGB", (100, 100), "white").save(src)
    out = tmp_path / "out.jpg"
    detections = [{"label_en": "car", "label_zh": "小汽车", "conf": 0.9, "box": [1, 2, 30, 40]}]
    result = draw.draw_detections(str(src), detections, str(out))
    assert result == str(out)
    assert out.exists()
