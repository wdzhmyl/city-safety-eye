"""冒烟测试：验证各层可导入、离线兜底报告可正常生成（智慧工地安全巡检场景）。

不依赖云端 Key，可直接运行：python -m pytest tests/ -v
"""
from core import fallback_report
from utils import visualize
from config import settings


def test_settings_loaded():
    """配置层应提供场景、推理模型与置信度阈值。"""
    assert settings.SCENARIO
    assert settings.DETECTION_MODEL
    assert 0.0 < settings.DETECTION_CONF <= 1.0


def test_fallback_report_with_detections():
    """离线兜底层应能对推理结果生成结构化施工安全巡检报告。"""
    detections = [
        {"label_en": "truck", "label_zh": "工程运输车", "conf": 0.9, "box": [0, 0, 10, 10], "in_zone": True},
        {"label_en": "person", "label_zh": "作业人员", "conf": 0.8, "box": [0, 0, 5, 5], "in_zone": False},
        {"label_en": "truck", "label_zh": "工程运输车", "conf": 0.7, "box": [0, 0, 20, 20], "in_zone": False},
    ]
    intrusions = [d for d in detections if d.get("in_zone")]
    text = fallback_report.build_fallback_report(detections, intrusions=intrusions)
    assert "施工安全巡检报告" in text
    assert "工程运输车：2 个" in text
    assert "作业人员：1 个" in text
    assert "危险作业区闯入 1 起" in text


def test_fallback_report_empty():
    """无推理结果时不应报错。"""
    text = fallback_report.build_fallback_report([])
    assert "未检测到目标" in text


def test_draw_detections(tmp_path):
    """工具层应能生成标注图文件。"""
    from PIL import Image

    src = tmp_path / "src.jpg"
    Image.new("RGB", (100, 100), "white").save(src)
    out = tmp_path / "out.jpg"
    detections = [{"label_en": "truck", "label_zh": "工程运输车", "conf": 0.9,
                   "box": [1, 2, 30, 40], "in_zone": False}]
    result = visualize.draw_detections(str(src), detections, str(out))
    assert result == str(out)
    assert out.exists()
