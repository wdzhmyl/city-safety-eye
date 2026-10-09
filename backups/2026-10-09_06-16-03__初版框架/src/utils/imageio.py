"""工具层：图片读写与临时文件管理。"""
import tempfile

from PIL import Image


def numpy_to_tempfile(arr, suffix=".jpg"):
    """把 numpy 数组（Gradio 上传的图）落盘为临时图片文件。"""
    path = tempfile.mktemp(suffix=suffix)
    Image.fromarray(arr).save(path)
    return path


def new_tempfile(suffix=".jpg"):
    """生成一个新的临时文件路径。"""
    return tempfile.mktemp(suffix=suffix)
