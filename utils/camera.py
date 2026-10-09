"""解耦的摄像头管理：本地摄像头自动发现 + 网络/RTSP/手机视频流打开与读取。

与 UI 完全解耦——本模块只负责"拿到一帧画面"，具体选哪个摄像头由调用方（交互层）决定。
支持两类视频源：
- 本地摄像头：整数设备索引（OpenCV 自动探测可用索引），即电脑摄像头。
- 网络/手机摄像头：RTSP 地址或 http 视频流。手机安装 IP Webcam / DroidCam 后，
  手机会变成一个网络视频流，把地址粘贴进来即可像普通摄像头一样被选择使用。
"""
import cv2


class CameraManager:
    def __init__(self):
        self.cap = None
        self.source = None

    def discover(self, max_index=6):
        """探测本地可用的摄像头索引（0..max_index 中能成功打开的）。"""
        found = []
        for i in range(max_index):
            try:
                c = cv2.VideoCapture(i)
                if c.isOpened():
                    found.append(i)
                    c.release()
            except Exception:
                pass
        return found

    def open(self, source):
        """打开摄像头/视频流。source 可为 int（本地索引）或 str（RTSP/URL）。"""
        if self.cap is not None:
            self.cap.release()
        try:
            self.cap = cv2.VideoCapture(source)
        except Exception:
            self.cap = None
        self.source = source
        return self.cap is not None and self.cap.isOpened()

    def read(self):
        """读取一帧（BGR numpy 数组）；无画面或设备未打开时返回 None。"""
        if self.cap is None or not self.cap.isOpened():
            return None
        ret, frame = self.cap.read()
        return frame if ret else None

    def release(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
