import urllib
import urllib.request
import cv2
import numpy as np
import ssl
import threading
import time


class CameraWebIP:
    def __init__(self, url, size_out=(600, 400)):
        self.size_out = size_out
        self.url = url
        self.ctx = ssl.create_default_context()
        self.ctx.check_hostname = False
        self.ctx.verify_mode = ssl.CERT_NONE
        self.imgself = None
        self.success = False
        self.stopped = True
        self.updateFrame()

    def updateFrame(self):
        try:
            resp = urllib.request.urlopen(
                self.url, context=self.ctx, timeout=5)
            arr = np.frombuffer(resp.read(), np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
            self.imgself = cv2.resize(img, self.size_out)
            self.success = True
        except Exception:
            self.success = False

    def getFrame(self):
        return np.copy(self.imgself) if self.imgself is not None else None

    def start_thread(self):
        self.stopped = False
        threading.Thread(target=self.update_thread, daemon=True).start()

    def update_thread(self):
        while not self.stopped:
            self.updateFrame()
            time.sleep(0.01)

    def stop(self):
        self.stopped = True


class CameraSelf:
    def __init__(self, id_cam, size_out=(600, 400), exposure_value=80, exposure_auto_value=0, fps_value=30):
        self.size_out = size_out
        self.cap = cv2.VideoCapture(id_cam)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, size_out[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, size_out[1])
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, exposure_auto_value)
        self.cap.set(cv2.CAP_PROP_EXPOSURE, exposure_value)
        self.cap.set(cv2.CAP_PROP_FPS, fps_value)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.cap.set(cv2.CAP_DSHOW, 1)
        self.imgself = None
        self.success = False
        self.is_flip = False
        self.flip_mode = 0
        self.stopped = True
        self.updateFrame()

    def setProperty(self, cap_prop, value):
        self.cap.set(cap_prop, value)

    def setExposure(self, exposure_value, exposure_auto_value=0):
        """
        Set exposure mode and value.
        Note: Many backends (e.g. v4l2 on Linux) expect specific AUTO_EXPOSURE flags 
        (1=auto, 3=manual) or floating ranges (0.25/0.75 in OpenCV), so raw CAP_PROP_EXPOSURE 
        may be ignored unless correct backend flags are used.
        """
        self.setProperty(cv2.CAP_PROP_AUTO_EXPOSURE, exposure_auto_value)
        self.setProperty(cv2.CAP_PROP_EXPOSURE, exposure_value)

    def setAutoExposure(self, enable=True):
        """
        Toggle auto exposure.
        OpenCV uses 0.25 for manual, 0.75 for auto on some backends—adjust as needed.
        """
        mode = 0.75 if enable else 0.25
        self.setProperty(cv2.CAP_PROP_AUTO_EXPOSURE, mode)

    def setBrightness(self, value):
        self.setProperty(cv2.CAP_PROP_BRIGHTNESS, value)

    def setContrast(self, value):
        self.setProperty(cv2.CAP_PROP_CONTRAST, value)

    def setSaturation(self, value):
        self.setProperty(cv2.CAP_PROP_SATURATION, value)

    def setHue(self, value):
        self.setProperty(cv2.CAP_PROP_HUE, value)

    def setGain(self, value):
        self.setProperty(cv2.CAP_PROP_GAIN, value)

    def setWhiteBalance(self, red, blue):
        """
        Some backends support direct white‑balance channels.
        """
        self.setProperty(cv2.CAP_PROP_WHITE_BALANCE_RED_V, red)
        self.setProperty(cv2.CAP_PROP_WHITE_BALANCE_BLUE_U, blue)

    def updateFrame(self):
        self.success, img = self.cap.read()
        if self.success:
            img = cv2.resize(img, self.size_out)
            if self.is_flip:
                img = cv2.flip(img, self.flip_mode)
            self.imgself = img
        else:
            self.success = False

    def getFrame(self):
        return np.copy(self.imgself) if self.imgself is not None else None

    def start_thread(self):
        self.stopped = False
        threading.Thread(target=self.update_thread, daemon=True).start()

    def update_thread(self):
        while not self.stopped:
            self.updateFrame()
            time.sleep(0.01)

    def stop(self):
        self.stopped = True
        if self.cap:
            self.cap.release()
