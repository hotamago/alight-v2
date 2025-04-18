import cv2
import pyautogui
import threading
import numpy as np
import time


class ScreenshotB():
    imgself = None
    size_out = (600, 400)
    stopped = False
    _thread = None

    def __init__(self, size_out=(600, 400)):
        self.size_out = size_out
        self.stopped = False
        self._thread = None
        self.updateFrame()

    def updateFrame(self):
        self.imgself = cv2.resize(cv2.cvtColor(
            np.array(pyautogui.screenshot()), cv2.COLOR_RGB2BGR), self.size_out)

    def getFrame(self):
        return np.copy(self.imgself)

    # Multi threading
    def start_thread(self):
        self.stopped = False
        self._thread = threading.Thread(target=self.update_thread, daemon=True)
        self._thread.start()

    def update_thread(self):
        while not self.stopped:
            self.updateFrame()
            time.sleep(0.01)

    def stop(self):
        self.stopped = True
        if self._thread:
            self._thread.join()
