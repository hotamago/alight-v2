import urllib
import urllib.request
import cv2
import numpy as np
import ssl
import threading
import time
import requests

# Camera manual_sensor set: GET http://192.168.50.80:8080/settings/manual_sensor?set=on
# ios: PORT http://192.168.50.80:8080/settings/iso?set=100
# exposure_ns: PORT http://192.168.50.80:8080/settings/exposure_ns?set=100000

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
        self.auto_mode = None
        self.current_iso = 100
        self.current_exposure_ns = 100000
        self.updateFrame()

    def updateFrame(self):
        try:
            resp = urllib.request.urlopen(
                f"{self.url}/shot.jpg", context=self.ctx, timeout=5)
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

    def set_iso(self, val):
        self.current_iso = val
        try:
            response = requests.post(f"{self.url}/settings/iso?set={self.current_iso}")
            if response.status_code == 200:
                print(f"ISO set to {self.current_iso}.")
            else:
                print(f"Failed to set ISO to {self.current_iso}.")
        except requests.RequestException as e:
            print(f"Error setting ISO to {self.current_iso}: {e}")

    def set_exposure_ns(self, val):
        self.current_exposure_ns = val
        try:
            response = requests.post(f"{self.url}/settings/exposure_ns?set={self.current_exposure_ns}")
            if response.status_code == 200:
                print(f"Exposure time set to {self.current_exposure_ns}.")
            else:
                print(f"Failed to set exposure time to {self.current_exposure_ns}.")
        except requests.RequestException as e:
            print(f"Error setting exposure time to {self.current_exposure_ns}: {e}")

    def set_auto_mode(self):
        """
        Set camera to auto mode.
        """
        if self.auto_mode == True:
            return
        self.auto_mode = True
        # using requests send GET request to set auto mode
        try:
            response = requests.get(f"{self.url}/settings/manual_sensor?set=off")
            if response.status_code == 200:
                print("Camera set to auto mode.")
            else:
                print("Failed to set camera to auto mode.")
        except requests.RequestException as e:
            print(f"Error setting camera to auto mode: {e}")

    def set_manual_mode(self):
        """
        Set camera to manual mode.
        """
        if self.auto_mode == False:
            return
        self.auto_mode = False
        # using requests send GET request to set manual mode, send POST ios, exposure_ns
        try:
            response = requests.get(f"{self.url}/settings/manual_sensor?set=on")
            if response.status_code == 200:
                print("Camera set to manual mode.")
                # Set ISO and exposure time
                iso_response = requests.post(f"{self.url}/settings/iso?set={self.current_iso}")
                exposure_response = requests.post(f"{self.url}/settings/exposure_ns?set={self.current_exposure_ns}")
                if iso_response.status_code == 200 and exposure_response.status_code == 200:
                    print("ISO and exposure time set successfully.")
                else:
                    print("Failed to set ISO or exposure time.")
            else:
                print("Failed to set camera to manual mode.")
        except requests.RequestException as e:
            print(f"Error setting camera to manual mode: {e}")

class CameraSelf:
    def __init__(self, id_cam, size_out=(600, 400), exposure_value=80, exposure_auto_value=0, fps_value=30):
        self.size_out = size_out
        self.cap = cv2.VideoCapture(id_cam)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, size_out[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, size_out[1])
        self.cap.set(cv2.CAP_PROP_FPS, fps_value)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.cap.set(cv2.CAP_DSHOW, 1)
        self.imgself = None
        self.success = False
        self.is_flip = False
        self.flip_mode = 0
        self.stopped = True

        self.property_config = {
            cv2.CAP_PROP_AUTO_EXPOSURE: exposure_auto_value,
            cv2.CAP_PROP_EXPOSURE: exposure_value,
            cv2.CAP_PROP_BRIGHTNESS: None,
            cv2.CAP_PROP_CONTRAST: None,
            cv2.CAP_PROP_SATURATION: None,
            cv2.CAP_PROP_HUE: None,
            cv2.CAP_PROP_GAIN: None,
        }
        self.reapply_properties()

        self.updateFrame()

    def reapply_properties(self):
        for prop, value in self.property_config.items():
            if value is not None:
                res = self.cap.set(prop, value)
                if not res:
                    raise ValueError(f"Failed to set property {cap_prop} to {value}.")

    def setProperty(self, cap_prop, value):
        # Check if proerty change is needed
        if self.property_config.get(cap_prop) == value:
            return
        # Check if the property is supported
        if not self.cap.isOpened():
            raise ValueError("Camera is not opened. Cannot set properties.")
        # Current value
        print(f"Current value of {cap_prop}: {self.cap.get(cap_prop)}")
        # Set the property
        self.property_config[cap_prop] = value
        # self.reapply_properties()
        res = self.cap.set(cap_prop, value)
        if not res:
            raise ValueError(f"Failed to set property {cap_prop} to {value}.")

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
        # self.property_config[cv2.CAP_PROP_AUTO_EXPOSURE] = 0.75 if enable else 0.25
        pass

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
