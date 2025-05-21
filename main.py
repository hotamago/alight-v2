"""
Open library
"""
from modules.helper import *
from constant import *
from config import cfg

import cv2
import numpy as np
import threading
import time
import pyautogui
from pynput.mouse import Button, Controller
import yaml
import os

import modules.smoothFun as smoothB
from modules.qrcode import QRCodeB
from modules.imageProcess import ImageProcessor
from modules.camera import CameraWebIP, CameraSelf
from modules.projector import Projector
from modules.math import MatrixBincase

import modules.patternMaker as patternMakerB
import modules.calibrateCamera as calibrationB
import modules.screenshot as screenshotB

from sklearn.neighbors import KDTree

"""
Bincase library
"""


"""
Init object
"""
mouse = Controller()

matrixBincase = MatrixBincase()
imageProcesser = ImageProcessor()
# detectQR = cv2.QRCodeDetector()
# qr = QRCodeB(version=cfg['qr_version'], box_size=cfg['qr_box_size'], border=cfg['qr_border'])
stereo = cv2.StereoBM_create(
    numDisparities=cfg['numDisparitiesDepth'], blockSize=cfg['blockSizeDepth'])

pm1 = patternMakerB.PatternMaker(cfg['size_chess'], fullscreensize[0], fullscreensize[1], cfg['corner_chess_size'])
pm1.make_checkerboard_pattern()
imgPattern = pm1.get()
cvt_c = pm1.get_size_chess()
print("Chess size: ", cvt_c)
calibration = calibrationB.Calibration(cvt_c, cfg['num_image_cal'])

### Init camera ###
camera1 = None
if cfg['on_cam1']:
    if cfg['camera1_type'] == 0:
      camera1 = CameraSelf(cfg['camera1_id'], cfg['size_window'],
                          cfg['cam1_exposure'], cfg['cam1_exposure_auto'], cfg['fps_cam1'])
    elif cfg['camera1_type'] == 1:
      camera1 = CameraWebIP(cfg['urlcam1'], cfg['size_window'])

"""
Function main process
"""

def distance(a, b):
    return ((a[0] - b[0])**2 + (a[1] - b[1])**2)**0.5

def process_mouse_click(cfg, mouse, raw_clicked, point, window_size, screen_size, state):
    """
    Xử lý click, double-click, drag & right-click dựa trên trạng thái giữ trong `state`.

    Tham số:
    - cfg: dict cấu hình gồm các khóa:
        * is_flip_mouse: bool
        * double_click_threshold: float
        * time_delay_press: float
        * time_delay_right_click: float
        * circle_in_right_click: float
    - raw_clicked: bool, tín hiệu click hiện tại
    - point: tuple (x, y) tọa độ trong cửa sổ
    - window_size: tuple (win_w, win_h)
    - screen_size: tuple (scr_w, scr_h)
    - state: dict duy trì trạng thái giữa các lần gọi, khởi tạo:
        state.update({
            'last_click_time': 0,
            'click_start_time': None,
            'start_pos': None,
            'max_move_dist': 0,
            'old_clicked': False,
            'old_right_clicked': False
        })
    """
    now = time.time()
    win_w, win_h = window_size
    scr_w, scr_h = screen_size

    # Convert to screen coords
    x, y = point
    if cfg.get('is_flip_mouse', False):
        mx = int(x * scr_w / win_w)
        my = int(y * scr_h / win_h)
    else:
        mx = scr_w - int(x * scr_w / win_w)
        my = scr_h - int(y * scr_h / win_h)
    pos = (mx, my)

    # Track movement during click
    if raw_clicked:
        if state['click_start_time'] is None:
            state['click_start_time'] = now
            state['start_pos'] = pos
        dist = distance(pos, state['start_pos'])
        state['max_move_dist'] = max(state['max_move_dist'], dist)

    # Cập nhật con trỏ
    if pos[0] >= 0 and pos[1] >= 0:
        mouse.position = pos

    # Xử lý sự kiện click
    if raw_clicked and not state['old_clicked']:
        # double click
        if now - state['last_click_time'] < cfg.get('double_click_threshold', 0.3):
            mouse.click(Button.left, 2)
        else:
            mouse.click(Button.left)
        state['last_click_time'] = now
        state['old_clicked'] = True

    # Giữ kéo (drag)
    elif raw_clicked and (now - (state['click_start_time'] or now) > cfg['time_delay_press']):
        mouse.press(Button.left)
        state['old_clicked'] = True

    # Thả chuột và right-click nếu phù hợp
    elif not raw_clicked and state['click_start_time'] is not None:
        hold_time = now - state['click_start_time']
        if hold_time > cfg['time_delay_press']:
            mouse.release(Button.left)
        if (not state['old_right_clicked']
                and hold_time > cfg['time_delay_right_click']
                and state['max_move_dist'] <= cfg['circle_in_right_click']):
            mouse.click(Button.right)
            state['old_right_clicked'] = True
        # Reset state
        state.update({
            'click_start_time': None,
            'start_pos': None,
            'max_move_dist': 0,
            'old_clicked': False,
            'old_right_clicked': False
        })

def main_process():
    size_window = tuple(cfg['size_window'])
    fullscreensize = tuple(cfg['fullscreensize'])

    # --- realtime camera config UI ---
    if cfg['camera1_type'] == 0:
      cv2.namedWindow("Camera Config", cv2.WINDOW_NORMAL)
      cv2.resizeWindow("Camera Config", 600, 150)
      cv2.imshow("Camera Config", np.zeros((150, 600), dtype=np.uint8))
      if cfg['on_cam1']:
          cv2.createTrackbar("Exp1",        "Camera Config", int(
              (cfg['cam1_exposure'] + 10) * 10), 200, lambda v: None)
          cv2.createTrackbar("AutoExp1",    "Camera Config", int(
              cfg['cam1_exposure_auto']), 3, lambda v: None)
          cv2.createTrackbar("Brightness1", "Camera Config", int(
              cfg.get('cam1_brightness', 128)), 255, lambda v: None)
          cv2.createTrackbar("Contrast1",   "Camera Config", int(
              cfg.get('cam1_contrast', 128)),  255, lambda v: None)
          cv2.createTrackbar("Saturation1", "Camera Config", int(
              cfg.get('cam1_saturation', 128)), 255, lambda v: None)
          cv2.createTrackbar("Flip", "Camera Config", cfg['cam1_flip_mode'], 2, lambda v: None)
    elif cfg['camera1_type'] == 1:
        cv2.namedWindow("Camera Config", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Camera Config", 600, 150)
        # Range iso 100-6400
        cv2.createTrackbar("ISO", "Camera Config", int(
            camera1.current_iso - 100), 6400, lambda v: None)
        # Range exposure 100000-10000000 (convert to // 10000)
        cv2.createTrackbar("Exposure", "Camera Config", int(
            camera1.current_exposure_ns - 100000) // 10000, 10000000 // 10000, lambda v: None)
        cv2.createTrackbar("Flip", "Camera Config", cfg['cam1_flip_mode'], 2, lambda v: None)

    if cfg['on_cam1'] and camera1:
        camera1.start_thread()

    maCam1 = ((0, 0), (0, 0), (0, 0), (0, 0))
    maCam1YXZ = (maCam1[0], maCam1[2], maCam1[1], maCam1[3])

    # Case status
    is_detect_corners = False
    is_detect_corners_1 = False

    array_points_paint = []

    # Event mouse
    mousePos = smoothB.average_vecN_smooth(cfg['numAverageMouseMove'])
    is_mouse_time_start = False
    mouse_time_start = 0
    first_mouse_pos = (0, 0)
    last_mouse_pos = (0, 0)

    ### Smooth system ###
    mousePos = smoothB.average_vecN_smooth(cfg['numAverageMouseMove'])
    valueCntNear = [smoothB.average_smooth(cfg['numEleArgvan'])] * cfg['n_points_touch']
    old_clicked = False
    old_right_clicked = False

    ### FPS system ###
    start_time = time.time()
    everyX = 1  # displays the frame rate every 1 second
    counterFrame = 0

    ### Currunt system config ###
    # 0 = hand, 1 = pen
    mode_running = 1

    # State
    state = {}

    # Store the last click time for double click detection
    main_process.last_click_time = 0

    FPP = cfg['FramePerProcess']
    curFPP = 0
    while True:
        counterFrame += 1
        if (time.time() - start_time) > everyX:
            if cfg['show_FPS_console']:
                print("FPS: ", counterFrame / (time.time() - start_time))
            counterFrame = 0
            start_time = time.time()

        # apply realtime camera config settings
        if cfg['on_cam1'] and camera1:
            if cfg['camera1_type'] == 0 and (not is_detect_corners or not calibration.done):
              raw_e1 = cv2.getTrackbarPos("Exp1",        "Camera Config")
              e1 = raw_e1 / 10.0 - 10.0
              ae1 = cv2.getTrackbarPos("AutoExp1",    "Camera Config")
              camera1.setExposure(e1, ae1)
              # pull brightness/contrast/saturation into cfg, then apply
              raw_b1 = cv2.getTrackbarPos("Brightness1", "Camera Config")
              cfg['cam1_brightness'] = raw_b1
              camera1.setBrightness(raw_b1)
              raw_c1 = cv2.getTrackbarPos("Contrast1", "Camera Config")
              cfg['cam1_contrast'] = raw_c1
              camera1.setContrast(raw_c1)
              raw_s1 = cv2.getTrackbarPos("Saturation1", "Camera Config")
              cfg['cam1_saturation'] = raw_s1
              camera1.setSaturation(raw_s1)

            # Auto mode for camera 1 when need to detect corners or calibrate
            if cfg['camera1_type'] == 1 and (not is_detect_corners or not calibration.done):
              camera1.set_auto_mode()

        # Keyboard control
        q = cv2.waitKey(1)
        if q == ord('r'):
            is_detect_corners = False   # reset to re-detect corners
            step_detect_corners = 0
            print("Reset corners detection.")
        if q == ord('q') or (cfg['on_cam1'] and camera1 and camera1.stopped):
            if cfg['on_cam1'] and camera1:
                camera1.stop()
            break

        # ========= Shortcut =========
        
        if q == ord('s'):
          cfg['on_black_points_touch_screen'] = not cfg['on_black_points_touch_screen']
          cv2.destroyWindow("Black points touch screen")
          print("Switch black boand")
          
        if q == ord('c'):
          cfg['on_controller'] = not cfg['on_controller']
          
          if cfg['on_controller'] == True:
            print("On controller")
          else:
            print("False controller")
        
        if q == ord('r'):
          camera1.setExposure(cfg['cam1_exposure'], cfg['cam1_exposure_auto'])
          is_detect_corners_1 = False
          is_detect_corners = False
          calibration.reset()
          print("Start reset")
        if q == ord('p'):
          array_points_paint = []
          print("Reset paint")

        """
        Drop frame
        """
        curFPP += 1
        if curFPP < FPP:
          continue
        else:
          curFPP = 0
        
        """
        Count FPS
        """
        counterFrame+=1
        if (time.time() - start_time) > everyX :
          if cfg['show_FPS_console']:
            print("FPS: ", counterFrame / (time.time() - start_time))
          counterFrame = 0
          start_time = time.time()

        imgCam1 = None
        if cfg['on_cam1'] and camera1:
          imgCam1 = camera1.getFrame()
        imgCam1_org = np.copy(imgCam1)

        if (cfg['on_cam1'] and imgCam1 is None):
          if not is_detect_corners:
              time.sleep(0.1)
          continue

        if cfg['on_debug']:
            if cfg['on_cam1'] and imgCam1 is not None:
                cv2.imshow("Camera test 1", imgCam1)
                cv2.setMouseCallback(
                    "Camera test 1", onMouse, param=(imgCam1, cfg['gamma1']))
                
        imgCam1_onlyc1 = None
        if is_detect_corners_1:
          imgCam1 = matrixBincase.fast_tranform_image_opencv(imgCam1, maCam1YXZ, size_window)
        if calibration.done:
          mtx, dist, newcameramtx, roi = calibration.get()
          imgCam1 = imageProcesser.undistort(imgCam1, mtx, dist, newcameramtx, roi)
        if is_detect_corners_1 and calibration.done:
          imgCam1_onlyc1 = np.copy(imgCam1)
          imgCam1_onlyc1 = cv2.flip(imgCam1_onlyc1, cv2.getTrackbarPos("Flip", "Camera Config")-1)

        if not is_detect_corners or (not calibration.done):
            if not is_detect_corners_1:
              showOneBigQRcorners()
              if cfg['on_cam1'] and imgCam1 is not None:
                  is_detect_corners_1, maCam1, maCam1YXZ = get4CornersSimple(imgCam1, lambda x: (
                      x[0], x[2], x[1], x[3]), delta_point=tuple(cfg['delta_point_qr']))

                  if (is_detect_corners_1 or not cfg['on_cam1']):
                      is_detect_corners = True
                      destroyQRcorners()
                      print("Corners detected.")
                  else:
                      time.sleep(0.05)
            elif not calibration.done:
              setFullScreenCV("imgPattern")
              cv2.imshow("imgPattern", imgPattern)
              calibration.add(imgCam1)
              if calibration.done:
                # ~ screenshotB.startTheard()
                is_detect_corners = True
                # cv2.destroyAllWindows()
                cv2.destroyWindow("imgPattern")
        else:
          if cfg['on_config']:
            if cfg['on_cam1'] and imgCam1 is not None:
              imgCam1Draw = np.copy(imgCam1_org)
              matrixBincase.draw_line(
                  imgCam1Draw, maCam1YXZ[0], maCam1YXZ[1], maCam1YXZ[2], maCam1YXZ[3], 3)
              cv2.imshow("Camera test 1", imgCam1Draw)
              cv2.setMouseCallback(
                  "Camera test 1", onMouse, param=(imgCam1, cfg['gamma1']))

          # Preprocess image
          if mode_running == 1: # Mode lazer pen
            if cfg['camera1_type'] == 0:
              camera1.setProperty(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
              camera1.setProperty(cv2.CAP_PROP_EXPOSURE, -15)
              camera1.setProperty(cv2.CAP_PROP_BRIGHTNESS, 0)
              camera1.setProperty(cv2.CAP_PROP_CONTRAST, 255//2)
              camera1.setProperty(cv2.CAP_PROP_SATURATION, 255//2)
              camera1.setProperty(cv2.CAP_PROP_GAIN, 255//2)
            elif cfg['camera1_type'] == 1:
              camera1.set_manual_mode()
              camera1.set_iso(cv2.getTrackbarPos("ISO", "Camera Config") + 100)
              camera1.set_exposure_ns(cv2.getTrackbarPos("Exposure", "Camera Config") * 10000 + 10000)

            # camera1.setExposure(10, 1)
            
            contoursFigue_cam1 = []
            if cfg['on_cam1']:
                imgCamFTI = np.copy(imgCam1_onlyc1)
                imgFigue = cv2.inRange(imgCamFTI, (0, 0, 60), (255, 255, 255))
                if cfg['on_debug']:
                    cv2.imshow("imgCamFTI", imgCamFTI)
                contoursFigue_cam1, hierarchyFigue = cv2.findContours(imgFigue, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

          # Process image
          if (not cfg['on_cam1']):
              continue

          """
          Process, Caculate point
          """
          list_5_bestest_hull_point = []
          areaValueOr = 0
          areaValueCr = 0
          ratioClicked = 1
          vectorClickLen = 0
          if len(contoursFigue_cam1) > 0:
           if mode_running == 1:
                cnt_5_bestest_hull_point = cfg['n_points_touch']
                for hulls in contoursFigue_cam1:
                    point = np.median(hulls, axis=0)[0]
                    list_5_bestest_hull_point.append(
                        (int(point[0] + cfg['delta_Point'][0]), int(point[1] + cfg['delta_Point'][1])))
                    cnt_5_bestest_hull_point -= 1
                    if cnt_5_bestest_hull_point <= 0:
                        break

          """
          Check clicked points touch
          """
          isClicked = False
          isClickedPoints = [False] * len(list_5_bestest_hull_point)
          if mode_running == 1:
              if len(list_5_bestest_hull_point) > 0:
                  isClicked = True
                  isClickedPoints = [True] * len(list_5_bestest_hull_point)

          """
          Mode: Black points touch screen
          """
          if cfg['on_black_points_touch_screen']:
            # imgFigueDraw = np.copy(imgCamFTI)
            imgFigueDraw = np.zeros((size_window[1], size_window[0], 3))
            imgFigueDraw[:, :] = (0, 0, 0)
            index_contourF = 0
            if cfg['on_paint_test']:
                for point in array_points_paint:
                    cv2.circle(
                        imgFigueDraw, point, cfg['maxRadiusFigueContour'], (255, 255, 255), -1, cv2.LINE_AA)
            for point in list_5_bestest_hull_point:
                if isClickedPoints[index_contourF]:
                    array_points_paint.append(point)
                    cv2.circle(
                        imgFigueDraw, point, cfg['maxRadiusFigueContour'], cfg['color_clicked'], -1, cv2.LINE_AA)
                else:
                    cv2.circle(imgFigueDraw, point, cfg['maxRadiusFigueContour'],
                                cfg['color_nonClicked'], -1, cv2.LINE_AA)
                index_contourF += 1
            imgFigueDraw = cv2.resize(imgFigueDraw, fullscreensize)
            if not cfg['is_debug_clicked']:
                setFullScreenCV("Black points touch screen")
            cv2.imshow("Black points touch screen", imgFigueDraw)

          """
          Process UI, Control mouse or touchscreen
          """
          mousePos.add((0, 0))
          if cfg['on_cam1'] and cfg['on_controller']:
            if len(list_5_bestest_hull_point) > 0:
              # Convert img pos to window pos
              pointMouseNow = list_5_bestest_hull_point[0]
              process_mouse_click(cfg, mouse, isClicked, pointMouseNow, size_window, pyautogui.size(), state)


"""
Run function main
"""
if __name__ == "__main__":
    main_process()
