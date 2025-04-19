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
    camera1 = CameraSelf(cfg['camera1_id'], cfg['size_window'],
                         cfg['cam1_exposure'], cfg['cam1_exposure_auto'], cfg['fps_cam1'])

"""
Function main process
"""


def main_process():
    size_window = tuple(cfg['size_window'])
    fullscreensize = tuple(cfg['fullscreensize'])

    # --- realtime camera config UI ---
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

    if cfg['on_cam1'] and camera1:
        camera1.start_thread()

    maCam1 = ((0, 0), (0, 0), (0, 0), (0, 0))
    maCam1YXZ = (maCam1[0], maCam1[2], maCam1[1], maCam1[3])

    # Case status
    is_detect_corners = False
    is_detect_corners_1 = False
    is_detect_corners_2 = False

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
            if mode_running == 0 or not is_detect_corners or not calibration.done:
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
        # if q == ord('1'):
        #   mode_running = 0
        #   print("switch to hand mode")
        # if q == ord('2'):
        #   mode_running = 1
        #   print("switch to lazer mode")
        
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
          is_detect_corners_2 = False
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
        if is_detect_corners_2:
          imgCam1 = matrixBincase.fast_tranform_image_opencv(imgCam1, maCam2YXZ, size_window)

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
              imgCam1Draw = np.copy(imgCam1)
              matrixBincase.draw_line(
                  imgCam1Draw, maCam1YXZ[0], maCam1YXZ[1], maCam1YXZ[2], maCam1YXZ[3], 3)
              cv2.imshow("Camera test 1", imgCam1Draw)
              cv2.setMouseCallback(
                  "Camera test 1", onMouse, param=(imgCam1, cfg['gamma1']))

          # Preprocess image
          if mode_running == 0: # Mode figue
            """
            Camera 1: camera
            """
            contoursFigue_cam1 = []
            if cfg['on_cam1']:
              contoursFigue_cam1 = auto_ProcessImage_nofti(imgCam1_onlyc1, cfg['gamma1'], cfg['fillCam1_01'], cfg['noseCam1'], cfg['on_show_cam1'], cfg['on_cam1Hsv'], cfg['on_cam1Ycbcr'], cfg['on_cam1FTI'], "Camera test 1")

          elif mode_running == 1: # Mode lazer pen
            camera1.setProperty(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
            camera1.setProperty(cv2.CAP_PROP_EXPOSURE, -15)
            camera1.setProperty(cv2.CAP_PROP_BRIGHTNESS, 0)
            camera1.setProperty(cv2.CAP_PROP_CONTRAST, 255//2)
            camera1.setProperty(cv2.CAP_PROP_SATURATION, 255//2)
            camera1.setProperty(cv2.CAP_PROP_GAIN, 255//2)

            # camera1.setExposure(10, 1)
            imgCam1 = camera1.getFrame()
            
            contoursFigue_cam1 = []
            if cfg['on_cam1']:
                imgCamFTI = np.copy(imgCam1)
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

            if mode_running == 0:
              """
              Caculate point click
              """
              list_highest_point_hull = []
              for hulls in contoursFigue_cam1:
                highest_point_hull = max(hulls, key=lambda x: x[0][1])
                list_highest_point_hull.append(highest_point_hull[0])
              list_highest_point_hull.sort(key=lambda x: -x[1])
              cnt_5_bestest_hull_point = cfg['n_points_touch']
              for point in list_highest_point_hull:
                list_5_bestest_hull_point.append(point + cfg['delta_Point'])
                cnt_5_bestest_hull_point -= 1
                if cnt_5_bestest_hull_point <= 0:
                    break
              for i in range(0, len(list_5_bestest_hull_point)):
                list_5_bestest_hull_point[i] = matrixBincase.tramform_points(
                  list_5_bestest_hull_point[i], maCam2YXZ, size_window)

              """
              Caculate info of convexHull
              """
              if len(contoursFigue_cam1) > 0:
                  np_contours = np.vstack(
                      contoursFigue_cam1).reshape((-1, 1, 2))
                  chull = cv2.convexHull(np_contours)

                  imgDrawC = np.zeros(
                      (size_window[1], size_window[0], 3))
                  imgDrawC = cv2.drawContours(
                      imgDrawC, chull, -1, (0, 255, 0), 3)

                  areaValueOr = cv2.contourArea(chull)

                  Moo = cv2.moments(chull)
                  cx = int(Moo['m10']/Moo['m00'])
                  cy = int(Moo['m01']/Moo['m00'])
                  vectorClickLen = distanceB2Points(
                      list_5_bestest_hull_point[0], [cx, cy])

                  (x, y), radius = cv2.minEnclosingCircle(chull)
                  center = (int(x), int(y))
                  radius = int(radius)
                  areaValueCr = radius*radius*np.pi

                  ratioClicked = (areaValueOr + 0.001) / \
                      (areaValueCr + 0.001)

                  cv2.circle(imgDrawC, center, radius,
                              cfg['color_clicked'], 1, cv2.LINE_AA)
                  cv2.imshow("imgDrawC", imgDrawC)

            elif mode_running == 1:
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
          if mode_running == 0:
              # ~ print(ratioClicked, " : ", areaValueOr , "/", areaValueCr)
              # ~ if vectorClickLen >= 50.0:
              if ratioClicked <= 0.45:
                  isClicked = True
                  isClickedPoints = [True] * len(list_5_bestest_hull_point)
          elif mode_running == 1:
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
              width, height = pyautogui.size()
              pointMouseNow = list_5_bestest_hull_point[0]
              if cfg['is_flip_mouse']:
                mouseComputer = (int(
                    pointMouseNow[0]*width/size_window[0]), int(pointMouseNow[1]*height/size_window[1]))
              else:
                mouseComputer = (int(width - pointMouseNow[0]*width/size_window[0]), int(
                  height - pointMouseNow[1]*height/size_window[1]))

              # Time
              if isClicked:
                if not is_mouse_time_start:
                  mouse_time_start = time.time()
                  is_mouse_time_start = True
                  first_mouse_pos = mouseComputer
                  last_mouse_pos = mouseComputer
                if distanceB2Points(mouseComputer, first_mouse_pos) > distanceB2Points(last_mouse_pos, first_mouse_pos):
                  last_mouse_pos = mouseComputer

              if mouseComputer >= (0, 0):
                mousePos.addPrev(mouseComputer)
                mouse.position = mouseComputer
                # ~ mouse.move(mouseComputer[0], mouseComputer[1])
              if isClicked and (not old_clicked):
                # Press and release
                mouse.click(Button.left)
                old_clicked = True

                # Double click
                # mouse.click(Button.left, 2)

                # Scroll two steps down
                # mouse.scroll(0, 2)
              elif isClicked and ((time.time() - mouse_time_start) > cfg['time_delay_press']):
                mouse.press(Button.left)
                old_clicked = True
              elif not isClicked:
                if is_mouse_time_start:
                  if ((time.time() - mouse_time_start) > cfg['time_delay_press']):
                    mouse.release(Button.left)

                  if (not old_right_clicked):
                    # ~ print((time.time() - mouse_time_start), distanceB2Points(last_mouse_pos, first_mouse_pos))
                    if ((time.time() - mouse_time_start) > cfg['time_delay_right_click']) and (distanceB2Points(last_mouse_pos, first_mouse_pos) <= cfg['circle_in_right_click']):
                        mouse.click(Button.right)
                        old_right_clicked = True

                    # mouse.position = (0, 0)

                old_clicked = False
                old_right_clicked = False
                is_mouse_time_start = False

              else:
                if is_mouse_time_start:
                  if ((time.time() - mouse_time_start) > cfg['time_delay_press']):
                    mouse.release(Button.left)

                  if (not old_right_clicked):
                    # ~ print((time.time() - mouse_time_start), distanceB2Points(last_mouse_pos, first_mouse_pos))
                    if ((time.time() - mouse_time_start) > cfg['time_delay_right_click']) and (distanceB2Points(last_mouse_pos, first_mouse_pos) <= cfg['circle_in_right_click']):
                      mouse.click(Button.right)
                      old_right_clicked = True

                  # mouse.position = (0, 0)

                old_clicked = False
                old_right_clicked = False
                is_mouse_time_start = False


"""
Run function main
"""
if __name__ == "__main__":
    main_process()
