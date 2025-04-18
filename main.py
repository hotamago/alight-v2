"""
Open library
"""
import cv2
import numpy as np
import threading
import time
import pyautogui
from pynput.mouse import Button, Controller
import yaml
import os

from sklearn.neighbors import KDTree

"""
Bincase library
"""
from modules.math import MatrixBincase
from modules.projector import Projector
from modules.camera import CameraWebIP, CameraSelf
from modules.imageProcess import ImageProcessor
from modules.qrcode import QRCodeB
from modules.helper import (onMouse, showQRcorners, destroyQRcorners, get4Corners, 
                            auto_ProcessImage, auto_ProcessImage_onlyfti, 
                            auto_ProcessImage_onlyhand, setFullScreenCV)

from config import cfg
from constant import *

import modules.smoothFun as smoothB

"""
Init object
"""
mouse = Controller()

matrixBincase = MatrixBincase()
imageProcesser = ImageProcessor()
detectQR = cv2.QRCodeDetector()
qr = QRCodeB(version=cfg['qr_version'], box_size=cfg['qr_box_size'], border=cfg['qr_border'])
stereo = cv2.StereoBM_create(numDisparities=cfg['numDisparitiesDepth'], blockSize=cfg['blockSizeDepth'])

### Init camera ###
camera1 = None
camera2 = None
if cfg['on_cam1']:
  camera1 = CameraSelf(cfg['camera1_id'], cfg['size_window'], cfg['cam1_exposure'], cfg['cam1_exposure_auto'], cfg['fps_cam1'])
if cfg['on_cam2']:
  camera2 = CameraSelf(cfg['camera2_id'], cfg['size_window'], cfg['cam2_exposure'], cfg['cam2_exposure_auto'], cfg['fps_cam2'])

"""
Configuration UI Setup
"""
config_window_name = "Configuration"
cv2.namedWindow(config_window_name)

# --- Trackbar Callback Functions ---
def update_gamma1(val):
    cfg['gamma1'] = val / 100.0

def update_gamma2(val):
    cfg['gamma2'] = val / 100.0

def update_noseCam1_0(val):
    if val % 2 == 0: val += 1
    if val < 3: val = 3
    cfg['noseCam1'][0] = val

def update_noseCam1_1(val):
    if val % 2 == 0: val += 1
    if val < 3: val = 3
    cfg['noseCam1'][1] = val

def update_noseCam2_0(val):
    if val % 2 == 0: val += 1
    if val < 3: val = 3
    cfg['noseCam2'][0] = val

def update_noseCam2_1(val):
    if val % 2 == 0: val += 1
    if val < 3: val = 3
    cfg['noseCam2'][1] = val

def update_deltaContoursClicked(val):
    cfg['deltaContoursClicked'] = val

def update_maxRadiusFigueWithFigueShallow(val):
    cfg['maxRadiusFigueWithFigueShallow'] = val

def update_delta_Point_X(val):
    cfg['delta_Point'][0] = val - 50

def update_delta_Point_Y(val):
    cfg['delta_Point'][1] = val - 50

# --- Create Trackbars ---
cv2.createTrackbar('Gamma 1 (x100)', config_window_name, int(cfg['gamma1'] * 100), 255, update_gamma1)
cv2.createTrackbar('Gamma 2 (x100)', config_window_name, int(cfg['gamma2'] * 100), 255, update_gamma2)
cv2.createTrackbar('Noise Cam1 K1', config_window_name, cfg['noseCam1'][0], 21, update_noseCam1_0)
cv2.createTrackbar('Noise Cam1 K2', config_window_name, cfg['noseCam1'][1], 21, update_noseCam1_1)
cv2.createTrackbar('Noise Cam2 K1', config_window_name, cfg['noseCam2'][0], 21, update_noseCam2_0)
cv2.createTrackbar('Noise Cam2 K2', config_window_name, cfg['noseCam2'][1], 21, update_noseCam2_1)
cv2.createTrackbar('Click Threshold', config_window_name, cfg['deltaContoursClicked'], 100, update_deltaContoursClicked)
cv2.createTrackbar('Click Radius', config_window_name, cfg['maxRadiusFigueWithFigueShallow'], 50, update_maxRadiusFigueWithFigueShallow)
cv2.createTrackbar('Delta Point X', config_window_name, cfg['delta_Point'][0] + 50, 100, update_delta_Point_X)
cv2.createTrackbar('Delta Point Y', config_window_name, cfg['delta_Point'][1] + 50, 100, update_delta_Point_Y)

# --- Function to Save Config ---
def save_config():
    config_path = os.path.join(os.path.dirname(__file__), 'config.yml')
    try:
        with open(config_path, 'w') as f:
            yaml.dump(cfg, f, default_flow_style=False)
        print(f"Configuration saved to {config_path}")
    except Exception as e:
        print(f"Error saving configuration: {e}")

"""
Function main process
"""
def main_process():
  size_window = tuple(cfg['size_window']) 
  fullscreensize = tuple(cfg['fullscreensize'])
  
  if cfg['on_cam1'] and camera1:
    camera1.start_thread()
  if cfg['on_cam2'] and camera2:
    camera2.start_thread()
  
  maCam1 = ((0,0), (0,0), (0,0), (0,0))
  maCam1YXZ = (maCam1[0], maCam1[2], maCam1[1], maCam1[3])

  maCam2 = ((0,0), (0,0), (0,0), (0,0))
  maCam2YXZ = (maCam2[0], maCam2[2], maCam2[1], maCam2[3])
  
  is_detect_corners = False

  mousePos = smoothB.average_vecN_smooth(cfg['numAverageMouseMove'])

  valueCntNear = [smoothB.average_smooth(cfg['numEleArgvan'])] * cfg['n_points_touch']
  
  start_time = time.time()
  everyX = 1
  counterFrame = 0
  
  FPP = cfg['FramePerProcess']
  curFPP = 0
  while True:
    counterFrame+=1
    if (time.time() - start_time) > everyX :
      if cfg['show_FPS_console']:
        print("FPS: ", counterFrame / (time.time() - start_time))
      counterFrame = 0
      start_time = time.time()
      
    q = cv2.waitKey(1)
    if q == ord('q') or (cfg['on_cam1'] and camera1 and camera1.stopped) or (cfg['on_cam2'] and camera2 and camera2.stopped):
      if cfg['on_cam1'] and camera1:
        camera1.stop()
      if cfg['on_cam2'] and camera2:
        camera2.stop()
      break
    elif q == ord('s'):
        save_config()
    
    curFPP += 1
    if curFPP < FPP:
      continue
    else:
      curFPP = 0

    imgCam1 = None
    imgCam2 = None
    if cfg['on_cam1'] and camera1:
      imgCam1 = camera1.getFrame()
    if cfg['on_cam2'] and camera2:
      imgCam2 = camera2.getFrame()

    if (cfg['on_cam1'] and imgCam1 is None) or (cfg['on_cam2'] and imgCam2 is None):
        if not is_detect_corners:
             time.sleep(0.1) 
        continue

    if cfg['on_debug']:
      if cfg['on_cam1'] and imgCam1 is not None:
        cv2.imshow("Camera test 1", imgCam1)
        cv2.setMouseCallback("Camera test 1", onMouse, param = (imgCam1, cfg['gamma1']))
      
      if cfg['on_cam2'] and imgCam2 is not None:
        cv2.imshow("Camera test 2", imgCam2)
        cv2.setMouseCallback("Camera test 2", onMouse, param = (imgCam2, cfg['gamma2']))
      continue

    if not is_detect_corners:
      showQRcorners()
      is_detect_corners_1, is_detect_corners_2 = False, False
      if cfg['on_cam1'] and imgCam1 is not None:
        is_detect_corners_1, maCam1, maCam1YXZ = get4Corners(imgCam1, lambda x: (x[0], x[2], x[1], x[3]), delta_point=tuple(cfg['delta_point_qr']))
      if cfg['on_cam2'] and imgCam2 is not None:
        is_detect_corners_2, maCam2, maCam2YXZ = get4Corners(imgCam2, lambda x: (x[0], x[2], x[1], x[3]), delta_point=tuple(cfg['delta_point_qr']))

      if (is_detect_corners_1 or not cfg['on_cam1']) and (is_detect_corners_2 or not cfg['on_cam2']):
        is_detect_corners = True
        destroyQRcorners()
        print("Corners detected.")
      else:
          time.sleep(0.05) 
    else:
      if cfg['on_config']:
        if cfg['on_cam1'] and imgCam1 is not None:
          imgCam1Draw = np.copy(imgCam1)
          matrixBincase.draw_line(imgCam1Draw, maCam1YXZ[0], maCam1YXZ[1], maCam1YXZ[2], maCam1YXZ[3], 3)
          cv2.imshow("Camera test 1", imgCam1Draw)
          cv2.setMouseCallback("Camera test 1", onMouse, param = (imgCam1, cfg['gamma1']))
        
        if cfg['on_cam2'] and imgCam2 is not None:
          imgCam2Draw = np.copy(imgCam2)
          matrixBincase.draw_line(imgCam2Draw, maCam2YXZ[0], maCam2YXZ[1], maCam2YXZ[2], maCam2YXZ[3], 3)
          cv2.imshow("Camera test 2", imgCam2Draw)
          cv2.setMouseCallback("Camera test 2", onMouse, param = (imgCam2, cfg['gamma2']))

        continue

      contoursFigue_cam1 = []
      if cfg['on_cam1'] and imgCam1 is not None:
        contoursFigue_cam1 = auto_ProcessImage(imgCam1, maCam1YXZ, cfg['gamma1'], cfg['fillCam1_01'], cfg['noseCam1'], cfg['on_show_cam1'], cfg['on_cam1Hsv'], cfg['on_cam1Ycbcr'], cfg['on_cam1FTI'], "Camera test 1")

      contoursFigue_cam2 = []
      if cfg['on_cam2'] and imgCam2 is not None:
        contoursFigue_cam2 = auto_ProcessImage(imgCam2, maCam2YXZ, cfg['gamma2'], cfg['fillCam2_01'], cfg['noseCam2'], cfg['on_show_cam2'], cfg['on_cam2Hsv'], cfg['on_cam2Ycbcr'], cfg['on_cam2FTI'], "Camera test 2")
      
      if (not cfg['on_cam1']) or (not cfg['on_cam2']) or imgCam1 is None or imgCam2 is None:
        continue 
      
      imgCamFTI1 = auto_ProcessImage_onlyfti(imgCam1, maCam1YXZ) 
      imgCamFTI2 = auto_ProcessImage_onlyfti(imgCam2, maCam2YXZ) 
      
      imgCamFTI1Mask = auto_ProcessImage_onlyhand(imgCam1, maCam1YXZ, cfg['gamma1'], cfg['fillCam1_01'], cfg['noseCam1'])
      imgCamFTI2Mask = auto_ProcessImage_onlyhand(imgCam2, maCam2YXZ, cfg['gamma2'], cfg['fillCam2_01'], cfg['noseCam2'])
      
      imgCamFTI1gray = cv2.cvtColor(imgCamFTI1, cv2.COLOR_BGR2GRAY)
      imgCamFTI2gray = cv2.cvtColor(imgCamFTI2, cv2.COLOR_BGR2GRAY)
      
      res1 = cv2.bitwise_and(imgCamFTI1gray, imgCamFTI1gray, mask=imgCamFTI1Mask)
      res2 = cv2.bitwise_and(imgCamFTI2gray, imgCamFTI2gray, mask=imgCamFTI2Mask)
      
      kernel = np.ones((3, 3), np.float32)/9
      res1 = cv2.filter2D(res1, -1, kernel)
      res2 = cv2.filter2D(res2, -1, kernel)
      
      disparity = stereo.compute(res1, res2)
      norm_disparity = cv2.normalize(disparity, None, 10, 245, cv2.NORM_MINMAX)
      color_disparity = cv2.applyColorMap(norm_disparity.astype(np.uint8), cv2.COLORMAP_HSV)
      bul_map = cv2.addWeighted(res1, 0.5, res2, 0.5, 0)
      cv2.imshow("Debug 1", color_disparity)
      cv2.imshow("Debug 2", bul_map)

      list_5_bestest_hull_point = []
      if len(contoursFigue_cam1) > 0:
        list_highest_point_hull = []
        for hulls in contoursFigue_cam1:
          if len(hulls) > 0:
             highest_point_hull = max(hulls, key=lambda x: x[0][1])
             list_highest_point_hull.append(highest_point_hull[0])
        
        if list_highest_point_hull:
            list_highest_point_hull.sort(key=lambda x: -x[1])
            cnt_5_bestest_hull_point = cfg['n_points_touch'] 
            delta_Point_np = np.array(cfg['delta_Point'], dtype=np.int32)
            for point in list_highest_point_hull:
              list_5_bestest_hull_point.append(point + delta_Point_np) 
              cnt_5_bestest_hull_point-=1
              if cnt_5_bestest_hull_point <= 0:
                break

      isClicked = False
      isClickedPoints = [False] * len(list_5_bestest_hull_point)

      for i in range(0, min(len(list_5_bestest_hull_point), cfg['n_points_touch'])):
        valueCntNear[i].add(0)

      if len(contoursFigue_cam2) > 0 and len(list_5_bestest_hull_point) > 0:
          try:
              np_contours = np.vstack(contoursFigue_cam2).reshape(-1, 2)
              if np_contours.size > 0:
                  index_contourF = 0
                  kdtree = KDTree(np_contours, leaf_size=2)

                  for contourF in list_5_bestest_hull_point:
                      if index_contourF >= len(valueCntNear): break

                      cntNear = 0
                      contourF_reshape = contourF.reshape(1, -1)
                      cntNear = kdtree.query_radius(contourF_reshape, r=cfg['maxRadiusFigueWithFigueShallow'], count_only=True)[0]

                      valueCntNear[index_contourF].addPrev(cntNear)
                      cntArgvanNear = valueCntNear[index_contourF].getAverage()
                      if cfg['is_debug_clicked']:
                          print(f"Point {index_contourF}: AvgNear={cntArgvanNear:.2f} (Threshold: {cfg['deltaContoursClicked']})")
                      if cntArgvanNear > cfg['deltaContoursClicked']:
                          isClickedPoints[index_contourF] = True
                          isClicked = True
                      index_contourF += 1
          except ValueError as e:
              print(f"Error processing contoursFigue_cam2: {e}")
              isClicked = False
              isClickedPoints = [False] * len(list_5_bestest_hull_point)

      if cfg['on_black_points_touch_screen']:
        imgFigueDraw = np.zeros((size_window[1], size_window[0], 3), dtype=np.uint8)
        index_contourF = 0
        for point in list_5_bestest_hull_point:
           point_tuple = tuple(point.astype(int))
           color = tuple(cfg['color_clicked']) if isClickedPoints[index_contourF] else tuple(cfg['color_nonClicked'])
           radius = cfg['maxRadiusFigueContour']
           cv2.circle(imgFigueDraw, point_tuple, radius, color, -1, cv2.LINE_AA)
           index_contourF += 1
        
        imgFigueDraw_resized = cv2.resize(imgFigueDraw, fullscreensize)
        if not cfg['is_debug_clicked']:
          setFullScreenCV("Black points touch screen")
        cv2.imshow("Black points touch screen", imgFigueDraw_resized)

      mousePos.add((0, 0))
      if cfg['on_cam1'] and cfg['on_cam2'] and cfg['on_controller']:
        if len(list_5_bestest_hull_point) > 0:
          width, height = pyautogui.size()
          pointMouseNow = list_5_bestest_hull_point[0]
          if cfg['is_flip_mouse']:
            mouseComputer = (int(pointMouseNow[0]*width/size_window[0]), int(pointMouseNow[1]*height/size_window[1]))
          else:
            mouseComputer = (int(width - pointMouseNow[0]*width/size_window[0]), int(height - pointMouseNow[1]*height/size_window[1]))

          mouseComputer = (max(0, min(width - 1, mouseComputer[0])), max(0, min(height - 1, mouseComputer[1])))

          if mouseComputer >= (0, 0):
            smoothedMousePos = tuple(map(int, mousePos.addPrev(mouseComputer)))
            mouse.position = smoothedMousePos
            
            if isClickedPoints[0]:
              mouse.click(Button.left)

  cv2.destroyAllWindows()

"""
Run function main
"""
if __name__ == "__main__":
    main_process()
