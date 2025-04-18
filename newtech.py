"""
Open library
"""
import math
import cv2
import numpy as np
import logging
import threading
import time
# from data_struct.kd_tree import KdTree
import pyautogui
from pynput.mouse import Button, Controller

from pyzbar.pyzbar import decode as detectAndDecodeQR
from pyzbar.pyzbar import ZBarSymbol

from sklearn.neighbors import KDTree

"""
Bincase library
"""
from module.mathB import MatrixBincase
from module.projector import Projector
from module.camera import CameraWebIP, CameraSelf
# from module.detectB import DetectHander
from module.imageProcess import ImageProcessor
from module.qrcodeB import QRCodeB

from config.main import *
from constant.main import *
from supportFun.main import *

import module.smoothB as smoothB
import module.patternMakerB as patternMakerB
import module.calibrateCameraB as calibrationB
import module.screenshotB as screenshotB

"""
Init object
"""
# 4, 8, 12, 16, 20
# 8, 12, 16
# detectHander = DetectHander([8])

mouse = Controller()
matrixBincase = MatrixBincase()
imageProcesser = ImageProcessor()
pm1 = patternMakerB.PatternMaker(size_chess, fullscreensize[0], fullscreensize[1], corner_chess_size)
pm1.make_checkerboard_pattern()
cvt_c = pm1.get_size_chess()
print("Chess size: ", cvt_c)
calibration = calibrationB.Calibration(cvt_c, num_image_cal)
# ~ screenshotB = screenshotB.ScreenshotB(size_window)

detectQR = cv2.QRCodeDetector()
qr = QRCodeB(version=qr_version, box_size=qr_box_size, border=qr_border)

# np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

### Init camera ###
if on_cam1:
  camera1 = CameraSelf(0, size_window, cam1_exposure, cam1_exposure_auto, fps_cam1)
  camera1.is_flip = is_cam1_flip
  camera1.flip_mode = cam1_flip_mode
  # camera1 = CameraWebIP("http://192.168.137.217:8080/shot.jpg", size_window)

imgPattern = pm1.get()

"""
Function main process
"""
def main_process():
  global size_window, fullscreensize, on_controller, on_black_points_touch_screen
  
  camera1.startTheard()

  """
  Status system
  """
  # Matrix screen
  maCam1 = ((0,0), (0,0), (0,0), (0,0))
  maCam1YXZ = (maCam1[0], maCam1[2], maCam1[1], maCam1[3])
  
  maCam2 = ((0,0), (0,0), (0,0), (0,0))
  maCam2YXZ = (maCam1[0], maCam1[2], maCam1[1], maCam1[3])
  
  # Case status
  is_detect_corners = False
  is_detect_corners_1 = False
  is_detect_corners_2 = False
  
  array_points_paint = []

  # Event mouse
  mousePos = smoothB.average_vecN_smooth(numAverageMouseMove)
  is_mouse_time_start = False
  mouse_time_start = 0
  first_mouse_pos = (0, 0)
  last_mouse_pos = (0, 0)

  ### Smooth system ###
  valueCntNear = [smoothB.average_smooth(numEleArgvan)] * n_points_touch
  old_clicked = False
  old_right_clicked = False
  
  ### FPS system ###
  start_time = time.time()
  everyX = 1 # displays the frame rate every 1 second
  counterFrame = 0
  
  ### Currunt system config ###
  # 0 = hand, 1 = pen
  mode_running = 1
  
  """
  Loop frame
  """
  FPP = FramePerProcess
  curFPP = 0
  while True:
    """
    Exit action
    """
    q = cv2.waitKey(1)
    # ~ if q == ord('q') or camera1.stopped or screenshotB.stopped:
    if q == ord('q') or camera1.stopped:
      camera1.stop()
      # ~ screenshotB.stop()
      break
    if q == ord('1'):
      mode_running = 0
      print("switch to hand mode")
    if q == ord('2'):
      mode_running = 1
      print("switch to lazer mode")
    
    if q == ord('s'):
      on_black_points_touch_screen = not on_black_points_touch_screen
      cv2.destroyAllWindows()
      print("Switch black boand")
      
    if q == ord('c'):
      on_controller = not on_controller
      
      if on_controller == True:
        print("On controller")
      else:
        print("False controller")
    
    if q == ord('r'):
      camera1.setExposure(cam1_exposure, cam1_exposure_auto)
      cv2.destroyAllWindows()
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
      if show_FPS_console:
        print("FPS: ", counterFrame / (time.time() - start_time))
      counterFrame = 0
      start_time = time.time()

    ### Variable frame to image ###
    if on_cam1:
      imgCam1 = camera1.getFrame()

    # cv2.imshow("Camera test 1", imgCam1)
    # cv2.setMouseCallback("Camera test 1", onMouse, param = (imgCam1, gamma1))
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

    if (not is_detect_corners) or (not calibration.done):
      gray = cv2.cvtColor(imgCam1, cv2.COLOR_BGR2GRAY)
      # ~ gray = imageProcesser.adjust_gamma(gray, 0.8)
      cv2.imshow("Camera test 1", gray)
      
      if not is_detect_corners_1:
        showQRcorners()
        is_detect_corners_1, maCam1, maCam1YXZ = get4Corners(imgCam1, lambda x: (x[0], x[2], x[1], x[3]), (50, 50))
        if is_detect_corners_1:
          cv2.destroyAllWindows()
      else:
          if not calibration.done:
            setFullScreenCV("imgPattern")
            cv2.imshow("imgPattern", imgPattern)
            calibration.add(imgCam1)
            if calibration.done:
              # ~ screenshotB.startTheard()
              cv2.destroyAllWindows()
          else:
            showQRcorners()
            is_detect_corners_2, maCam2, maCam2YXZ = get4Corners(imgCam1, lambda x: (x[0], x[2], x[1], x[3]))
              
            if is_detect_corners_2:
              is_detect_corners = True
              cv2.destroyAllWindows()
        
    else:
      """
      Process find interaction
      """
      if mode_running == 0: # Mode figue
        camera1.setExposure(cam1_exposure, cam1_exposure_auto)
        
        """
        Camera 1: camera
        """
        contoursFigue_cam1 = []
        if on_cam1:
          contoursFigue_cam1 = auto_ProcessImage_nofti(imgCam1_onlyc1, gamma1, fillCam1_01, noseCam1, on_show_cam1, on_cam1Hsv, on_cam1Ycbcr, on_cam1FTI, "Camera test 1")

      elif mode_running == 1: # Mode lazer pen
        camera1.setExposure(10, 1)
        
        """
        Camera 1: lazer
        """
        contoursFigue_cam1 = []
        if on_cam1:
          imgCam, maCamYXZ, gamma, fillCam_01, noseCam, on_show_cam, on_camHsv, on_camYcbcr, on_camFTI, title_on_show_cam = imgCam1, maCam1YXZ, gamma1, fillCam1_01, noseCam1, on_show_cam1, on_cam1Hsv, on_cam1Ycbcr, on_cam1FTI, "Camera test 1"
          
          imgCamFTI = np.copy(imgCam)
          imgFigue = cv2.inRange(imgCamFTI, (0, 0, 60), (255, 255, 255))
          
          # cv2.RETR_EXTERNAL - Get outside
          # cv2.RETR_LIST - Get all
          contoursFigue_cam1, hierarchyFigue = cv2.findContours(imgFigue, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
          ### Debug mode ###
          if on_show_cam:
            imgFigueDraw = cv2.cvtColor(imgFigue, cv2.COLOR_GRAY2RGB)
            cv2.imshow(title_on_show_cam, imgFigueDraw)
            cv2.setMouseCallback(title_on_show_cam, onMouse, param = (imgCam, gamma))
          if on_camHsv:
            imgCamDraw = imageProcesser.get_hsv_image(np.copy(imgCamFTI), gamma)
            cv2.imshow(title_on_show_cam + "Hsv", imgCamDraw)
            cv2.setMouseCallback(title_on_show_cam + "Hsv", onMouse, param = (imgCamFTI, gamma))
          if on_camYcbcr:
            imgCamDraw = imageProcesser.get_ycbcr_image(np.copy(imgCamFTI), gamma)
            cv2.imshow(title_on_show_cam + "Ycbcr", imgCamDraw)
            cv2.setMouseCallback(title_on_show_cam + "Ycbcr", onMouse, param = (imgCamFTI, gamma))
          if on_camFTI:
            imgCamDraw = np.copy(imgCamFTI)
            cv2.imshow(title_on_show_cam + "FTI", imgCamDraw)
            cv2.setMouseCallback(title_on_show_cam + "FTI", onMouse, param = (imgCamFTI, gamma))
      
      """
      Only for cams activate
      """
      if (not on_cam1):
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
          cnt_5_bestest_hull_point = n_points_touch
          for point in list_highest_point_hull:
            list_5_bestest_hull_point.append(point + delta_Point)
            cnt_5_bestest_hull_point-=1
            if cnt_5_bestest_hull_point <= 0:
              break
          for i in range(0, len(list_5_bestest_hull_point)):
            list_5_bestest_hull_point[i] = matrixBincase.tramform_points(list_5_bestest_hull_point[i], maCam2YXZ, size_window)
          
          """
          Caculate info of convexHull
          """
          if len(contoursFigue_cam1) > 0:
            np_contours = np.vstack(contoursFigue_cam1).reshape((-1, 1, 2))
            chull = cv2.convexHull(np_contours)
            
            imgDrawC = np.zeros((size_window[1], size_window[0], 3))
            imgDrawC = cv2.drawContours(imgDrawC, chull, -1, (0,255,0), 3)
            
            areaValueOr = cv2.contourArea(chull)
            
            Moo = cv2.moments(chull)
            cx = int(Moo['m10']/Moo['m00'])
            cy = int(Moo['m01']/Moo['m00'])
            vectorClickLen = distanceB2Points(list_5_bestest_hull_point[0], [cx, cy])
            
            (x,y),radius = cv2.minEnclosingCircle(chull)
            center = (int(x),int(y))
            radius = int(radius)
            areaValueCr = radius*radius*np.pi
            
            ratioClicked = (areaValueOr + 0.001) / (areaValueCr + 0.001)
            
            cv2.circle(imgDrawC, center, radius, color_clicked, 1, cv2.LINE_AA)
            cv2.imshow("imgDrawC", imgDrawC)
              
        elif mode_running == 1:
          cnt_5_bestest_hull_point = n_points_touch
          for hulls in contoursFigue_cam1:
            point = np.median(hulls, axis=0)[0]
            list_5_bestest_hull_point.append((int(point[0] + delta_Point[0]), int(point[1] + delta_Point[1])))
            cnt_5_bestest_hull_point-=1
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
      if on_black_points_touch_screen:
        # imgFigueDraw = np.copy(imgCamFTI)
        imgFigueDraw = np.zeros((size_window[1], size_window[0], 3))
        imgFigueDraw[:,:] = (0, 0, 0)
        index_contourF = 0
        if on_paint_test:
          for point in array_points_paint:
            cv2.circle(imgFigueDraw, point, maxRadiusFigueContour, (255, 255, 255), -1, cv2.LINE_AA)
        for point in list_5_bestest_hull_point:
          if isClickedPoints[index_contourF]:
            array_points_paint.append(point)
            cv2.circle(imgFigueDraw, point, maxRadiusFigueContour, color_clicked, -1, cv2.LINE_AA)
          else:
            cv2.circle(imgFigueDraw, point, maxRadiusFigueContour, color_nonClicked, -1, cv2.LINE_AA)
          index_contourF += 1
        imgFigueDraw = cv2.resize(imgFigueDraw, fullscreensize)
        if not is_debug_clicked:
          setFullScreenCV("Black points touch screen")
        cv2.imshow("Black points touch screen", imgFigueDraw)

      """
      Process UI, Control mouse or touchscreen
      """
      mousePos.add((0, 0))
      if on_cam1 and on_controller:
        if len(list_5_bestest_hull_point) > 0:
          # Convert img pos to window pos
          width, height = pyautogui.size()
          pointMouseNow = list_5_bestest_hull_point[0]
          if is_flip_mouse:
            mouseComputer = (int(pointMouseNow[0]*width/size_window[0]), int(pointMouseNow[1]*height/size_window[1]))
          else:
            mouseComputer = (int(width - pointMouseNow[0]*width/size_window[0]), int(height - pointMouseNow[1]*height/size_window[1]))
          
          # Time
          if isClicked:
            if not is_mouse_time_start:
              mouse_time_start = time.time()
              is_mouse_time_start = True
              first_mouse_pos = mouseComputer
              last_mouse_pos = mouseComputer
            if distanceB2Points(mouseComputer,first_mouse_pos) > distanceB2Points(last_mouse_pos, first_mouse_pos):
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
          elif isClicked and ((time.time() - mouse_time_start) > time_delay_press):
            mouse.press(Button.left)
            old_clicked = True
          elif not isClicked:
            if is_mouse_time_start:
              if ((time.time() - mouse_time_start) > time_delay_press):
                mouse.release(Button.left)
              
              if (not old_right_clicked):
                # ~ print((time.time() - mouse_time_start), distanceB2Points(last_mouse_pos, first_mouse_pos))
                if ((time.time() - mouse_time_start) > time_delay_right_click) and (distanceB2Points(last_mouse_pos, first_mouse_pos) <= circle_in_right_click):
                  mouse.click(Button.right)
                  old_right_clicked = True
                
              # mouse.position = (0, 0)
            
            old_clicked = False
            old_right_clicked = False
            is_mouse_time_start = False
              
        else:
          if is_mouse_time_start:
            if ((time.time() - mouse_time_start) > time_delay_press):
              mouse.release(Button.left)
            
            if (not old_right_clicked):
              # ~ print((time.time() - mouse_time_start), distanceB2Points(last_mouse_pos, first_mouse_pos))
              if ((time.time() - mouse_time_start) > time_delay_right_click) and (distanceB2Points(last_mouse_pos, first_mouse_pos) <= circle_in_right_click):
                mouse.click(Button.right)
                old_right_clicked = True
              
            # mouse.position = (0, 0)
          
          old_clicked = False
          old_right_clicked = False
          is_mouse_time_start = False

"""
Run function main
"""
main_process()
