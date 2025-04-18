"""
Open library
"""
import cv2
import numpy as np
import math

from pyzbar.pyzbar import decode as decode_qr, ZBarSymbol

"""
Bincase library
"""
from modules.math import MatrixBincase
from modules.imageProcess import ImageProcessor
from modules.qrcode import QRCodeB

from config import cfg
from constant import *

"""
Init object
"""
matrix_bincase = MatrixBincase()
image_processor = ImageProcessor()
qr = QRCodeB(
    version=cfg['qr_version'],
    box_size=cfg['qr_box_size'],
    border=cfg['qr_border']
)

"""
Main
"""
mouse_x, mouse_y = 0, 0
list_4_points = []
list_10_hsv = []
def onMouse(event, x, y, flags, param):
    global mouse_x, mouse_y, list_4_points, list_10_hsv
    mouse_x, mouse_y = x, y
    if event == cv2.EVENT_LBUTTONDOWN:
        print('pos(x,y) = (', mouse_x, ",", mouse_y, ')', sep='')
        hsv_color = image_processor.get_hsv_pos(param[0], param[1], (mouse_y, mouse_x))
        print('hsv = (', hsv_color[0], ",", hsv_color[1], ",", hsv_color[2], ')', sep='')

        ycbcr_color = image_processor.get_ycbcr_pos(param[0], param[1], (mouse_y, mouse_x))
        print('ycbcr = (', ycbcr_color[0], ",", ycbcr_color[1], ",", ycbcr_color[2], ')', sep='')

        list_4_points.append((mouse_x, mouse_y))
        list_10_hsv.append(hsv_color)

        if len(list_4_points) >= 4:
            print('(', end='', sep='')
            for i in range(0, 4):
                print('(', list_4_points[i][0], ',', list_4_points[i][1], ')', end='', sep='')
                if i < 3:
                    print(',', end='', sep='')
            print(')')
            list_4_points.clear()
        if len(list_10_hsv) >= 10:
            print('(', end='', sep='')
            for i in range(0, 10):
                print('(', list_10_hsv[i][0], ',', list_10_hsv[i][1], ',', list_10_hsv[i][2], ')', end='', sep='')
                if i < 9:
                    print(',', end='', sep='')
            print(')')
            list_10_hsv.clear()

# Detect hand
def auto_ProcessImage_onlyhand(imgCam, maCamYXZ, gamma, fillCam_01, noseCam):
  ### Process image: Perspective, filter_color, filter_noise, findContours ###
  imgCamFTI = matrix_bincase.fast_tranform_image_opencv(imgCam, maCamYXZ, size_window)
  imgFigue = image_processor.detect_hand_v2(imgCamFTI, gamma, fillCam_01, noseCam)
  return imgFigue
  
def auto_ProcessImage_onlyfti(imgCam, maCamYXZ):
  ### Process image: Perspective, filter_color, filter_noise, findContours ###
  imgCamFTI = matrix_bincase.fast_tranform_image_opencv(imgCam, maCamYXZ, size_window)
  return imgCamFTI

# Detect hand
def auto_ProcessImage_nofti(imgCam, gamma, fillCam_01, noseCam, on_show_cam, on_camHsv, on_camYcbcr, on_camFTI, title_on_show_cam):
  ### Process image: Perspective, filter_color, filter_noise, findContours ###
  imgCamFTI = np.copy(imgCam)
  imgFigue = image_processor.detect_hand_v2(imgCamFTI, gamma, fillCam_01, noseCam)
  
  # cv2.RETR_EXTERNAL - Get outside
  # cv2.RETR_LIST - Get all
  contoursFigue, hierarchyFigue = cv2.findContours(imgFigue, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
  
  ### Debug mode ###
  if on_show_cam:
    imgFigueDraw = cv2.cvtColor(imgFigue, cv2.COLOR_GRAY2RGB)
    cv2.imshow(title_on_show_cam, imgFigueDraw)
    cv2.setMouseCallback(title_on_show_cam, onMouse, param = (imgCam, gamma))
  if on_camHsv:
    imgCamDraw = image_processor.get_hsv_image(np.copy(imgCamFTI), gamma)
    cv2.imshow(title_on_show_cam + "Hsv", imgCamDraw)
    cv2.setMouseCallback(title_on_show_cam + "Hsv", onMouse, param = (imgCamFTI, gamma))
  if on_camYcbcr:
    imgCamDraw = image_processor.get_ycbcr_image(np.copy(imgCamFTI), gamma)
    cv2.imshow(title_on_show_cam + "Ycbcr", imgCamDraw)
    cv2.setMouseCallback(title_on_show_cam + "Ycbcr", onMouse, param = (imgCamFTI, gamma))
  if on_camFTI:
    imgCamDraw = np.copy(imgCamFTI)
    cv2.imshow(title_on_show_cam + "FTI", imgCamDraw)
    cv2.setMouseCallback(title_on_show_cam + "FTI", onMouse, param = (imgCamFTI, gamma))
  
  return contoursFigue

# Detect hand
def auto_ProcessImage(imgCam, maCamYXZ, gamma, fillCam_01, noseCam, on_show_cam, on_camHsv, on_camYcbcr, on_camFTI, title_on_show_cam):
  ### Process image: Perspective, filter_color, filter_noise, findContours ###
  imgCamFTI = matrix_bincase.fast_tranform_image_opencv(imgCam, maCamYXZ, size_window)
  """
  imgFigue = image_processor.filter_Color(imgCamFTI, gamma, fillCam_01[0], fillCam_01[1])
  imgFigue = image_processor.image_noise_filter(imgFigue, cv2.MORPH_CLOSE, noseCam[0])
  imgFigue = image_processor.image_noise_filter(imgFigue, cv2.MORPH_OPEN, noseCam[1])
  """
  imgFigue = image_processor.detect_hand_v2(imgCamFTI, gamma, fillCam_01, noseCam)
  
  # cv2.RETR_EXTERNAL - Get outside
  # cv2.RETR_LIST - Get all
  # cv2.CHAIN_APPROX_SIMPLE - Get simple
  contoursFigue, hierarchyFigue = cv2.findContours(imgFigue, cv2.RETR_LIST, cv2.RETR_EXTERNAL)
  
  ### Debug mode ###
  if on_show_cam:
    imgFigueDraw = cv2.cvtColor(imgFigue, cv2.COLOR_GRAY2RGB)
    cv2.imshow(title_on_show_cam, imgFigueDraw)
    cv2.setMouseCallback(title_on_show_cam, onMouse, param = (imgCam, gamma))
  if on_camHsv:
    imgCamDraw = image_processor.get_hsv_image(np.copy(imgCamFTI), gamma)
    cv2.imshow(title_on_show_cam + "Hsv", imgCamDraw)
    cv2.setMouseCallback(title_on_show_cam + "Hsv", onMouse, param = (imgCamFTI, gamma))
  if on_camYcbcr:
    imgCamDraw = image_processor.get_ycbcr_image(np.copy(imgCamFTI), gamma)
    cv2.imshow(title_on_show_cam + "Ycbcr", imgCamDraw)
    cv2.setMouseCallback(title_on_show_cam + "Ycbcr", onMouse, param = (imgCamFTI, gamma))
  if on_camFTI:
    imgCamDraw = np.copy(imgCamFTI)
    cv2.imshow(title_on_show_cam + "FTI", imgCamDraw)
    cv2.setMouseCallback(title_on_show_cam + "FTI", onMouse, param = (imgCamFTI, gamma))
  
  return contoursFigue

# Auto detect corners
imgQRcorners = qr.given_image_corners_qr(fullscreensize, core_value_qr)
def setFullScreenCV(nameWindow):
  cv2.namedWindow(nameWindow, cv2.WND_PROP_FULLSCREEN)
  cv2.setWindowProperty(nameWindow, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
def showQRcorners():
  """
  Show QR code corners
  """
  setFullScreenCV("imgQRcorners")
  cv2.imshow("imgQRcorners", imgQRcorners)
def destroyQRcorners():
  cv2.destroyWindow("imgQRcorners")
  
def get4Corners(imgCam, lambda_format_ma, delta_point = (0, 0)):
  global imgQRcorners
  maCam = ((0, 0), (0, 0), (0, 0), (0, 0))
  maCamYXZ = ((0, 0), (0, 0), (0, 0), (0, 0))
  is_detect_corners = False

  """
  Detect corners by QR
  """
  list_is_detect_corners = [False] * 4
  maCam_beta = [[0, 0]] * 4

  grey = cv2.cvtColor(imgCam, cv2.COLOR_BGR2GRAY)
  values = decode_qr(grey, symbols=[ZBarSymbol.QRCODE])
  if len(values) > 0:
    for value in values:
      qr_value = value.data.decode()
      points4 = [(p.x, p.y) for p in value.polygon]
      if qr_value in set_detect_value_qr:
        idx = set_detect_value_qr.index(qr_value)
        list_is_detect_corners[idx] = True
        maCam_beta[idx] = array_get_corner_qr[idx](points4)

  # cv2.imshow("imgCam", imgCam)
  # print(list_is_detect_corners)

  if list_is_detect_corners.count(True) == 4:
    is_detect_corners = True
  if is_detect_corners:
    maCam_beta += np.array([
      [-(delta_point[0]), -(delta_point[1])], 
      [(delta_point[0]), -(delta_point[1])], 
      [-(delta_point[0]), (delta_point[1])], 
      [(delta_point[0]), (delta_point[1])]], 
      dtype=np.float32)
      
    maCam = tuple(maCam_beta)
    maCamYXZ = lambda_format_ma(maCam)
  return is_detect_corners, maCam, maCamYXZ

def get4Corners_chess(imgCam, size_chess, lambda_format_ma, delta_point = (25, 30)):
  global imgQRcorners
  maCam = ((0, 0), (0, 0), (0, 0), (0, 0))
  maCamYXZ = ((0, 0), (0, 0), (0, 0), (0, 0))

  gray = cv2.cvtColor(imgCam, cv2.COLOR_BGR2GRAY)
  # Find the chess board corners
  ret, points_beta = cv2.findChessboardCorners(gray, size_chess, flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
  # print(points_beta.shape)
  if ret == True:
    points = np.array(list(map(lambda point: point[0], points_beta)))

    rect = np.zeros((4, 2), dtype = "float32")

    s = points.sum(axis = 1)
    rect[0] = points[np.argmin(s)]
    rect[3] = points[np.argmax(s)]

    diff = np.diff(points, axis = 1)
    rect[1] = points[np.argmin(diff)]
    rect[2] = points[np.argmax(diff)]

    rect += np.array([[-delta_point[0], -delta_point[1]], [delta_point[0], -delta_point[1]], [-delta_point[0], delta_point[1]], [delta_point[0], delta_point[1]]], dtype=np.float32)

    maCam = tuple(rect)
    maCamYXZ = lambda_format_ma(maCam)

  return ret, maCam, maCamYXZ

def get4Corners_circle(imgCam, lambda_format_ma, minDist, param1, param2, minRadius, maxRadius, delta_point = (25, 25)):
  maCam = ((0, 0), (0, 0), (0, 0), (0, 0))
  maCamYXZ = ((0, 0), (0, 0), (0, 0), (0, 0))

  gray = cv2.cvtColor(imgCam, cv2.COLOR_BGR2GRAY)
  circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, minDist, param1=param1, param2=param2, minRadius=minRadius, maxRadius=maxRadius)
  if (circles is not None) and circles[0].shape[0] >= 4:
    circles = np.around(circles)
    points_beta = circles[0]
    points = np.array(list(map(lambda point: (point[0], point[1]), points_beta)))

    rect = np.zeros((4, 2), dtype = "float32")

    s = points.sum(axis = 1)
    p_point = [np.argmin(s), 0, 0, np.argmax(s)]
    rect[0] = points[np.argmin(s)]
    rect[3] = points[np.argmax(s)]

    diff = np.diff(points, axis = 1)
    p_point[1] = np.argmin(diff)
    p_point[2] = np.argmax(diff)
    rect[1] = points[np.argmin(diff)]
    rect[2] = points[np.argmax(diff)]

    rect += np.array([
      [-(delta_point[0] + points_beta[p_point[0]][2]), -(delta_point[1] + points_beta[p_point[0]][2])], 
      [(delta_point[0] + points_beta[p_point[1]][2]), -(delta_point[1] + points_beta[p_point[1]][2])], 
      [-(delta_point[0] + points_beta[p_point[2]][2]), (delta_point[1] + points_beta[p_point[2]][2])], 
      [(delta_point[0] + points_beta[p_point[3]][2]), (delta_point[1] + points_beta[p_point[3]][2])]], 
      dtype=np.float32)

    maCam = tuple(rect)
    maCamYXZ = lambda_format_ma(maCam)
    return True, maCam, maCamYXZ
    
  return False, maCam, maCamYXZ

def increase_brightness(img, value=30):
  hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
  h, s, v = cv2.split(hsv)

  lim = 255 - value
  v[v > lim] = 255
  v[v <= lim] += value

  final_hsv = cv2.merge((h, s, v))
  img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
  return img
  
def decrease_brightness(img, value=30):
  hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
  h, s, v = cv2.split(hsv)

  lim = value
  v[v < lim] = 0
  v[v >= lim] -= value

  final_hsv = cv2.merge((h, s, v))
  img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
  return img

def distanceB2Points(p0, p1):
    return math.sqrt((p0[0] - p1[0])**2 + (p0[1] - p1[1])**2)