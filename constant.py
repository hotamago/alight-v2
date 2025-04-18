"""
Open library
"""
import pyautogui

"""
Init constant
"""
# Screen
# fullscreensize = (1024, 740)
fullscreensize = pyautogui.size()
# ~ size_window = (640, 480)
# ~ size_window = (800, 400)
# size_window = (640, 480)
size_window = (800, 800)
delta_point = 10

# QR
core_value_qr = "bin"
set_detect_value_qr = ["tl", "tr", "bl", "br"]
set_detect_value_qr = list(map(lambda x: core_value_qr + "-" + x, set_detect_value_qr))
# QR lambda
lambda_qr_t = lambda point: point[1]
lambda_qr_l = lambda point: point[0]
lambda_qr_b = lambda point: -point[1]
lambda_qr_r = lambda point: -point[0]
sign_qr = ((-1, -1), (1, -1), (-1, 1), (1, 1))
# QR function, array lambda
def get_corner_qr(list_points, cmp1, cmp2, sign):
  list_points.sort(key=cmp1)
  list_points = [list_points[i] for i in range(2)]
  list_points.sort(key=cmp2)
  plus0 = int(delta_point * sign[0])
  plus1 = int(delta_point * sign[1])
  return (list_points[0][0] + plus0, list_points[0][1] + plus1)
array_get_corner_qr = [
  lambda x: get_corner_qr(x, lambda_qr_t, lambda_qr_l, sign_qr[0]),
  lambda x: get_corner_qr(x, lambda_qr_t, lambda_qr_r, sign_qr[1]),
  lambda x: get_corner_qr(x, lambda_qr_b, lambda_qr_l, sign_qr[2]),
  lambda x: get_corner_qr(x, lambda_qr_b, lambda_qr_r, sign_qr[3]),
  ]
