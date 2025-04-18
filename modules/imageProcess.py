import cv2
import numpy as np


class ImageProcessor:
    def __init__(self):
        pass

    def undistort(self, img, mtx, dist, newcameramtx, roi):
        # undistort
        dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
        h_og,  w_og = img.shape[:2]
        # ~ mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w_og,h_og), 5)
        # ~ dst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
        # crop the image
        x, y, w, h = roi
        # ~ dst = dst[y:y+h, x:x+w]
        dst = cv2.resize(dst[y:y+h, x:x+w], (w_og, h_og))
        return dst

    def adjust_gamma(self, image, gamma):
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255
                          for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(image, table)

    def get_hsv_image(self, img, gamma):
        blurred = cv2.GaussianBlur(img, (9, 1), 0)
        # blurred = cv2.dilate(img, np.ones((15, 1), np.uint8))
        adjusted = self.adjust_gamma(blurred, gamma)
        hsv = cv2.cvtColor(adjusted, cv2.COLOR_BGR2HSV)
        return hsv

    def get_ycbcr_image(self, img, gamma):
        blurred = cv2.GaussianBlur(img, (9, 1), 0)
        # blurred = cv2.dilate(img, np.ones((15, 1), np.uint8))
        adjusted = self.adjust_gamma(blurred, gamma)
        hsv = cv2.cvtColor(adjusted, cv2.COLOR_BGR2YCrCb)
        return hsv

    def filter_Color(self, img, gamma, lower, upper):
        hsv = self.get_hsv_image(img, gamma)
        # ensure proper array types for inRange
        lowerb = np.array(lower, dtype=np.uint8)
        upperb = np.array(upper, dtype=np.uint8)
        mask = cv2.inRange(hsv, lowerb, upperb)
        ret, otsu = cv2.threshold(
            mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return otsu

    def get_hsv_pos(self, img, gamma, pos):
        hsv = self.get_hsv_image(img, gamma)
        return hsv[pos[0]][pos[1]]

    def get_ycbcr_pos(self, img, gamma, pos):
        hsv = self.get_ycbcr_image(img, gamma)
        return hsv[pos[0]][pos[1]]

    # cv2.MORPH_CLOSE cv2.MORPH_OPEN
    def image_noise_filter(self, img, type_filter, size, type_box=cv2.MORPH_RECT):
        # ensure ksize is a 2‐tuple of ints
        if isinstance(size, int):
            ksize = (size, size)
        else:
            ksize = tuple(size)
        kernel = cv2.getStructuringElement(type_box, ksize)
        return cv2.morphologyEx(img, type_filter, kernel)

    def image_noise_filter_both(self, img, size, type_box=cv2.MORPH_RECT):
        img = self.image_noise_filter(img, cv2.MORPH_CLOSE, size[0])
        img = self.image_noise_filter(img, cv2.MORPH_OPEN, size[1])
        return img

    def filter_Color_non(self, img, filter_color):
        # ensure proper array types for inRange
        lowerb = np.array(filter_color[0], dtype=np.uint8)
        upperb = np.array(filter_color[1], dtype=np.uint8)
        mask = cv2.inRange(img, lowerb, upperb)
        ret, otsu = cv2.threshold(
            mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return otsu

    def detect_hand_v2(self, img, gamma, fillCam_01, noseCam):
        # converting from gbr to hsv color space
        img_hsv = self.get_hsv_image(img, gamma)
        # skin color range for hsv color space
        hsv_mask = self.filter_Color_non(img_hsv, fillCam_01[0])
        hsv_mask = self.image_noise_filter_both(hsv_mask, noseCam)

        # converting from gbr to YCbCr color space
        img_ycbcr = self.get_ycbcr_image(img, gamma)
        # skin color range for hsv color space
        ycbcr_mask = self.filter_Color_non(img_ycbcr, fillCam_01[1])
        ycbcr_mask = self.image_noise_filter_both(ycbcr_mask, noseCam)

        # merge skin detection (ycbcr and hsv)
        global_mask = cv2.bitwise_and(ycbcr_mask, hsv_mask)
        global_mask = cv2.medianBlur(global_mask, 3)
        global_mask = self.image_noise_filter_both(global_mask, noseCam)

        global_result = global_mask

        res = global_result
        # res = self.image_noise_filter(res, cv2.MORPH_CLOSE, noseCam[0])
        # res = self.image_noise_filter(res, cv2.MORPH_OPEN, noseCam[1])

        return res
