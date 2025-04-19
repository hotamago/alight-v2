# Detect projector screen corners
import cv2
import numpy as np

def detect_corners(caped_frame, lower_bound=None):
    # Preprocess
    gray = cv2.cvtColor(caped_frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3,3), 0)
    edged = cv2.Canny(blur, 50, 150)

    # Find and sort contours
    cnts, _ = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    # Look for quadrilateral
    screen_corners = None
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.01 * peri, True)
        if len(approx) == 4:
            screen_corners = approx.reshape(4, 2)
            if lower_bound is not None:
                # check if lower_bound (x,y) is inside the detected quadrilateral
                cnt = screen_corners.astype(np.int32)
                isOutside = False
                for i in range(4):
                    if cv2.pointPolygonTest(cnt, lower_bound[i], True) < 0:
                        isOutside = True
                        break
                if isOutside:
                    screen_corners = None
                    continue
            break

    if screen_corners is not None:
        # Order corners: tl, tr, br, bl
        def order_points(pts):
            rect = np.zeros((4,2), dtype='float32')
            s = pts.sum(axis=1)
            diff = np.diff(pts, axis=1)
            rect[1] = pts[np.argmin(s)]       # top-left
            rect[2] = pts[np.argmax(s)]       # bottom-right
            rect[0] = pts[np.argmin(diff)]    # top-right
            rect[3] = pts[np.argmax(diff)]    # bottom-left
            return rect

        screen_corners = order_points(screen_corners)
        
        return screen_corners
    
    return None