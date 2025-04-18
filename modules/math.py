import numpy as np
import cv2


class MatrixBincase:
    # ma is matrix 2x3
    def __init__(self):
        pass

    def find_coeffs(self, pa, pb):
        matrix = []
        for p1, p2 in zip(pa, pb):
            matrix.extend([
                [p1[0], p1[1], 1, 0, 0, 0, -p2[0]*p1[0], -p2[0]*p1[1]],
                [0, 0, 0, p1[0], p1[1], 1, -p2[1]*p1[0], -p2[1]*p1[1]]
            ])
        A = np.array(matrix, dtype=float)
        B = np.array(pb).reshape(8)
        res = np.linalg.solve(A, B)
        coeffs = np.concatenate([res, [1.0]])
        return coeffs.reshape(3, 3).astype(np.float32)

    def tranform_from_matrix(self, xy, ma):
        return np.array([(ma[0][0]*xy[0] + ma[0][1]*xy[1] + ma[0][2])/(ma[2][0]*xy[0] + ma[2][1]*xy[1] + ma[2][2]), (ma[1][0]*xy[0] + ma[1][1]*xy[1] + ma[1][2])/(ma[2][0]*xy[0] + ma[2][1]*xy[1] + ma[2][2])])

    def tranform_image_maxtrix(self, img, ma):
        # Get height and width, handling both grayscale and color images
        height, width = img.shape[:2]
        # Use cv2.warpPerspective for optimized transformation
        # ma is assumed to be the forward transformation matrix (src -> dst)
        # dsize is (width, height) for the output image size
        imgFinal = cv2.warpPerspective(
            img, ma, (width, height), flags=cv2.INTER_LINEAR)
        return imgFinal

    def tramform_points(self, pos, po4, wh, mode=0):
        M = cv2.getPerspectiveTransform(np.float32(po4), np.float32(
            ((0, 0), (0, wh[1]), (wh[0], 0), (wh[0], wh[1]))))
        # ma = np.linalg.inv(M)
        newXY = self.tranform_from_matrix(pos, M)
        newYX = np.array([int(newXY[0]), int(newXY[1])])
        if mode == 0:
            return newYX
        elif mode == 1:
            return np.array([int(newXY[1]), int(newXY[0])])
        return None

    def slow_tranform_image(self, img, po4, wh):
        M = self.find_coeffs(
            po4, ((0, 0), (0, wh[1]), (wh[0], 0), (wh[0], wh[1])))
        imgFinal = self.tranform_image_maxtrix(img, M)
        return imgFinal

    def fast_tranform_image_opencv(self, img, po4, wh):
        M = cv2.getPerspectiveTransform(np.float32(po4), np.float32(
            ((0, 0), (0, wh[1]), (wh[0], 0), (wh[0], wh[1]))))
        imgFinal = cv2.warpPerspective(
            img, M, (wh[0], wh[1]), flags=cv2.INTER_LINEAR)
        return imgFinal

    def draw_line(self, img, ro, x, y, z, size_line):
        # convert any point-like input to int tuple
        pts = [tuple(map(int, pt)) for pt in (ro, x, y, z)]
        bl, br, tl, tr = pts  # rename for clarity: bottom-left, bottom-right, top-left, top-right
        img = cv2.line(img, bl, tl, (255, 255, 255), size_line)
        img = cv2.line(img, bl, br, (255, 255, 255), size_line)
        img = cv2.line(img, tl, tr, (255, 255, 255), size_line)
        img = cv2.line(img, br, tr, (255, 255, 255), size_line)
        return img
