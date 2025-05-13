import logging
import cv2
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Calibration:
    """
    Camera calibration using chessboard patterns.
    """

    def __init__(self, size_chess: tuple[int, int], num_image_cal: int = 15) -> None:
        self.size_chess = size_chess
        self.origin_num_image_cal = num_image_cal
        self.num_image_cal = num_image_cal
        self.criteria = (cv2.TERM_CRITERIA_EPS +
                         cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        self.objp = np.zeros((size_chess[0]*size_chess[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:size_chess[0],
                                    0:size_chess[1]].T.reshape(-1, 2)
        self.objpoints: list[np.ndarray] = []
        self.imgpoints: list[np.ndarray] = []
        self.done = False
        self.mtx = self.dist = self.newcameramtx = self.roi = None

    def add(self, img: np.ndarray) -> bool:
        """
        Process an image for calibration. Returns True if calibration is finished.
        """
        if self.done:
            logger.debug("Calibration already completed.")
            return True

        # Validate input
        if img is None or img.size == 0:
            logger.warning("Empty image provided.")
            return False

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if self.num_image_cal > 0:
            ret, corners = cv2.findChessboardCorners(
                gray, self.size_chess,
                flags=(
                    cv2.CALIB_CB_ADAPTIVE_THRESH |
                    cv2.CALIB_CB_FAST_CHECK |
                    cv2.CALIB_CB_NORMALIZE_IMAGE
                    )
                )
            if ret:
                self.num_image_cal -= 1
                self.objpoints.append(self.objp.copy())
                corners2 = cv2.cornerSubPix(
                    gray, corners, (11, 11), (-1, -1), self.criteria)
                self.imgpoints.append(corners2)
                logger.info(
                    f"Chessboard detected: {len(self.objpoints)}/{self.origin_num_image_cal}")

        if self.num_image_cal <= 0:
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
                self.objpoints, self.imgpoints, gray.shape[::-1], None, None)
            h, w = img.shape[:2]
            newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
                mtx, dist, (w, h), 1, (w, h))
            self.mtx, self.dist, self.newcameramtx, self.roi = mtx, dist, newcameramtx, roi
            self.done = True
            logger.info("Calibration complete.")
            return True

        return False

    def get(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, int, int, int]]:
        """
        Return calibration results. Raises if not done.
        """
        if not self.done:
            raise RuntimeError("Calibration not yet completed.")
        return self.mtx, self.dist, self.newcameramtx, self.roi

    def reset(self) -> None:
        """
        Reset calibration state.
        """
        self.num_image_cal = self.origin_num_image_cal
        self.done = False
        self.objpoints.clear()
        self.imgpoints.clear()

    def undistort(self, img: np.ndarray) -> np.ndarray:
        """
        Undistort an image using the calibration results.
        """
        if not self.done:
            raise RuntimeError("Calibration not yet completed.")
        return cv2.undistort(img, self.mtx, self.dist, None, self.newcameramtx)
