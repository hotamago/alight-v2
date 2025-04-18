import cv2
import mediapipe as mp
import time


class DetectHander():
    size_circle = 10
    color_circle = (255, 0, 255)

    def __init__(self, list_id_hands):
        mpHands = mp.solutions.hands
        self.mpHands = mpHands
        # tuned and cached Hands instance
        self.hands = mpHands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mpDraw = mp.solutions.drawing_utils
        self.list_id_hands = list_id_hands

    def process(self, img):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return self.hands.process(imgRGB)

    # new helper to yield (id, x, y) for all landmarks
    def _extract_landmarks(self, img, results):
        h, w, _ = img.shape
        for handLms in (results.multi_hand_landmarks or []):
            for id, lm in enumerate(handLms.landmark):
                yield id, int(lm.x * w), int(lm.y * h)

    def get_pos_hands(self, img, results=None):
        if results is None:
            results = self.process(img)
        # use helper and filter by desired IDs
        return [
            (cx, cy)
            for id, cx, cy in self._extract_landmarks(img, results)
            if id in self.list_id_hands
        ]

    def draw_all_hands(self, img, results=None):
        if results is None:
            results = self.process(img)
        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                # corrected reference to HAND_CONNECTIONS
                self.mpDraw.draw_landmarks(
                    img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def draw_circle_hands(self, img, results=None):
        if results is None:
            results = self.process(img)
        # reuse helper for circle drawing
        for id, cx, cy in self._extract_landmarks(img, results):
            if id in self.list_id_hands:
                cv2.circle(img, (cx, cy), self.size_circle,
                           self.color_circle, cv2.FILLED)
        return img
