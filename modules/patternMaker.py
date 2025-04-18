import cv2
import numpy as np

class PatternMaker:
    def __init__(self, cols, width, height, conner_size = (5, 5)):
        self.cols = cols
        self.rows = -1
        self.width = width
        self.height = height
        self.conner_size = conner_size
        self.g = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self.g.fill(255)

    def make_circle_pattern(self, radius = 20):
        xspacing = self.conner_size[0] // 2 + radius
        yspacing = self.conner_size[1] // 2 + radius

        cv2.circle(self.g, (xspacing, yspacing), radius, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(self.g, (self.width - xspacing, yspacing), radius, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(self.g, (xspacing, self.height - yspacing), radius, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(self.g, (self.width - xspacing, self.height - yspacing), radius, (0, 0, 0), -1, cv2.LINE_AA)

    def make_checkerboard_pattern(self):
        # optimized numpy-based checkerboard
        xspacing, yspacing = self.conner_size[0] // 2, self.conner_size[1] // 2
        tile_w = (self.width - self.conner_size[0]) // self.cols
        tile_h = tile_w
        rows = (self.height - self.conner_size[1]) // tile_h
        self.rows = rows
        # build and tile base pattern
        base = ((np.add.outer(np.arange(rows), np.arange(self.cols)) % 2) * 255).astype(np.uint8)
        pattern = np.kron(base, np.ones((tile_h, tile_w), dtype=np.uint8))
        # reset canvas and blit pattern
        self.g.fill(255)
        y0, x0 = yspacing, xspacing
        h, w = pattern.shape
        self.g[y0:y0+h, x0:x0+w] = pattern[..., None]

    def get(self):
        return self.g
    
    def get_size_chess(self):
        return [self.cols - 1, self.rows - 1]
