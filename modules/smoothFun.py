"""
Smooth class
"""

import math
import numpy as np


class average_smooth:
    _n, _index = 0, 0
    _arr = None

    def __init__(self, n):
        self._n = n
        self._arr = np.zeros(n)
        self._index = 0

    def add(self, value):
        self._arr[self._index] = value
        self._index = (self._index + 1) % self._n

    def addPrev(self, value):
        self._arr[(self._index - 1 + self._n) % self._n] = value

    def getAverage(self):
        return np.average(self._arr)


class average_vecN_smooth:
    _n, _dim = 0, 0
    _arr = []

    def __init__(self, n, dim=2):
        self._n, self._dim = n, dim
        self._arr = [average_smooth(n)] * dim

    def add(self, value):
        for i in range(self._dim):
            self._arr[i].add(value[i])

    def addPrev(self, value):
        for i in range(self._dim):
            self._arr[i].addPrev(value[i])

    def getAverage(self):
        res = [0] * self._dim
        for i in range(self._dim):
            res[i] = self._arr[i].getAverage()
        return res


class vectorN_smooth:
    _old_value = None
    _dim = 2
    _speed = 10

    def __init__(self, speed, dim=2):
        self._old_value = [0] * dim
        self._speed = speed
        self._dim = dim

    def update(self, value):
        sumPow = 0
        for i in range(self._dim):
            sumPow += pow(value[i] - self._old_value[i], 2)
        length = math.sqrt(sumPow)

        velocity = [0] * self._dim
        for i in range(self._dim):
            velocity[i] = (value[i] - self._old_value[i]) / (length + 1) * self._speed

        self._old_value = value

        for i in range(self._dim):
            value[i] += i
        return value


# --- One Euro Filter (1€) cho làm mượt tọa độ chuột ---
class OneEuroFilter:
    def __init__(self, freq=60.0, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        self.freq = float(freq)
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.x_prev = None
        self.dx_prev = None

    def _alpha(self, cutoff):
        # alpha = 1 / (1 + tau / te), tau = 1/(2π*cutoff), te = 1/freq
        tau = 1.0 / (2.0 * math.pi * cutoff)
        te = 1.0 / self.freq
        return 1.0 / (1.0 + tau / te)

    def filter(self, x, dt=None):
        if dt is not None and dt > 0:
            self.freq = 1.0 / dt
        if self.x_prev is None:
            self.x_prev, self.dx_prev = x, 0.0
            return x
        # ước lượng đạo hàm
        dx = (x - self.x_prev) * self.freq
        a_d = self._alpha(self.d_cutoff)
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev
        # cutoff phụ thuộc tốc độ
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._alpha(cutoff)
        x_hat = a * x + (1 - a) * self.x_prev
        self.x_prev, self.dx_prev = x_hat, dx_hat
        return x_hat
