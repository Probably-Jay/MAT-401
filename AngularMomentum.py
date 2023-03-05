import numpy as np


class AngularMomentum(object):
    def __init__(self, x: float = 0, y: float = 0, z: float = 0):
        self.data = np.array([float(x), float(y), float(z)])

    @property
    def x(self) -> float:
        return self.data[0]

    @x.setter
    def x(self, val: float):
        self.data[0] = val

    @property
    def y(self) -> float:
        return self.data[1]

    @y.setter
    def y(self, val: float):
        self.data[1] = val

    @property
    def z(self) -> float:
        return self.data[2]

    @z.setter
    def z(self, val: float):
        self.data[2] = val
