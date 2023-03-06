import numpy as np


class Vector(object):
    def __init__(self, x: float = 0, y: float = 0, z: float = 0):
        self.elements = np.array([float(x), float(y), float(z)])

    @property
    def x(self) -> float:
        return self.elements[0]

    @x.setter
    def x(self, val: float):
        self.elements[0] = val

    @property
    def y(self) -> float:
        return self.elements[1]

    @y.setter
    def y(self, val: float):
        self.elements[1] = val

    @property
    def z(self) -> float:
        return self.elements[2]

    @z.setter
    def z(self, val: float):
        self.elements[2] = val

    def elements(self) -> np.array:
        return self.elements

