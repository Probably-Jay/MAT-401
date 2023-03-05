import matplotlib.pyplot as plt
from AngularMomentum import AngularMomentum

timeStep = 0.08
numberOfIterations = 800

angularMomentum = [AngularMomentum() for i in range(numberOfIterations)]

# angularMomentum.x = [0] * numberOfIterations
# angularMomentum.y = [0] * numberOfIterations
# angularMomentum.z = [0] * numberOfIterations

t = [float(0)] * numberOfIterations

M = 3.14
a = 3
b = 2
c = 1


angularMomentum[0] = AngularMomentum(1, 1, 1)

t[0] = 0

I1 = M * (b ** 2 + c ** 2) / 5
I2 = M * (a ** 2 + c ** 2) / 5
I3 = M * (a ** 2 + b ** 2) / 5

g1 = (I3 - I2) / I1
g2 = (I1 - I3) / I2
g3 = (I2 - I1) / I3

for n in range(numberOfIterations - 1):
    kx1 = - timeStep * g1 * angularMomentum[n].y * angularMomentum[n].z
    ky1 = - timeStep * g2 * angularMomentum[n].x * angularMomentum[n].z
    kz1 = - timeStep * g3 * angularMomentum[n].x * angularMomentum[n].y

    kx2 = -timeStep * g1 * (angularMomentum[n].y + 0.5 * ky1) * (angularMomentum[n].z + 0.5 * kz1)
    ky2 = -timeStep * g2 * (angularMomentum[n].x + 0.5 * kx1) * (angularMomentum[n].z + 0.5 * kz1)
    kz2 = -timeStep * g3 * (angularMomentum[n].x + 0.5 * kx1) * (angularMomentum[n].y + 0.5 * ky1)

    (angularMomentum[n + 1]).x = angularMomentum[n].x + kx2
    (angularMomentum[n + 1]).y = angularMomentum[n].y + ky2
    (angularMomentum[n + 1]).z = angularMomentum[n].z + kz2

    t[n + 1] = n * timeStep

listX = list(map(lambda am: am.x, angularMomentum))
plt.plot(t, listX)
plt.plot(t, list(map(lambda am: am.y, angularMomentum)))
plt.plot(t, list(map(lambda am: am.z, angularMomentum)))

plt.show()

# To summarise the algorithm:
# • Lines 2 to 11: define known constants, the stepsize, the total number of iterative
# steps, nmax , the size of the required arrays to store the output from the algorithm.
