import matplotlib.pyplot as plt

h = 0.08
nmax = 800
wx = [0] * nmax
wy = [0] * nmax
wz = [0] * nmax

t = [0] * nmax

M = 3.14
a = 3
b = 2
c = 1

wx[0] = 1
wy[0] = 1
wz[0] = 1

t[0] = 0

I1 = M * (b ** 2 + c ** 2) / 5
I2 = M * (a ** 2 + c ** 2) / 5
I3 = M * (a ** 2 + b ** 2) / 5

g1 = (I3 - I2) / I1
g2 = (I1 - I3) / I2
g3 = (I2 - I1) / I3

for n in range(nmax - 1):
    kx1 = - h * g1 * wy[n] * wz[n]
    ky1 = - h * g2 * wx[n] * wz[n]
    kz1 = - h * g3 * wx[n] * wy[n]

    kx2 = -h * g1 * (wy[n] + ky1) * (wz[n] + kz1)
    ky2 = -h * g2 * (wx[n] + kx1) * (wz[n] + kz1)
    kz2 = -h * g3 * (wx[n] + kx1) * (wy[n] + ky1)

    wx[n + 1] = wx[n] + (0.5 * (kx1 + kx2))
    wy[n + 1] = wy[n] + (0.5 * (ky1 + ky2))
    wz[n + 1] = wz[n] + (0.5 * (kz1 + kz2))

    t[n + 1] = n * h

plt.plot(t, wx)
plt.plot(t, wy)
plt.plot(t, wz)

plt.show()

# To summarise the algorithm:
# • Lines 2 to 11: define known constants, the stepsize, the total number of iterative
# steps, nmax , the size of the required arrays to store the output from the algorithm.
