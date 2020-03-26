#!/usr/bin/env python
import numpy as np
import misc
import matplotlib.pyplot as plt
from scipy.fftpack import fft
plt.switch_backend('Agg')

np.random.seed(0)

nx = 40
nens = 10000
ii, jj = np.mgrid[0:nx, 0:nx]
dist = np.sqrt((ii - jj)**2)
dist = np.minimum(dist, nx - dist)
L = 2
xens = np.random.multivariate_normal(np.zeros(nx), np.exp(-dist/L)*np.cos(np.pi*dist/L), nens).T
xp = np.zeros((nx, nens))
xm = np.mean(xens, axis=1)
for m in range(nens):
  xp[:, m] = xens[:, m] - xm

krange = np.array([5])
yp1 = xp.copy()
yp2 = xp.copy()
for m in range(nens):
  yp1[:, m] = misc.spec_bandpass(xp[:, m], krange, 1)
  yp2[:, m] = misc.spec_bandpass(xp[:, m], krange, 0)

Pt = np.matmul(yp1, yp2.T) / (10000-1)
ns = 200
P = np.matmul(yp1[:,0:ns], yp2[:,0:ns].T) / (ns-1)

ax = plt.subplot(221)
# c = ax.imshow(P, cmap='seismic')
# plt.colorbar(c)
plt.plot(Pt[20, :], color='k')
plt.plot(P[20, :], color='r')

ax = plt.subplot(223)
ax.contourf(Pt)

ax = plt.subplot(222)
ax.plot(yp1[:, 0:20])
ax = plt.subplot(224)
ax.plot(yp2[:, 0:20])

plt.savefig('1.pdf')
