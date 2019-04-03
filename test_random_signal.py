#!/usr/bin/env python
import numpy as np
import misc
import matplotlib.pyplot as plt
from scipy.fftpack import fft
plt.switch_backend('Agg')

nx = 40
nens = 10
ii, jj = np.mgrid[0:nx, 0:nx]
dist = np.sqrt((ii - jj)**2)
dist = np.minimum(dist, nx - dist)
xens = np.random.multivariate_normal(np.zeros(nx), np.exp(-dist/1), nens).T
xp = np.zeros((nx, nens))
xm = np.mean(xens, axis=1)
for m in range(nens):
  xp[:, m] = xens[:, m] - xm

P = np.matmul(xp, xp.T) / (nens-1)

ax = plt.subplot(221)
# c = ax.imshow(P, cmap='seismic')
# plt.colorbar(c)
plt.plot(P[20, :])

ax = plt.subplot(222)
ax.plot(xp[:, 0:100])

xph = np.zeros((nx, nens), dtype=complex)
for m in range(nens):
  xph[:, m] = fft(xp[:, m])

# pwr = np.sqrt(np.real(xph*np.conj(xph)))
P1 = np.real(np.matmul(xph, np.conj(xph).T)) / (nens-1)

ax = plt.subplot(223)
# c = ax.imshow(P1, cmap='seismic')
# plt.colorbar(c)
plt.plot(P1[2, :])
# ax.set_xlim(0, int(nx/2)+1)
# ax.set_ylim(0, int(nx/2)+1)

ax = plt.subplot(224)
ax.plot(np.real(xph[:, 0:100]))
ax.set_xlim(0, int(nx/2)+1)

plt.savefig('1.pdf')
