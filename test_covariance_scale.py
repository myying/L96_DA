#!/usr/bin/env python
import numpy as np
import misc
import config as p
import matplotlib.pyplot as plt
from scipy.fftpack import fft
plt.switch_backend('Agg')

# read in obs and prior ensemble
xens = np.load("output/ensemble_prior.npy")
nx, nens, nt = xens.shape

##time covariance
P1 = np.zeros((nt, nt))
for i in range(nx):
  x = xens[i, :, :].T
  P1 += (misc.error_covariance(x))
P1 = P1/nx

##spatial covariance
P2 = np.zeros((nx, nx))
for i in range(10, nt):
  x = xens[:, :, i]
  P2 += (misc.error_covariance(x))
P2 = P2/(nt-10)

ax = plt.subplot(221)
c = ax.contourf(P1, np.arange(-5, 5, 0.1), cmap='seismic')
plt.colorbar(c)
ax = plt.subplot(222)
ax.plot(P1[30, 0:60], 'k')

ax = plt.subplot(223)
c = ax.contourf(P2, np.arange(-5, 5, 0.1), cmap='seismic')
plt.colorbar(c)
ax = plt.subplot(224)
ax.plot(P2[20, :], 'k')

plt.savefig('1.pdf')
