#!/usr/bin/env python
import numpy as np
import misc
import matplotlib.pyplot as plt
from scipy.fftpack import fft
plt.switch_backend('Agg')
fig = plt.figure(figsize=(8, 8))

# read in ensemble perturbations
xens = np.load("output/ensemble_forecast.npy")[:, 0:, :]
nx, nens, nt = xens.shape
print(xens.shape)

tt = 40
xp = np.zeros((nx, nens))
xm = np.mean(xens[:, :, tt], axis=1)
for m in range(nens):
  xp[:, m] = xens[:, m, tt] - xm

# u, s, v = np.linalg.svd(xp)
P = np.matmul(xp, xp.T) / (nens-1)

ax = plt.subplot(321)
c = ax.contourf(P, np.arange(-20, 21, 1)/1, cmap='seismic')
ax.set_xticks([0, 9, 19, 29, 39])
ax.set_xticklabels(['1', '10', '20', '30', '40'])
ax.set_yticks([0, 9, 19, 29, 39])
ax.set_yticklabels(['1', '10', '20', '30', '40'])
plt.colorbar(c)

ax = plt.subplot(323)
plt.plot(P[20, :])

ax = plt.subplot(325)
ax.plot(xp)

xph = np.zeros((nx, nens), dtype=complex)
for m in range(nens):
  xph[:, m] = fft(xp[:, m])
P1 = np.real(np.matmul(xph, np.conj(xph).T)) / (nens-1)

ax = plt.subplot(322)
c = ax.contourf(P1, np.arange(-1000, 1010, 10)/1, cmap='seismic')
ax.set_xticks([0, 10, 20, 29, 39])
ax.set_xticklabels(['0', '10', '20', '-11', '-1'])
ax.set_yticks([0, 10, 20, 29, 39])
ax.set_yticklabels(['0', '10', '20', '-11', '-1'])
plt.colorbar(c)

ax = plt.subplot(324)
plt.plot(P1[8, :])
# plt.plot(np.diag(P1))
# ax.set_xlim(0, int(nx/2)+1)
# ax.set_ylim(0, int(nx/2)+1)

ax = plt.subplot(326)
ax.plot(np.real(xph))
ax.set_xlim(0, int(nx/2)+1)

plt.savefig('1.pdf')
