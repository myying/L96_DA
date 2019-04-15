#!/usr/bin/env python
import numpy as np
import misc
import data_assimilation as DA
import config as p
import matplotlib.pyplot as plt
from scipy.fftpack import fft
plt.switch_backend('Agg')
plt.figure(figsize=(8, 8))

nx = p.nx
v = misc.fourier_basis(nx)

###SOAR process with parameters f1 and f2
####x(t) = f1 x(t-1) + f2 x(t-2) + e(t)
nsample = 10000
x = np.zeros((nx, nsample))
f1 = 0.6
f2 = 0.0
sig = 1
for n in range(nsample):
  x[0, n] = np.random.normal(0, sig)
  x[1, n] = f1*x[0, n] + np.random.normal(0, sig)
  x[nx-1, n] = f1*x[0, n] + np.random.normal(0, sig)
  # for i in range(int(nx/2)-2):
  for i in range(nx-2):
    x[i+2, n] = f1*x[i+1, n] + f2*x[i, n] + np.random.normal(0, sig)
  # for i in range(nx-1, int(nx/2)+1, -1):
    # x[i-2, n] = f1*x[i-1, n] + f2*x[i, n] + np.random.normal(0, sig)

R = misc.error_covariance(x)
R1 = DA.R_matrix(nx, 1, p.obs_ind, np.sqrt(sig**2/(1-f1**2)), 2, 0, 1)
print(np.mean(np.diag(R)))
print(np.mean(np.diag(R1)))

ax = plt.subplot(221)
wo = np.diag(np.dot(v.T, np.dot(R, v)))
wo1 = np.diag(np.dot(v.T, np.dot(R1, v)))
ax.plot(np.sqrt(wo[::2]), 'r')
ax.plot(np.sqrt(wo1[::2]), 'k')
ax.set_ylim(0, 3)
ax.set_xlabel('wavenumber')

ax = plt.subplot(222)
ax.plot(R[20, :], 'r')
ax.plot(R1[20, :], 'k')
ax.set_xlim(0, nx)
ax.set_ylim(-1, 2)
ax.set_xlabel('distance')

ax = plt.subplot(223)
ax.contourf(R, np.arange(-2, 2.1, 0.1), cmap='seismic')
ax = plt.subplot(224)
ax.contourf(R1, np.arange(-2, 2.1, 0.1), cmap='seismic')

###spectral distribution
# ax = plt.subplot(223)
# for L in [0, 2, 5, 10]:
#   R = DA.R_matrix(nx, 1, p.obs_ind, 1, L, 0, 1)
#   wo = np.diag(np.dot(v.T, np.dot(R, v)))
#   ax.plot(np.sqrt(wo[::2]), label='L = {}'.format(L))
# ax.legend(fontsize=10)
# ax.set_ylim(0, 3)
# ax.set_xlabel('wavenumber')

###correlation length scale
# ax = plt.subplot(224)
# for L in [0, 2, 5, 10]:
#   R = DA.R_matrix(nx, 1, p.obs_ind, 1, L, 0, 1)
#   ax.plot(R[0, :], label='L = {}'.format(L))
# ax.legend(fontsize=10)
# ax.set_xlim(0, 20)
# ax.set_ylim(-1, 2)
# ax.set_xlabel('distance')

plt.savefig('1.pdf')
