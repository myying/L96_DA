#!/usr/bin/env python
import numpy as np
import misc
import data_assimilation as DA
import config as p
import matplotlib.pyplot as plt
from scipy.fftpack import fft
plt.switch_backend('Agg')
plt.figure(figsize=(3, 3))
# plt.figure(figsize=(12, 7))
nx = p.nx
##Fourier basis
v = np.zeros((nx, nx))
v[:, 0] = 1.0/np.sqrt(nx)
ii = np.mgrid[0:nx]
for n in range(1, nx, 2):
  v[:, n] = np.cos(np.pi*(n+1)*ii/nx)/np.sqrt(nx/2)
for n in range(2, nx, 2):
  v[:, n] = -np.sin(np.pi*n*ii/nx)/np.sqrt(nx/2)
v[:, -1] = np.cos(np.pi*ii)/np.sqrt(nx)


ax = plt.subplot(111)
for L in [0, 2, 5, 10]:
  R = DA.R_matrix(nx, 1, p.obs_ind, 1, L, 0, 2)
  wo = np.diag(np.dot(v.T, np.dot(R, v)))
  ax.plot(np.sqrt(wo[::2]), label='L = {}'.format(L))
  # ax.plot(R[0, :], label='L = {}'.format(L))
ax.legend(fontsize=10)
ax.set_ylim(0, 3)
# ax.set_xlim(0, 20)
# ax.set_ylim(-1, 2)

plt.savefig('1.pdf')
