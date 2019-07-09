#!/usr/bin/env python
import numpy as np
import misc
import data_assimilation as DA
import config as p
import matplotlib.pyplot as plt
from scipy.fftpack import fft

plt.switch_backend('Agg')
plt.figure(figsize=(12, 7))

U = misc.fourier_basis(p.nx)
res = np.zeros(p.nx)
res[0:10] = 1.0
L = np.dot(U.T, np.dot(np.diag(res), U))
a = np.random.normal(0, 1, p.nx)
b = np.dot(L, a)
c = misc.spec_bandpass(a, np.array([5]), 0)

ax = plt.subplot(111)
# ax.contourf(U, cmap='seismic')
ax.plot(a, 'k')
ax.plot(b, 'r')
ax.plot(c, 'b')

plt.savefig('1.pdf')
