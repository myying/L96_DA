#!/usr/bin/env python
import numpy as np
import param as p
import DA
import misc
from scipy.fftpack import fft
import matplotlib.pyplot as plt
plt.switch_backend('Agg')

x = np.load("data_xt.npy")[:, 0]
dist = np.abs(np.arange(p.nx) - 30)
dist = np.minimum(dist, p.nx - dist)
rho = DA.GC_local_func(dist, 8)
x1 = misc.spec_convolve(misc.grid2spec(rho), misc.grid2spec(x))

plt.subplot(211)
plt.plot(x, 'k')
plt.plot(rho, 'b')
plt.plot(rho * x, 'r')
plt.plot(misc.spec2grid(x1), 'c')
plt.axis([0, p.nx, -10, 15])
plt.subplot(212)
plt.plot(np.real(misc.grid2spec(x)), 'k')
# plt.plot(np.imag(misc.grid2spec(x)), 'k')
plt.plot(np.real(misc.grid2spec(rho)), 'b')
# plt.plot(np.imag(misc.grid2spec(rho)), 'b')
plt.plot(np.real(misc.grid2spec(rho * x)), 'r')
# plt.plot(np.imag(misc.grid2spec(rho * x)), 'r')
plt.plot(np.real(x1), 'c')
plt.axis([0, p.nx/2+1, -1, 2])
plt.savefig("1.pdf")
