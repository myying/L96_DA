#!/usr/bin/env python
import numpy as np
import param as p
import DA
import misc
from scipy.fftpack import fft
import matplotlib.pyplot as plt
plt.switch_backend('Agg')

for i in np.arange(p.nx):
    dist = np.abs(np.arange(p.nx) - i)
    dist = np.minimum(dist, p.nx - dist)
    rho = DA.GC_local_func(dist, 8)
    plt.figure()
    plt.subplot(211)
    plt.plot(rho)
    plt.axis([0, p.nx, 0, 1])
    plt.subplot(212)
    plt.plot(np.real(misc.grid2spec(rho)))
    plt.plot(np.imag(misc.grid2spec(rho)))
    plt.axis([0, p.nx/2+1, -0.5, 0.5])
    plt.savefig("{:03d}.png".format(i))
    plt.close()
