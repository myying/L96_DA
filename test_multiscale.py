#!/usr/bin/env python
import numpy as np
import misc
import matplotlib.pyplot as plt

x = np.load("output/truth.npy")[:, 0]
nx = x.size

# krange = np.array([2, 5, 8])
krange = np.arange(0, 20, 2)
ns = krange.size + 1
xf = np.zeros([nx, ns])

plt.switch_backend('Agg')
ax = plt.subplot(111)
# ax.plot(x)
for s in range(ns):
  xf[:, s] = misc.spec_bandpass(x, krange, s)
# ax.plot(np.sum(xf, axis=1))
ax.plot(xf)

plt.savefig('1.pdf')
