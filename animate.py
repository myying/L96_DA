#!/usr/bin/env python
import numpy as np
import config as p
import matplotlib.pyplot as plt
import sys
plt.switch_backend('Agg')

xt = np.load("output/truth.npy")
yo = np.load("output/obs.npy")
xens = np.load("output/ensemble_forecast.npy")

nx, nens, nt = xens.shape
tt = int(sys.argv[1])
ax = plt.subplot(211)
ax.plot(xens[:, 0:100, tt], 'c')
ax.plot(np.mean(xens[:, :, tt], axis=1), 'b')
ax.plot(xt[:, tt], 'k')
ax.set_ylim(-10, 15)

ax = plt.subplot(212)
rmse = np.sqrt(np.mean((np.mean(xens, 1) - xt)**2, 0))
sprd = np.sqrt(np.mean(np.std(xens, axis=1)**2, 0))
ax.plot(rmse[:tt], 'k', label='error')
ax.plot(sprd[:tt], 'g', label='spread')
ax.set_xlabel('time')
ax.set_xticks(np.arange(0, 220, 20))
ax.set_xticklabels(np.round(np.arange(0, 220, 20)*0.02, 2))
ax.set_xlim(0, 100)
ax.set_ylim(0, 5)
ax.legend(fontsize=12, loc=1)

plt.savefig('animation/{:04d}.png'.format(tt), dpi=200)
