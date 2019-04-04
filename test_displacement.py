#!/usr/bin/env python
import numpy as np
import config as p
import matplotlib.pyplot as plt
import data_assimilation as DA
import misc
plt.switch_backend('Agg')

xt = np.load("output/truth.npy")
yo = np.load("output/obs.npy")
xens = np.load("output/ensemble_forecast.npy")

nx, nens, nt = xens.shape
xb = xens[:, 0, :]
# xb = misc.warp(xt, 1.0+np.zeros(xt.shape), 1.0+np.zeros(xt.shape))
dx, dt = DA.optical_flow(xb, xt, 2)
# dx[np.where(dx>0.1)] = 0.1
# dx[np.where(dx<-0.1)] = -0.1
# dt[np.where(dt>0.1)] = 0.1
# dt[np.where(dt<-0.1)] = -0.1
# print(np.mean(dx))
# print(np.mean(dt))
xa = misc.warp(xb, -dx, -dt)

ii, jj = np.mgrid[0:nx, 0:nt]
clevel = np.arange(-15, 20, 1)
plt.figure(figsize=(10, 10))

ax = plt.subplot(221)
ax.contourf(ii, jj, xt, clevel, cmap='seismic')
ax.contour(ii, jj, xt, (5,), colors='k')

ax = plt.subplot(222)
ax.contourf(ii, jj, xb, clevel, cmap='seismic')
ax.contour(ii, jj, xb, (5,), colors='r')
ax.contour(ii, jj, xt, (5,), colors='k')
ax.quiver(ii[:, ::10], jj[:, ::10], dx[:, ::10], dt[:, ::10], scale=50)

ax = plt.subplot(224)
ax.contourf(ii, jj, xa, clevel, cmap='seismic')
ax.contour(ii, jj, xa, (5,), colors='r')
ax.contour(ii, jj, xt, (5,), colors='k')

plt.savefig('1.pdf')

