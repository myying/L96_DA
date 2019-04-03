#!/usr/bin/env python
import numpy as np
import L96_model as L96
import config as p
import misc
import matplotlib.pyplot as plt
from matplotlib import ticker
plt.switch_backend('Agg')

# read in some initial conditions
xt = np.load("output/truth.npy")
nx, nt = xt.shape

num_sample = 100
err = np.zeros((num_sample, nx, nt))
for i in range(num_sample):
  xb = xt.copy()
  xb[:, 0] = xt[:, 0] + np.random.normal(loc=0.0, scale=0.001, size=[nx])
  for n in range(nt-1):
    xb[:, n+1] = L96.forward(xb[:, n], nx, p.F, p.dt)
  err[i, :, :] = xb - xt

#rmse = np.sqrt(np.mean(np.mean(err**2, axis=1), axis=0))
#plt.plot(rmse)

## bin error growth scenario (initial error, forecast time)
#dt = 0.01
#init_err = np.arange(0.005, 5, 0.005)
#fcst_time = np.arange(0.05, 5, 0.05)
#ni = init_err.size
#nj = fcst_time.size
#count = np.zeros((ni, nj)) + 1e-10
#growth = np.zeros((ni, nj)) + 1e-10
#for j in range(nj):
#  for k in range(500):
#    for s in range(num_sample):
#      err1 = np.sqrt(np.mean(err[s, :, k]**2))
#      err2 = np.sqrt(np.mean(err[s, :, k+int(fcst_time[j]/dt)]**2))
#      if err1<5 and err1>=0.005:
#        growth[int(err1/0.005)-1, j] += err2/err1
#        count[int(err1/0.005)-1, j] += 1
#growth = growth/count
#np.save('growth', growth)
# print(growth)

growth = np.load('growth.npy')
ii, jj = np.mgrid[0.005:5:0.005, 0.05:5:0.05]
ax = plt.subplot(111)
c = ax.contour(ii, jj, growth, (2, 5, 10, 20, 50, 100, 200), colors='k')
#ax.clabel(c, fmt='%.0f', inline=1)
plt.xscale('log')
# plt.yscale('log')
ax.set_xlabel('initial error')
ax.set_ylabel('forecast time')
# cbar = plt.colorbar(c, ticks=[0, 1, 2])
# cbar.ax.set_yticklabels(['1', '10', '100'])
plt.savefig('1.pdf')
