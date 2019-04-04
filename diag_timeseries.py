#!/usr/bin/env python
import numpy as np
import config as p
import matplotlib.pyplot as plt
plt.switch_backend('Agg')

xt = np.load("output/truth.npy")
yo = np.load("output/obs.npy")
xens0 = np.load("output/ensemble_forecast.npy")
xens = np.load("output/ensemble_prior.npy")
xens1 = np.load("output/ensemble_post.npy")

nx, nens, nt = xens.shape
tt = 80
ax = plt.subplot(211)
ax.plot(xens1[:, 0:20, tt], 'c')
ax.plot(np.mean(xens1[:, :, tt], axis=1), 'b')
ax.plot(p.obs_ind, yo[p.obs_ind, tt], 'rx')
ax.plot(xt[:, tt], 'k')

ax = plt.subplot(212)
rmse0 = np.sqrt(np.mean((np.mean(xens0, 1) - xt)**2, 0))
rmse = np.sqrt(np.mean((np.mean(xens, 1) - xt)**2, 0))
rmse1 = np.sqrt(np.mean((np.mean(xens1, 1) - xt)**2, 0))
sprd = np.sqrt(np.mean(np.std(xens, axis=1)**2, 0))
sprd1 = np.sqrt(np.mean(np.std(xens1, axis=1)**2, 0))
rmse_out = np.zeros(2*nt)
rmse_out[0:nt*2:2] = rmse
rmse_out[1:nt*2:2] = rmse1
sprd_out = np.zeros(2*nt)
sprd_out[0:nt*2:2] = sprd
sprd_out[1:nt*2:2] = sprd1
time = np.zeros(2*nt)
time[0:nt*2:2] = np.arange(0, nt)
time[1:nt*2:2] = np.arange(0, nt)
ax.plot(time, rmse_out, 'k')
ax.plot(time, sprd_out, 'g')
ax.plot(rmse0, 'r')
ax.plot(np.arange(nt+1), p.obs_err*np.ones(nt+1), color='0.7')
ax.set_xlabel('time')
ax.set_ylabel('rmse')

# xa = np.mean(xens, axis=1)
# ax = plt.subplot(121)
# ax.contourf(xa.T, np.arange(-10, 15, 1))
# ax = plt.subplot(122)
# ax.contourf((xa-xt).T, np.arange(-10, 15, 1))

print(np.mean(rmse[::p.cycle_period]))
print(np.mean(rmse1[::p.cycle_period]))
# err_reduc = rmse1[::p.cycle_period]/rmse[::p.cycle_period]
# print(np.mean(err_reduc[3:]))

plt.savefig('1.pdf')
