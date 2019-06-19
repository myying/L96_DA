#!/usr/bin/env python
import numpy as np
import config as p
import data_assimilation as DA
import matplotlib.pyplot as plt
plt.switch_backend('Agg')

xt = np.load("output/truth.npy")
yo = np.load("output/obs.npy")
xens = np.load("output/ensemble_prior.npy")
xens1 = np.load("output/ensemble_post.npy")

nx, nens, nt = xens.shape
tt = 90
ax = plt.subplot(211)
ax.plot(xens1[:, 0:100, tt], 'c')
ax.plot(np.mean(xens1[:, :, tt], axis=1), 'b')
if(np.mod(tt, p.cycle_period) == 0):
  ax.plot(p.obs_ind[:, tt], yo[:, tt], 'rx')
ax.plot(xt[:, tt], 'k')
ax.set_ylim(-10, 15)

ax = plt.subplot(212)
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
ax.plot(time[:tt*2], rmse_out[:tt*2], 'k', label='error')
ax.plot(time[:tt*2], sprd_out[:tt*2], 'g', label='spread')
ax.plot(np.arange(nt), p.obs_err*np.ones(nt), color='0.7')
ax.set_xlabel('time')
ax.set_xticks(np.arange(0, 220, 20))
ax.set_xticklabels(np.round(np.arange(0, 220, 20)*0.02, 2))
ax.set_xlim(0, 100)
ax.set_ylim(0, 5)
ax.legend(fontsize=12, loc=1)

# xa = np.mean(xens, axis=1)
# ax = plt.subplot(121)
# ax.contourf(xa.T, np.arange(-10, 15, 1))
# ax = plt.subplot(122)
# ax.contourf((xa-xt).T, np.arange(-10, 15, 1))

# print(np.sqrt(np.mean(rmse[::p.cycle_period]**2)))
# print(np.sqrt(np.mean(rmse1[::p.cycle_period]**2)))
# err_reduc = rmse1[::p.cycle_period]/rmse[::p.cycle_period]
# print(np.mean(err_reduc[3:]))

plt.savefig('1.pdf')
