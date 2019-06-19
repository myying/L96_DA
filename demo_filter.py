#!/usr/bin/env python
import numpy as np
import config as p
import matplotlib.pyplot as plt
import data_assimilation as DA
import sys
plt.switch_backend('Agg')

xt = np.load("output/truth.npy")
yo = np.load("output/obs.npy")
xens = np.load("output/ensemble_forecast.npy")

nx, nens, nt = xens.shape
tt = 30
i = int(sys.argv[1])
obs_ind = np.arange(0, i, 1)

# xa = DA.EnKF_serial(xens[:, :, tt], np.array([obs_ind]), yo[:, tt], p.obs_err, p.ROI, filter_kind=2)
xa = DA.EnKF_serial(xens[:, :, tt], obs_ind, yo[:, tt], p.obs_err, p.ROI, filter_kind=2)

ax = plt.subplot(211)
##covariance
# xb = xens[:, :, tt]
# xp = xb.copy()
# for k in range(nens):
#   xp[:, k] = xb[:, k] - np.mean(xb, axis=1)
# cov = np.zeros(p.nx)
# for i in range(p.nx):
#   cov[i] = np.sum( xp[i, :] * xp[obs_ind, :]) / (nens-1)
# var1 = np.sum( xp**2, axis=1) /(nens-1)
# var2 = np.sum( xp[obs_ind, :]**2) /(nens-1)
# ax.plot(cov/np.sqrt(var1*var2))

##plot ensemble
# ax.plot(xens[:, 0:100, tt], 'c')
# ax.plot(np.mean(xens[:, :, tt], axis=1), 'b')
ax.plot(xa[:, 0:100], 'c')
ax.plot(np.mean(xa[:, :], axis=1), 'b')
ax.plot(xt[:, tt], 'k')
ax.plot(obs_ind, yo[obs_ind, tt], 'rx')
ax.set_ylim(-10, 15)

###plot distribution at observation location
# ax = plt.subplot(212)
# v = np.arange(-20, 20, 0.1)
# prior_mean = np.mean(xens[obs_ind, :, tt])
# prior_var = np.sum( (xens[obs_ind, :, tt] - prior_mean)**2 ) / (nens-1)
# ax.plot(xens[obs_ind, 0:20, tt], 0.1*np.ones(20), 'co')
# # ax.plot(prior_mean*np.ones(2), [0, 1], 'b')
# dist = np.exp(-0.5*(v-prior_mean)**2 / prior_var) / np.sqrt(2*np.pi*prior_var)
# ax.plot(v, dist, 'c')

# # ax.plot(yo[obs_ind, tt]*np.ones(2), [0, 1], 'r')
# dist = np.exp(-0.5*(v-yo[obs_ind, tt])**2 / (p.obs_err**2)) / np.sqrt(2*np.pi) / p.obs_err
# ax.plot(v, dist, 'r')

# post_mean = np.mean(xa[obs_ind, :])
# post_var = np.sum( (xa[obs_ind, :] - post_mean)**2 ) / (nens-1)
# ax.plot(xa[obs_ind, 0:20], 0.2*np.ones(20), 'go')
# # ax.plot(post_mean*np.ones(2), [0, 1], 'g')
# for i in range(20):
#   ax.plot([xens[obs_ind, i, tt], xa[obs_ind, i]], [0.1, 0.2], 'k', linewidth=0.5)
# dist = np.exp(-0.5*(v-post_mean)**2 / post_var) / np.sqrt(2*np.pi*post_var)
# ax.plot(v, dist, 'g')

# ax.set_xlim(-3, 6)
# ax.set_ylim(0, 0.6)

ind = obs_ind+2
###plot distribution at other location
# ax = plt.subplot(121)
# v = np.arange(-20, 20, 0.1)
# prior_mean = np.mean(xens[ind, :, tt])
# prior_var = np.sum( (xens[ind, :, tt] - prior_mean)**2 ) / (nens-1)
# ax.plot(0.1*np.ones(20), xens[ind, 0:20, tt], 'co')
# dist = np.exp(-0.5*(v-prior_mean)**2 / prior_var) / np.sqrt(2*np.pi*prior_var)
# ax.plot(dist, v, 'c')
# post_mean = np.mean(xa[ind, :])
# post_var = np.sum( (xa[ind, :] - post_mean)**2 ) / (nens-1)
# ax.plot(0.15*np.ones(20), xa[ind, 0:20], 'go')
# for i in range(20):
#   ax.plot([0.1, 0.15], [xens[ind, i, tt], xa[ind, i]], 'k', linewidth=0.5)
# dist = np.exp(-0.5*(v-post_mean)**2 / post_var) / np.sqrt(2*np.pi*post_var)
# ax.plot(dist, v, 'g')
# ax.set_ylim(-5, 6)
# ax.set_xlim(0, 0.5)

###plot joint distribution
# ax = plt.subplot(111)
# ax.plot(xens[obs_ind, 0:20, tt], xens[ind, 0:20, tt], 'co')
# ax.plot(xa[obs_ind, 0:20], xa[ind, 0:20], 'go')
# for i in range(20):
  # ax.plot([xens[obs_ind, i, tt], xa[obs_ind, i]], [xens[ind, i, tt], xa[ind, i]], 'k', linewidth=0.5)
# ax.set_xlim(-3, 6)
# ax.set_ylim(-5, 6)

# plt.savefig('1.pdf')
plt.savefig('{:03d}.png'.format(i), dpi=200)
