#!/usr/bin/env python
import numpy as np
import L96_model as L96
import config as p
import data_assimilation as DA
import misc

# read in initial data
xt = np.load("output/truth.npy")
yo = np.load("output/obs.npy")
xens0 = np.load("output/initial_ensemble.npy")

xens = np.zeros([p.nx, p.nens, p.nt+1])
xens[:, :, 0] = xens0[:, 0:p.nens]
xens1 = np.copy(xens)

# assimilation cycle
for tt in np.arange(p.nt):
  # analysis step
  if(np.mod(tt, p.cycle_period) == 0):
    xb = xens1[:, :, tt].copy()

    ##adaptive prior inflation
    # xens1[:, :, tt] = DA.adaptive_inflation(xens1[:, :, tt], p.obs_ind, yo[:, tt], p.obs_err)

    # xa = DA.EnKF(xb, p.obs_ind, yo[:, tt], p.obs_err, p.L, p.corr_kind, p.ROI)
    xa = DA.EnKF_serial(xb, p.obs_ind, yo[:, tt], p.obs_err, p.ROI, filter_kind=2)
    # xa = DA.EnSRF_spec(xb, p.obs_ind, yo[:, tt], p.obs_err, p.ROI)

    ##relaxation
    xb_mean = np.mean(xb, axis=1)
    xa_mean = np.mean(xa, axis=1)
    for k in np.arange(p.nens):
      xens1[:, k, tt] = (1-p.alpha)*(xa[:, k]-xa_mean) + p.alpha*(xb[:, k]-xb_mean) + xa_mean

  # forecast step
  xens1[:, :, tt+1] = L96.forward(xens1[:, :, tt], p.nx, p.F, p.dt)
  xens[:, :, tt+1] = L96.forward(xens1[:, :, tt], p.nx, p.F, p.dt)

np.save("output/ensemble_prior", xens)
np.save("output/ensemble_post", xens1)

