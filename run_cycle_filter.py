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

oberr = p.obs_err

# assimilation cycle
for tt in np.arange(p.nt):

    # analysis
    if(np.mod(tt, p.cycle_period) == 0):
        xens1[:, :, tt] = DA.EnKF(xens1[:, :, tt], p.obs_ind, yo[:, tt], oberr, p.ROI, p.alpha)
        # xens1[:, :, tt] = DA.EnKF_serial(xens1[:, :, tt], p.obs_ind, yo[:, tt], oberr, p.ROI, p.alpha, 2)

    # forecast
    for i in np.arange(p.nens):
        xens1[:, i, tt+1] = L96.forward(xens1[:, i, tt], p.nx, p.F, p.dt)
        xens[:, i, tt+1] = L96.forward(xens1[:, i, tt], p.nx, p.F, p.dt)

np.save("output/ensemble_prior", xens)
np.save("output/ensemble_post", xens1)

