#!/usr/bin/env python
import numpy as np
import L96_model as L96
import config as p
import data_assimilation as DA
import misc

# params = np.array([0.1, 0.2, 0.5, 0.8, 1.0, 1.2, 1.5, 1.8, 2.0, 3.0, 5.0])
params = np.array([0, 0.2, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0])
RMSEb = np.zeros(params.size)
RMSEa = np.zeros(params.size)
SPRDb = np.zeros(params.size)
SPRDa = np.zeros(params.size)

for i in range(params.size):
  # read in initial data
  xt = np.load("output/truth.npy")
  yo = np.load("output/obs.npy")
  xens0 = np.load("output/initial_ensemble.npy")
  xens = np.zeros([p.nx, p.nens, p.nt+1])
  xens[:, :, 0] = xens0[:, 0:p.nens]
  xens1 = np.copy(xens)
  cp = p.cycle_period

  print(params[i])
  R = DA.R_matrix(p.nx, p.time_window, p.obs_ind, p.obs_err, params[i], p.Lt, p.corr_kind)
  ####run trial
  for tt in np.arange(p.nt):
    # analysis step
    if(np.mod(tt, cp) == 0):
      xb = xens1[:, :, tt].copy()
      xa = DA.EnKF(xb, yo[:, tt], p.H, R, p.rho)
      ##relaxation
      xb_mean = np.mean(xb, axis=1)
      xa_mean = np.mean(xa, axis=1)
      for k in np.arange(p.nens):
        xens1[:, k, tt] = (1-p.alpha)*(xa[:, k]-xa_mean) + p.alpha*(xb[:, k]-xb_mean) + xa_mean
    # forecast step
    xens1[:, :, tt+1] = L96.M_nl(xens1[:, :, tt], p.nx, p.F, p.dt)
    xens[:, :, tt+1] = L96.M_nl(xens1[:, :, tt], p.nx, p.F, p.dt)

  #diagnostics
  RMSEb[i] = misc.rmse(xens[:, :, ::cp], xt[:, ::cp])
  RMSEa[i] = misc.rmse(xens1[:, :, ::cp], xt[:, ::cp])
  SPRDb[i] = misc.sprd(xens[:, :, ::cp])
  SPRDa[i] = misc.sprd(xens1[:, :, ::cp])


np.save("output/RMSEb", RMSEb)
np.save("output/RMSEa", RMSEa)
np.save("output/SPRDb", SPRDb)
np.save("output/SPRDa", SPRDa)

