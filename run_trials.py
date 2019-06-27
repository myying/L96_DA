#!/usr/bin/env python
import numpy as np
import L96_model as L96
import config as p
import data_assimilation as DA
import misc

# param2 = np.array([1.0, 1.05, 1.1, 1.2, 1.5, 2.0])
# param2 = np.arange(0.0, 1.0, 0.1)
param1 = np.array([10, 20, 40, 80, 160, 2000])
param2 = np.array([2, 5, 8, 10, 15, 20, 30, 50])

RMSEb = np.zeros((param1.size, param2.size))
RMSEa = np.zeros((param1.size, param2.size))
SPRDb = np.zeros((param1.size, param2.size))
SPRDa = np.zeros((param1.size, param2.size))

for i in range(param1.size):
  for j in range(param2.size):
    ##configuration
    nens = param1[i]
    ROI = param2[j]
    inflation = p.inflation
    F = p.F
    obs_err = p.obs_err
    L = p.L
    cp = p.cycle_period

    # read in initial data
    xt = np.load("output/truth.npy")
    obs = np.load("output/obs.npy")
    xens0 = np.load("output/initial_ensemble.npy")
    xens = np.zeros([p.nx, nens, p.nt])
    xens[:, :, 0] = xens0[:, 0:nens]
    xens1 = np.copy(xens)

    print(np.round((param1[i], param2[j]), 2))
    #run cycling filtering trial
    for tt in np.arange(p.nt-1):
      # analysis step
      if(np.mod(tt, cp) == 0) and tt>0:
        xb = xens1[:, :, tt].copy()
        yo = obs[:, tt]
        H = DA.H_matrix(p.nx, p.obs_ind, np.array([tt]), 0)
        R = DA.R_matrix(p.nx, p.obs_ind, np.array([tt]), obs_err, L, 0)
        rho = DA.local_matrix(p.nx, np.array([tt]), ROI, 0)
        xa = DA.EnKF(xb, yo, H, R, rho, tt)
        xa_mean = np.mean(xa, axis=1)
        for k in np.arange(nens):
          xens1[:, k, tt] = inflation*(xa[:, k]-xa_mean) + xa_mean
      # forecast step
      xens1[:, :, tt+1] = L96.M_nl(xens1[:, :, tt], p.nx, F, p.dt)
      xens[:, :, tt+1] = L96.M_nl(xens1[:, :, tt], p.nx, F, p.dt)

    #diagnostics
    Qb = misc.Q_out(xens[:, :, ::cp], xt[:, ::cp])
    Qa = misc.Q_out(xens1[:, :, ::cp], xt[:, ::cp])
    Pb = misc.P_out(xens[:, :, ::cp])
    Pa = misc.P_out(xens1[:, :, ::cp])
    RMSEb[i, j] = misc.rmse(Qb)
    RMSEa[i, j] = misc.rmse(Qa)
    SPRDb[i, j] = misc.sprd(Pb)
    SPRDa[i, j] = misc.sprd(Pa)

np.save("output/RMSEb", RMSEb)
np.save("output/RMSEa", RMSEa)
np.save("output/SPRDb", SPRDb)
np.save("output/SPRDa", SPRDa)

