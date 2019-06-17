#!/usr/bin/env python
import numpy as np
import L96_model as L96
import config as p
import data_assimilation as DA
import misc

param1 = np.array([1, 2, 5, 8, 10, 15, 20, 30, 50])
param2 = np.array([1.0, 1.05, 1.1, 1.2, 1.5, 2.0])
RMSEb = np.zeros((param1.size, param2.size))
RMSEa = np.zeros((param1.size, param2.size))
SPRDb = np.zeros((param1.size, param2.size))
SPRDa = np.zeros((param1.size, param2.size))

for i in range(param1.size):
  for j in range(param2.size):
    # read in initial data
    xt = np.load("output/truth.npy")
    yo = np.load("output/obs.npy")
    xens0 = np.load("output/initial_ensemble.npy")
    xens = np.zeros([p.nx, p.nens, p.nt+1])
    xens[:, :, 0] = xens0[:, 0:p.nens]
    xens1 = np.copy(xens)
    cp = p.cycle_period

    print(param1[i], param2[j])
    rho = DA.local_matrix(p.nx, p.time_window, param1[i], p.ROIt)
    ###run trials
    for tt in np.arange(p.nt):
      # analysis step
      if(np.mod(tt, cp) == 0):
        xb = xens1[:, :, tt].copy()
        xa = DA.EnKF(xb, yo[:, tt], p.H, p.R, rho)
        ##inflation
        xa_mean = np.mean(xa, axis=1)
        for k in np.arange(p.nens):
          xens1[:, k, tt] = param2[j]*(xa[:, k]-xa_mean) + xa_mean
      # forecast step
      xens1[:, :, tt+1] = L96.forward(xens1[:, :, tt], p.nx, p.F, p.dt)
      xens[:, :, tt+1] = L96.forward(xens1[:, :, tt], p.nx, p.F, p.dt)

    #diagnostics
    RMSEb[i, j] = misc.rmse(xens[:, :, ::cp], xt[:, ::cp])
    RMSEa[i, j] = misc.rmse(xens1[:, :, ::cp], xt[:, ::cp])
    SPRDb[i, j] = misc.sprd(xens[:, :, ::cp])
    SPRDa[i, j] = misc.sprd(xens1[:, :, ::cp])

np.save("output/RMSEb", RMSEb)
np.save("output/RMSEa", RMSEa)
np.save("output/SPRDb", SPRDb)
np.save("output/SPRDa", SPRDa)

