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

ROI = np.array([1, 2, 5, 8, 10, 15, 20, 30, 50])
alpha = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
RMSEb = np.zeros((ROI.size, alpha.size))
RMSEa = np.zeros((ROI.size, alpha.size))
SPRDb = np.zeros((ROI.size, alpha.size))
SPRDa = np.zeros((ROI.size, alpha.size))

for i in range(ROI.size):
  for j in range(alpha.size):
    print(ROI[i], alpha[j])
# assimilation cycle
    for tt in np.arange(p.nt):
      # analysis step
      if(np.mod(tt, p.cycle_period) == 0):
        xb = xens1[:, :, tt].copy()
        xa = DA.EnKF(xb, p.obs_ind, yo[:, tt], p.obs_err, p.L, p.corr_kind, ROI[i])
        ##relaxation
        xb_mean = np.mean(xb, axis=1)
        xa_mean = np.mean(xa, axis=1)
        for k in np.arange(p.nens):
          xens1[:, k, tt] = (1-alpha[j])*(xa[:, k]-xa_mean) + alpha[j]*(xb[:, k]-xb_mean) + xa_mean
      # forecast step
      xens1[:, :, tt+1] = L96.forward(xens1[:, :, tt], p.nx, p.F, p.dt)
      xens[:, :, tt+1] = L96.forward(xens1[:, :, tt], p.nx, p.F, p.dt)

#diagnostics
    Pb = np.zeros((p.nx, p.nx))
    Pa = np.zeros((p.nx, p.nx))
    for t in range(0, p.nt, p.cycle_period):
      Pb += misc.error_covariance(xens[:, :, t])
      Pa += misc.error_covariance(xens1[:, :, t])
    Pb = Pb/(p.nt/p.cycle_period)
    Pa = Pa/(p.nt/p.cycle_period)
    eb = np.mean(xens, axis=1) - xt
    ea = np.mean(xens1, axis=1) - xt
    Qb = np.dot(eb[:, ::p.cycle_period], eb[:, ::p.cycle_period].T)/(p.nt/p.cycle_period)
    Qa = np.dot(ea[:, ::p.cycle_period], ea[:, ::p.cycle_period].T)/(p.nt/p.cycle_period)
    RMSEb[i, j] = np.sqrt(np.mean(np.diag(Qb)))
    RMSEa[i, j] = np.sqrt(np.mean(np.diag(Qa)))
    SPRDb[i, j] = np.sqrt(np.mean(np.diag(Pb)))
    SPRDa[i, j] = np.sqrt(np.mean(np.diag(Pa)))

np.save("output/RMSEb", RMSEb)
np.save("output/RMSEa", RMSEa)
np.save("output/SPRDb", SPRDb)
np.save("output/SPRDa", SPRDa)

