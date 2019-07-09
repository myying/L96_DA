#!/usr/bin/env python
import numpy as np
import L96_model as L96
import config as p
import data_assimilation as DA
import misc

##ROI sensitivity to ensemble size N
# param1 = np.array([10, 20, 40, 80, 160, 320, 640, 1280])
# param2 = np.array([2, 5, 8, 10, 15, 20, 30, 50])
##inflation sensitivity to model error in F
# param1 = np.array([6, 7, 7.5, 7.9, 8])
# param2 = np.array([0.8, 1.0, 1.05, 1.1, 1.3, 1.5, 2.0])
##obs_err and L
# param1 = np.array([0.6, 0.8, 1.0, 1.2, 1.5])
# param2 = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
param1 = np.arange(2, 50, 2)
param2 = np.arange(1.0, 1.2, 0.01)
# param2 = np.array([1.2])

RMSEb = np.zeros((param1.size, param2.size))
RMSEa = np.zeros((param1.size, param2.size))
SPRDb = np.zeros((param1.size, param2.size))
SPRDa = np.zeros((param1.size, param2.size))

for i in range(param1.size):
  for j in range(param2.size):
    ##configuration
    nens = 20 #param1[i]
    ROI = param1[i]
    inflation = param2[j]
    F = 8 #param1[i]
    obs_err = 1.0 #param1[i]
    L = 5 #param2[j]
    cp = 1

    # read in initial data
    xt = np.load("output/truth.npy")
    obs = np.load("output/obs.npy")
    xens0 = np.load("output/initial_ensemble.npy")
    xens = np.zeros([p.nx, nens, p.nt])
    xens[:, :, 0] = xens0[:, 0:nens]
    xens1 = np.copy(xens)

    #run cycling filtering trial
    for tt in np.arange(p.nt-1):
      # analysis step
      if(np.mod(tt, cp) == 0) and tt>0:
        x = xens1[:, :, tt].copy()
        yo = obs[:, tt]
        H = DA.H_matrix(p.nx, p.obs_ind, np.array([tt]), 0)
        R = DA.R_matrix(p.nx, p.obs_ind, np.array([tt]), obs_err, L, 0)
        rho = DA.local_matrix(p.nx, np.array([tt]), ROI, 0)
        if p.multiscale:
          xa = x.copy()
          x_s = np.zeros((p.nx, nens))
          for s in range(p.krange.size+1):
            for k in range(nens):
              x_s[:, k] = misc.spec_bandpass(xa[:, k], p.krange, s)
            yo_s = misc.spec_bandpass(yo, p.krange, s)
            obs_err = p.obs_err_inf[s]
            ###opt 1: decompose state
            # xa_s = DA.EnKF_serial(x_s, xa, yo, obs_err, p.obs_ind[:, tt], ROI)
            # xa += xa_s - x_s
            ###opt 2: decompose obs
            xa = DA.EnKF_serial(xa, x_s, yo_s, obs_err, p.obs_ind[:, tt], ROI)
        else:
          # xa = DA.EnKF(x, yo, H, R, rho)
          xa = DA.EnKF_perturbed_obs(x, yo, H, R, rho, tt)
          # xa = DA.EnKF_serial(x, x, yo, obs_err, p.obs_ind[:, tt], ROI)
        xa_mean = np.mean(xa, axis=1)
        for k in np.arange(nens):
          xens1[:, k, tt] = inflation*(xa[:, k]-xa_mean) + xa_mean
      # forecast step
      xens1[:, :, tt+1] = L96.M_nl(xens1[:, :, tt], p.nx, F, p.dt)
      xens[:, :, tt+1] = L96.M_nl(xens1[:, :, tt], p.nx, F, p.dt)

    #diagnostics
    t1 = 500
    Qb = misc.Q_out(xens[:, :, t1::cp], xt[:, t1::cp])
    Qa = misc.Q_out(xens1[:, :, t1::cp], xt[:, t1::cp])
    Pb = misc.P_out(xens[:, :, t1::cp])
    Pa = misc.P_out(xens1[:, :, t1::cp])
    RMSEb[i, j] = misc.rmse(Qb)
    RMSEa[i, j] = misc.rmse(Qa)
    SPRDb[i, j] = misc.sprd(Pb)
    SPRDa[i, j] = misc.sprd(Pa)
    print(param1[i], param2[j])
    print('RMSEa=', RMSEa[i, j], '  CRa=', SPRDa[i, j]/RMSEa[i, j])

outdir = "L5_N20_F8"
np.save(outdir+"/RMSEb", RMSEb)
np.save(outdir+"/RMSEa", RMSEa)
np.save(outdir+"/SPRDb", SPRDb)
np.save(outdir+"/SPRDa", SPRDa)
