#!/usr/bin/env python
import numpy as np
import L96_model as L96
import config as p
import data_assimilation as DA
import misc

# read in initial data
xt = np.load("output/truth.npy")
obs = np.load("output/obs.npy")
xens0 = np.load("output/initial_ensemble.npy")

xens = np.zeros([p.nx, p.nens, p.nt])
xens[:, :, 0] = xens0[:, 0:p.nens]
xens1 = np.copy(xens)

##### assimilation cycle
for tt in range(p.nt-1):
  # analysis step
  if(np.mod(tt, p.cycle_period) == 0) and tt>0:
    xb = xens1[:, :, tt].copy()
    xa = xb

    ######Define obs network
    t1 = max(0, tt-p.time_window)
    t2 = min(p.nt, tt+p.time_window)
    t_ind = np.arange(t1, t2+1)
    # print(t_ind)

    x = xb.copy()
    dx = np.zeros(x.shape)
    ##assimilate obs from within window
    for t in t_ind:
      # print(t)
      yo = obs[:, t]
      H = DA.H_matrix(p.nx, p.obs_ind, np.array([t]), 0)
      R = DA.R_matrix(p.nx, p.obs_ind, np.array([t]), p.obs_err, p.L, 0)
      rho = DA.local_matrix(p.nx, np.array([tt]), p.ROI, 0)
      #####EnKF
      if p.filter_kind == 1:
        if p.multiscale:
          for s in range(p.krange.size+1):
            x1 = DA.EnKF(x, yo, H, R*p.obs_err_inf[s], rho)
            for k in range(p.nens):
              dx[:, k] += misc.spec_bandpass(x1[:, k]-x[:, k], p.krange, s)
        else:
          x1 = DA.EnKF(x, yo, H, R, rho)
          # x1 = DA.EnKF_perturbed_obs(x, yo, H, R, rho, tt)
        dx = x1 - x
      #####serial EnKF
      if p.filter_kind == 2:
        if p.multiscale:
          x1 = x.copy()
          x_s = np.zeros((p.nx, p.nens))
          for s in range(p.krange.size+1):
            for k in range(p.nens):
              x_s[:, k] = misc.spec_bandpass(x1[:, k], p.krange, s)
            yo_s = misc.spec_bandpass(yo, p.krange, s)
            ###opt 1: decompose state
            x1_s = DA.EnKF_serial(x_s, x1, yo, p.obs_err*p.obs_err_inf[s], p.obs_ind[:, t], p.ROI*p.ROI_adjust[s])
            x1 += x1_s - x_s
          ###opt 2: decompose obs
          #  x1 = DA.EnKF_serial(x1, x_s, yo_s, p.obs_err*p.obs_err_inf[s], p.obs_ind[:, t], p.ROI)
          dx = x1 - x
        else:
          x1 = DA.EnKF_serial(x, x, yo, p.obs_err, p.obs_ind[:, t], p.ROI)
          dx = x1 - x
    xa = xb + dx

    #####posterior inflation
    xb_mean = np.mean(xb, axis=1)
    xa_mean = np.mean(xa, axis=1)
    for k in np.arange(p.nens):
      xens1[:, k, tt] = (1-p.alpha)*(xa[:, k]-xa_mean) + p.alpha*(xb[:, k]-xb_mean) + xa_mean
      #xens1[:, k, tt] = p.inflation*(xa[:, k]-xa_mean) + xa_mean

  # forecast step
  xens1[:, :, tt+1] = L96.M_nl(xens1[:, :, tt], p.nx, p.F, p.dt)
  xens[:, :, tt+1] = L96.M_nl(xens1[:, :, tt], p.nx, p.F, p.dt)

np.save("output1/ensemble_prior", xens)
np.save("output1/ensemble_post", xens1)


