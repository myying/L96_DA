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

xens = np.zeros([p.nx, p.nens, p.nt])
xens[:, :, 0] = xens0[:, 0:p.nens]
xens1 = np.copy(xens)

### assimilation cycle
for tt in range(p.nt-1):
  ### analysis step
  if(np.mod(tt, p.cycle_period) == 0) and tt>0:
    t1 = max(0, tt-p.time_window)
    t2 = min(p.nt, tt+p.time_window)
    t_ind = np.arange(t1, t2+1)
    t_ana = int((tt - t1)/p.cycle_period)
    ###run prior trajectory
    xens2 = np.zeros([p.nx, p.nens, p.nt])
    xens2[:, :, t1] = xens1[:, :, t1]
    for n in np.arange(t1, tt+1):
      xens2[:, :, n] = xens1[:, :, n]
    for n in np.arange(tt, t2):
      xens2[:, :, n+1] = L96.M_nl(xens2[:, :, n], p.nx, p.F, p.dt)
    ###run smoother
    xb = np.zeros((p.nx*t_ind.size, p.nens))
    for k in range(p.nens):
      xb[:, k] = np.reshape(xens2[:, k, t_ind].T, p.nx*t_ind.size)
    obs = np.reshape(yo[:, t_ind].T, yo[:, t_ind].size)
    H = DA.H_matrix(p.nx, p.obs_ind, t_ind)
    R = DA.R_matrix(p.nx, p.obs_ind, t_ind, p.obs_err, p.L, p.Lt)
    rho = DA.local_matrix(p.nx, t_ind, p.ROI, p.ROIt)
    xa = DA.EnKF(xb, obs, H, R, rho, tt)

    #####posterior inflation
    xb_mean = np.mean(xb[t_ana*p.nx:(t_ana+1)*p.nx, :], axis=1)
    xa_mean = np.mean(xa[t_ana*p.nx:(t_ana+1)*p.nx, :], axis=1)
    for k in np.arange(p.nens):
      # xens1[:, k, tt] = (1-p.alpha)*(xa[:, k]-xa_mean) + p.alpha*(xb[:, k]-xb_mean) + xa_mean
      xens1[:, k, tt] = p.inflation*(xa[t_ana*p.nx:(t_ana+1)*p.nx, k]-xa_mean) + xa_mean

  ### forecast to next cycle
  xens1[:, :, tt+1] = L96.M_nl(xens1[:, :, tt], p.nx, p.F, p.dt)
  xens[:, :, tt+1] = L96.M_nl(xens1[:, :, tt], p.nx, p.F, p.dt)

np.save("output/ensemble_prior", xens)
np.save("output/ensemble_post", xens1)

