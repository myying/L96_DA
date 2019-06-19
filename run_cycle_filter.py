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

##### assimilation cycle
for tt in range(p.nt-1):
  # analysis step
  if(np.mod(tt, p.cycle_period) == 0):
    xb = xens1[:, :, tt].copy()

    ######Define obs network
    t_ind = np.arange(tt, tt+1)
    H = DA.H_matrix(p.nx, p.obs_ind, t_ind)
    # print(H)
    R = DA.R_matrix(p.nx, p.obs_ind, t_ind, p.obs_err, p.L, 0)

    #####adaptive prior inflation
    # xens1[:, :, tt] = DA.adaptive_inflation(xens1[:, :, tt], p.obs_ind, yo[:, tt], p.obs_err)

    ######Filter kinds:
    ###full matrix version EnKF
    rho = DA.local_matrix(p.nx, t_ind, p.ROI, p.ROIt)
    xa = DA.EnKF(xb, yo[:, tt], H, R, rho)

    ###serial EnKF
    # D = DA.D_matrix(p.nx, p.obs_ind, t_ind, tt, p.time_space_ratio)
    # print(D)
    # xa = DA.EnKF_serial(xb, yo[:, tt], H, R, D, p.ROI)

    ##transform to spectral
    # U = misc.fourier_basis(p.nx)
    # Uxb = np.dot(U, xb)
    # Uxa = DA.EnKF_serial(Uxb, yo[:, tt], p.H, p.R, p.ROI)

    ##multiscale filter
# krange = np.arange(1, 10, 2)
# ns = krange.size + 1
# obs_err_ms = p.obs_err * np.ones(ns)
# ROI_ms = np.array([15, 12, 10, 8, 5, 3])
# ROI_ms = 15 * np.ones(ns)
# obs_err_ms = np.zeros(ns)
# R = DA.R_matrix(p.nx, 1, p.obs_ind, p.obs_err, 2, 0, 1)
# v = misc.fourier_basis(p.nx)
# wo = np.sqrt(np.diag(np.dot(v.T, np.dot(R, v)))[::2])
# for s in range(ns):
#   if s == 0:
#     mid = krange[s]/2
#   if s == ns-1:
#     mid = (krange[s-1] + wo.size)/2
#   if s > 0 and s < ns-1:
#     mid = (krange[s-1] + krange[s])/2
#   obs_err_ms[s] = misc.interp1d(wo, mid)
    # for s in range(ks):
    #   xa = DA.EnKF_serial(xb, p.obs_ind, yo[:, tt], obs_err_ms, ROI_ms, krange)

    #####posterior inflation
    xb_mean = np.mean(xb, axis=1)
    xa_mean = np.mean(xa, axis=1)
    for k in np.arange(p.nens):
      # xens1[:, k, tt] = (1-p.alpha)*(xa[:, k]-xa_mean) + p.alpha*(xb[:, k]-xb_mean) + xa_mean
      xens1[:, k, tt] = p.inflation*(xa[:, k]-xa_mean) + xa_mean

  # forecast step
  xens1[:, :, tt+1] = L96.M_nl(xens1[:, :, tt], p.nx, p.F, p.dt)
  xens[:, :, tt+1] = L96.M_nl(xens1[:, :, tt], p.nx, p.F, p.dt)

np.save("output/ensemble_prior", xens)
np.save("output/ensemble_post", xens1)

