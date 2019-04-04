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
tw = p.time_window

# assimilation cycle
for tt in np.arange(p.nt):
  # analysis step
  if(np.mod(tt, p.cycle_period) == 0):
    t1 = max(0, tt-tw*p.cycle_period)
    t2 = min(p.nt, tt+tw*p.cycle_period)
    t_ind = np.arange(t1, t2+1, p.cycle_period)
    analysis_ind = int((tt - t1)/p.cycle_period)
    #run prior trajectory
    xens2 = np.zeros([p.nx, p.nens, p.nt+1])
    xens2[:, :, t1] = xens1[:, :, t1]
    for n in np.arange(t1, tt+1):
      xens2[:, :, n] = xens1[:, :, n]
    for n in np.arange(tt, t2):
      xens2[:, :, n+1] = L96.forward(xens2[:, :, n], p.nx, p.F, p.dt)
    #run smoother
    xens1[:, :, tt] = DA.EnKS_serial(xens2[:, :, t_ind], analysis_ind, p.obs_ind, yo[:, t_ind], oberr, p.ROI, p.ROIt, p.alpha)

  # forecast to next cycle
  xens1[:, :, tt+1] = L96.forward(xens1[:, :, tt], p.nx, p.F, p.dt)
  xens[:, :, tt+1] = L96.forward(xens1[:, :, tt], p.nx, p.F, p.dt)

np.save("output/ensemble_prior", xens)
np.save("output/ensemble_post", xens1)

