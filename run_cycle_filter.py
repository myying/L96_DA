#!/usr/bin/env python
import numpy as np
import L96_model as L96
import data_assimilation as DA
import misc
import sys

###CONFIG
###random seed fixed so that results are repeatable
np.random.seed(0)

### Model setup
nx = 40
dt = 0.2
F = np.round(float(sys.argv[5]), 2) #8.0

### Cycling experiment setup
nt = 5500
cycle_period = 1
time_window = 0 ##smoother analysis window (+-cycles)
time_space_ratio = 1.0 ##ratio of dt/dx

### Observation network setup
obs_err = np.round(float(sys.argv[3]), 2) #1.0
L = np.round(float(sys.argv[2]), 2)  #spatial corr in R
obs_thin = 1
obs_ind = np.tile(np.arange(0, nx, obs_thin), (nt, 1)).T

### Ensemble filter tuning parameters
filter_kind = sys.argv[1]
nens = int(sys.argv[4])
ROI = int(sys.argv[6])  #localization in space (grid points)
alpha = np.round(float(sys.argv[7]), 2) #0.0  ##relaxation coef
inflation = 1.0 #np.round(float(sys.argv[7]), 2)  ##multiplicative inflation

#for MS algorithms
dk = int(sys.argv[8])
krange = np.arange(dk, 20, dk)
R = DA.R_matrix(nx, obs_ind, np.array([0]), 1, 5, 0)
Lo = misc.matrix_spec(R)
obs_err_inf = np.ones(krange.size+1)
obs_err_inf[0] = np.sqrt(np.mean(Lo[0:krange[0]+1]**2))
obs_err_inf[krange.size] = np.sqrt(np.mean(Lo[krange[-1]+1:]**2))
for i in range(1, krange.size):
  obs_err_inf[i] = np.sqrt(np.mean(Lo[krange[i-1]+1:krange[i]+1]**2))
ROI_adjust = np.ones(krange.size+1)

casename = filter_kind+"_dk{}".format(dk)+"/L{}_s{}".format(L, obs_err)+"/N{}_F{}".format(nens, F)+"/ROI{}".format(ROI)+"_relax{:4.2f}".format(alpha)
print(casename)

# read in initial data
xt = np.load("output/truth.npy")
obs = np.load("output/obs.npy")
xens0 = np.load("output/initial_ensemble.npy")

xens = np.zeros([nx, nens, nt])
xens[:, :, 0] = xens0[:, 0:nens]
xens1 = np.copy(xens)

##### assimilation cycle
for tt in range(nt-1):
  # analysis step
  if(np.mod(tt, cycle_period) == 0) and tt>0:
    xb = xens1[:, :, tt].copy()
    xa = xb

    ######Define obs network
    t1 = max(0, tt-time_window)
    t2 = min(nt, tt+time_window)
    t_ind = np.arange(t1, t2+1)
    # print(t_ind)

    x = xb.copy()
    dx = np.zeros(x.shape)
    ##assimilate obs from within window
    for t in t_ind:
      # print(t)
      yo = obs[:, t]
      H = DA.H_matrix(nx, obs_ind, np.array([t]), 0)
      R = DA.R_matrix(nx, obs_ind, np.array([t]), obs_err, L, 0)
      rho = DA.local_matrix(nx, np.array([tt]), ROI, 0)
      #####EnKF
      if filter_kind == "EnKF":
        # x1 = DA.EnKF(x, yo, H, R, rho)
        x1 = DA.EnKF_perturbed_obs(x, yo, H, R, rho, tt)
      #####serial EnKF
      if filter_kind == "EnSRF":
        x1 = DA.EnKF_serial(x, x, yo, obs_err, obs_ind[:, t], ROI)
      dx += x1 - x
      if filter_kind == "MS_State":
        x1 = x.copy()
        x_s = np.zeros((nx, nens))
        for s in range(krange.size+1):
          for k in range(nens):
            x_s[:, k] = misc.spec_bandpass(x1[:, k], krange, s)
          x1_s = DA.EnKF_serial(x_s, x1, yo, obs_err*obs_err_inf[s], obs_ind[:, t], ROI*ROI_adjust[s])
          x1 += x1_s -x_s
        dx = x1 - x
      if filter_kind == "MS_Obs":
        x1 = x.copy()
        x_s = np.zeros((nx, nens))
        for s in range(krange.size+1):
          for k in range(nens):
            x_s[:, k] = misc.spec_bandpass(x1[:, k], krange, s)
          yo_s = misc.spec_bandpass(yo, krange, s)
          x1 = DA.EnKF_serial(x1, x_s, yo_s, obs_err*obs_err_inf[s], obs_ind[:, t], ROI)
        dx = x1 - x
    xa = xb + dx

    #####posterior inflation
    xb_mean = np.mean(xb, axis=1)
    xa_mean = np.mean(xa, axis=1)
    for k in np.arange(nens):
      xens1[:, k, tt] = (1-alpha)*(xa[:, k]-xa_mean) + alpha*(xb[:, k]-xb_mean) + xa_mean
      # xens1[:, k, tt] = inflation*(xa[:, k]-xa_mean) + xa_mean

  # forecast step
  xens1[:, :, tt+1] = L96.M_nl(xens1[:, :, tt], nx, F, dt)
  xens[:, :, tt+1] = L96.M_nl(xens1[:, :, tt], nx, F, dt)

outdir = "/glade/scratch/mying/L96_DA"
np.save(outdir+"/"+casename+"/ensemble_prior", xens)
np.save(outdir+"/"+casename+"/ensemble_post", xens1)

