import numpy as np
import misc
import data_assimilation as DA

###random seed fixed so that results are repeatable
np.random.seed(0)

### Model setup
nx = 40
F = 8.0
dt = 0.2

### Cycling experiment setup
nt = 5500
cycle_period = 1
time_window = 0 ##smoother analysis window (+-cycles)
time_space_ratio = 1.0 ##ratio of dt/dx

### Observation network setup
obs_err = 1.0
L = 0  #spatial corr in R
Lt = 0 #temporal corr in R
##type of network:
##1. uniform
obs_thin = 1
obs_ind = np.tile(np.arange(0, nx, obs_thin), (nt, 1)).T
##2. random
# nobs = 40
# obs_ind = np.random.uniform(0, nx, size=(nobs, nt))

### Ensemble filter tuning parameters
nens = 20
ROI = 10  #localization in space (grid points)
ROIt = 0  #localization in time (cycles)
alpha = 0.0  ##relaxation coef
inflation = 1.2  ##multiplicative inflation


### filter kind 1=EnKF w/ perturbed obs, 2=serial EnKF
filter_kind = 2
multiscale = True
dk = 4
krange = np.arange(dk, 20, dk)
R = DA.R_matrix(nx, obs_ind, np.array([0]), 1, 5, 0)
Lo = misc.matrix_spec(R)
obs_err_inf = np.ones(krange.size+1)
obs_err_inf[0] = np.sqrt(np.mean(Lo[0:krange[0]+1]**2))
obs_err_inf[krange.size] = np.sqrt(np.mean(Lo[krange[-1]+1:]**2))
for i in range(1, krange.size):
  obs_err_inf[i] = np.sqrt(np.mean(Lo[krange[i-1]+1:krange[i]+1]**2))
ROI_adjust = np.array([3, 1, 1, 0.8, 0.8])
