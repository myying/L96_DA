import numpy as np

###random seed fixed so that results are repeatable
np.random.seed(0)

### Model setup
nx = 40
F = 8.0
dt = 0.2

### Cycling experiment setup
nt = 1000
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
nens = 2000
ROI = 0  #localization in space (grid points)
ROIt = 0  #localization in time (cycles)
alpha = 0.0  ##relaxation coef
inflation = 1.0  ##multiplicative inflation


### filter kind 1=EnKF w/ perturbed obs, 2=serial EnKF
filter_kind = 1
##multiscale
multiscale = True
krange = np.array([3, 7, 10, 15])
obs_err_inf = np.array([1.7, 0.6, 0.4, 0.35, 0.3])
# obs_err_inf = np.array([1.6, 0.8, 0.6, 0.55, 0.5])
