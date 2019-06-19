import numpy as np

###random seed fixed so that results are repeatable
np.random.seed(0)

### Model setup
nx = 40
F = 8.0
dt = 0.2

### Cycling experiment setup
nt = 100
cycle_period = 1
time_window = 1 ##smoother analysis window (+-cycles)
time_space_ratio = 1.0 ##ratio of dt/dx

### Observation network setup
obs_err = 1
L = 0  #spatial corr in R
Lt = 0 #temporal corr in R
##uniform network
obs_thin = 0.2
obs_ind = np.tile(np.arange(0, nx, obs_thin), (nt, 1)).T
##random network
# nobs = 30
# obs_ind = np.random.uniform(0, nx, size=(nobs, nt))

### Ensemble filter tuning parameters
nens = 20
ROI = 10  #localization in space (grid points)
ROIt = 2  #localization in time (cycles)
alpha = 0.0  ##relaxation coef
inflation = 1.0  ##multiplicative inflation
