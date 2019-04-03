import numpy as np
nx = 40
F = 8.0
dt = 0.1
nt = 99
cycle_period = 4
obs_err = 1
H = np.eye(nx)
R = np.eye(nx) * (obs_err ** 2)
obs_thin = 1
obs_ind = np.arange(0, nx, obs_thin)
nens = 20
ROI = 5
alpha = 0.0
