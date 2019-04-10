import numpy as np
nx = 40
F = 8.0
dt = 0.1
nt = 99
cycle_period = 1
time_window = 2 ##smoother analysis window (+-cycles)
obs_err = 1
L = 5
H = np.eye(nx)
R = np.eye(nx) * (obs_err ** 2)
obs_thin = 1
obs_ind = np.arange(0, nx, obs_thin)
nens = 200
ROI = 0  #localization in space (grid points)
ROIt = 0  #localization in time (cycles)
alpha = 0.0
