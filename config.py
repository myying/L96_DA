import numpy as np
import data_assimilation as DA
nx = 40
F = 8.0
dt = 0.2
nt = 200
cycle_period = 1
time_window = 1 ##smoother analysis window (+-cycles)
obs_err = 1
L = 5  #spatial corr in R
Lt = 0 #temporal corr in R
corr_kind = 1  #1:AR(1) 2:AR(2)
obs_thin = 2
obs_ind = np.arange(0, nx, obs_thin)
H = DA.H_matrix(nx, time_window, obs_ind)
R = DA.R_matrix(nx, time_window, obs_ind, obs_err, L, Lt, corr_kind)
nens = 20
ROI = 10  #localization in space (grid points)
ROIt = 0  #localization in time (cycles)
rho = DA.local_matrix(nx, time_window, ROI, ROIt)
alpha = 0.0
inflation = 1.0
