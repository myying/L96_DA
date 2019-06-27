#!/usr/bin/env python
import numpy as np
import L96_model as L96
import config as p
import data_assimilation as DA

np.random.seed(0)

nx = p.nx
F = p.F
dt = p.dt
nens = p.nens
nobs, nt = p.obs_ind.shape

xt = np.zeros((nx, nt))  # truth
yo = np.zeros((nobs, nt))  # observations

# spin up to climatology
xt[:, 0] = np.random.normal(loc=0.0, scale=0.1, size=nx)
for tt in np.arange(1000):
  xt[:, 0] = L96.M_nl(xt[:, 0], nx, F, dt)

# nature run
for tt in np.arange(nt-1):
  # run model
  xt[:, tt+1] = L96.M_nl(xt[:, tt], nx, F, dt)

## generate observation
##true R matrix with correlation scale L and variance obs_err**2
####time uncorrelated obs error
for t in range(nt):
  H = DA.H_matrix(p.nx, p.obs_ind, np.arange(t, t+1), 0)
  R = DA.R_matrix(p.nx, p.obs_ind, np.arange(t, t+1), p.obs_err, p.L, 0)
  yo[:, t] = np.dot(H, xt[:, t]) + np.random.multivariate_normal(np.zeros(nobs), R)
  # print(yo[:, t])
####time correlated obs error
# H = DA.H_matrix(p.nx, p.obs_ind, np.arange(nt), 0)
# R = DA.R_matrix(p.nx, p.obs_ind, np.arange(nt), p.obs_err, p.L, p.Lt)
# yo_err = np.random.multivariate_normal(np.zeros(nobs*nt), R)
# yo = np.reshape(np.dot(H, np.reshape(xt.T, xt.size)) + yo_err, (nobs, nt))

# initial ensemble
error_mean = 0.0
error_magnitude = 1.0  ##set to 1.0, matches obs error
xmean = xt[:, 0] + np.random.normal(error_mean, error_magnitude, size=nx)
xens = np.zeros((nx, nens))
for i in np.arange(nens):
  xens[:, i] = xmean + np.random.normal(error_mean, error_magnitude, size=[nx])
# spin up the ensemble
# xens = L96.M_nl(xens, p.nx, p.F, 0.5)

# output to data file
np.save("output/truth", xt)
np.save("output/obs", yo)
np.save("output/initial_ensemble", xens)

