#!/usr/bin/env python
import numpy as np
import L96_model as L96
import config as p
import data_assimilation as DA

np.random.seed(0)

nx = p.nx
F = p.F
dt = p.dt
nt = p.nt
nens = p.nens
nobs = p.H.shape[0]

xt = np.zeros((nx, nt+1))  # truth
yo = np.zeros((nobs, nt+1))  # observations

# spin up to climatology
xt[:, 0] = np.random.normal(loc=0.0, scale=0.1, size=nx)
for tt in np.arange(1000):
  xt[:, 0] = L96.forward(xt[:, 0], nx, F, dt)

# nature run
for tt in np.arange(nt):
  # run model
  xt[:, tt+1] = L96.forward(xt[:, tt], nx, F, dt)

## generate observation
##true R matrix with correlation scale L and variance obs_err**2
for t in np.arange(nt):
  yo[:, t] = np.dot(p.H, xt[:, t]) + np.random.multivariate_normal(np.zeros(nobs), p.R)
# err = np.random.multivariate_normal(np.zeros(nx*nt), R)
# for t in range(nt):
#   yo[:, t] = xt[:, t] + err[t*nx:(t+1)*nx]

# initial ensemble
xmean = xt[:, 0] + np.random.normal(0.0, 1.0, size=nx)
xens = np.zeros((nx, nens))
for i in np.arange(nens):
  xens[:, i] = xmean + np.random.normal(0.0, 1.0, size=[nx])
# spin up the ensemble
# xens = L96.forward(xens, p.nx, p.F, 0.5)

# output to data file
np.save("output/truth", xt)
np.save("output/obs", yo)
np.save("output/initial_ensemble", xens)

