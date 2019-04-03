#!/usr/bin/env python3
import numpy as np
import L96_model as L96
import config as p

nx = p.nx
F = p.F
dt = p.dt
nt = p.nt
obs_err = p.obs_err
nens = p.nens 

xt = np.zeros((nx, nt+1))  # truth
yo = np.zeros((nx, nt+1))  # observations

# spin up to climatology
xt[:, 0] = np.random.normal(loc=0.0, scale=0.1, size=nx)
for tt in np.arange(1000):
  xt[:, 0] = L96.forward(xt[:, 0], nx, F, dt)

# nature run
for tt in np.arange(nt):
  # run model
  xt[:, tt+1] = L96.forward(xt[:, tt], nx, F, dt)

# generate observation
yo = xt + np.random.normal(loc=0.0, scale=obs_err, size=[nx, nt+1])

# initial ensemble
xens = np.zeros((nx, nens))
for i in np.arange(nens):
  xens[:, i] = xt[:, 0] + np.random.normal(loc=0.0, scale=0.1, size=[nx])
  # spin up the ensemble
  # xens[:, i] = L96.forward(xens[:, i], nx, F, 0.1)

# output to data file
np.save("output/truth", xt)
np.save("output/obs", yo)
np.save("output/initial_ensemble", xens)

