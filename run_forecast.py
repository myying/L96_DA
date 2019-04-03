#!/usr/bin/env python
import numpy as np
import L96_model as L96
import config as p
import data_assimilation as DA
import misc

# read in initial data
xt = np.load("output/truth.npy")
yo = np.load("output/obs.npy")
xens0 = np.load("output/initial_ensemble.npy")

xens = np.zeros([p.nx, p.nens, p.nt+1])
xens[:, :, 0] = xens0[:, 0:p.nens]

for tt in np.arange(p.nt):
    for i in np.arange(p.nens):
        xens[:, i, tt+1] = L96.forward(xens[:, i, tt], p.nx, p.F, p.dt)

np.save("output/ensemble_forecast", xens)
