#!/usr/bin/env python
import numpy as np
import L96_model as L96
import model_parameters as p
import data_assimilation as DA
import misc
import matplotlib.pyplot as plt
plt.switch_backend('Agg')


# read in initial data
xt = np.load("output/truth.npy")
yo = np.load("output/obs.npy")
xens0 = np.load("output/initial_ensemble.npy")

xens = np.zeros([p.nx, p.nens, p.nt+1])
xens[:, :, 0] = xens0[:, 0:p.nens]
xens1 = np.copy(xens)
xhens = np.zeros([int(p.nx/2+1), p.nens, p.nt+1], dtype=complex)

# assimilation cycle
for tt in np.arange(p.nt):

    # analysis
    if(np.mod(tt, p.cycle_period) == 0):
        # perform DA in state space
        xens1[:, :, tt] = DA.EnSRF(xens1[:, :, tt], p.obs_ind,
                                   yo[:, tt], p.obs_err, p.ROI)

        #convert to spectral space
        for k in np.arange(p.nens):
            xhens[:, k, tt] = misc.grid2spec(xens[:, k, tt])
        #perform DA in spectral space
        xhens[:, :, tt] = DA.EnSRF_spec(xhens[:, :, tt], p.obs_ind,
                                        yo[:, tt], p.obs_err, p.ROI)
        #convert back to state space
        for k in np.arange(p.nens):
            xens[:, k, tt] = misc.spec2grid(xhens[:, k, tt])

    # for k in np.arange(p.nens):
        # xhens[:, k, tt] = misc.grid2spec(xens[:, k, tt])

    # forecast
    for i in np.arange(p.nens):
        xens1[:, i, tt+1] = L96.M_nl(xens1[:, i, tt], p.nx, p.F, p.dt)
        xens[:, i, tt+1] = L96.M_nl(xens[:, i, tt], p.nx, p.F, p.dt)

np.save("output/ensemble_prior", xens)
# np.save("data_out_xhens", xhens)
np.save("output/ensemble_post", xens1)

#  plt.contourf(xt.T)
tt = 1000
ax = plt.subplot(211)
plt.plot(xens[:, :, tt], 'c')
plt.plot(np.mean(xens[:, :, tt], axis=1), 'b')
plt.plot(p.obs_ind, yo[p.obs_ind, tt], 'rx')
plt.plot(xt[:, tt], 'k')

ax = plt.subplot(212)
rmse = np.sqrt(np.mean((np.mean(xens[p.obs_ind, :, :], 1) -
                        xt[p.obs_ind, :]) ** 2, 0))
plt.plot(rmse, 'k')
plt.plot(np.arange(p.nt+1), np.ones(p.nt+1), color='0.7')
plt.xlabel('time')
plt.ylabel('rmse')

# for tt in np.arange(nt):
#     plt.figure()
#     plt.plot(np.real(fft(x[:,tt])),'k')
#     plt.axis([0, 20, -50, 125])
#     plt.plot(np.arange(0,nx,1), x)
#     plt.savefig("spec/{:05d}.png".format(tt))
#     plt.close()
plt.savefig("fig_analysis_p40_N40_spec.pdf")


xens = np.load("data_out_xens.npy")
xhens = np.load("data_out_xhens.npy")
tt = 0
x = xens[:, 0:p.nens, tt]
xh = xhens[:, 0:p.nens, tt]
for k in np.arange(p.nens):
    x[:, k] = x[:, k] - np.mean(x, 1)
    xh[:, k] = xh[:, k] - np.mean(xh, 1)
cov = np.matrix(x) * np.matrix(x).T / (p.nens-1)
covh = np.matrix(xh) * np.matrix(np.conj(xh)).T / (p.nens-1)
cov1 = np.matrix(xh) * np.matrix(x).T / (p.nens-1)
cov2 = np.matrix(x) * np.matrix(np.conj(xh)).T / (p.nens-1)
# for i in np.arange(p.nx):
#     cov[i, :] /= cov[i, i]
# for i in np.arange(p.nx/2+1):
#     covh[i, :] /= np.abs(covh[i, i])

