#!/usr/bin/env python
import numpy as np
import misc
import data_assimilation as DA
import config as p
import matplotlib.pyplot as plt
from scipy.fftpack import fft
plt.switch_backend('Agg')
plt.figure(figsize=(12, 7))

# read in obs and prior ensemble
x = np.load("output/truth.npy")
xens = np.load("output/ensemble_forecast.npy")[:, 0:20, :]
nx, nens, nt = xens.shape
tt = 8
xt = x[:, tt]
xb = xens[:, :, tt]
obs = np.load("output/obs.npy")
yo = obs[:, tt]

ROI = 5
xa = DA.EnKF(xb, p.obs_ind, yo, p.obs_err, ROI, alpha=0.0)

#xo = np.random.normal(0, 1, (nx, nens))
#R = misc.error_covariance(xo)
R = np.eye(nx)
# print(R)

##loalization
if ROI <= 0:
  rho = np.ones((nx, nx))
else:
  ii, jj = np.mgrid[0:nx, 0:nx]
  dist = np.sqrt((ii - jj)**2)
  dist = np.minimum(dist, nx - dist)
  rho = dist.copy()
  for i in range(nx):
    rho[:, i] = DA.GC_local_func(dist[:, i], ROI)

###test covariance relation
###1. see if SVD of ensemble perturbation can give unique singular 
###values to suffice La^-2 = Lb^-2 + Lo^-2
###ISSUE: the orthogonal basis are not shared between R and P
#xop = misc.ens_pert(xo)
#u, Lo, v = np.linalg.svd(xop)
#xbp = misc.ens_pert(xb)
#u, Lb, v = np.linalg.svd(xbp)
#xap = misc.ens_pert(xa)
#u, La, v = np.linalg.svd(xap)
#Pb = np.matmul(u, np.matmul(np.diag(Lb**2), u.T))/(nens-1)
#Pa = np.matmul(u, np.matmul(np.diag(La**2), u.T))/(nens-1)
#La1 = (Lb**-2 + Lo**-2)**-0.5
#Pa1 = np.matmul(u, np.matmul(np.diag(La1**2), u.T))/(nens-1)
#ax = plt.subplot(223)
#ax.semilogy(Lb**2, 'k')
#ax.semilogy(La**2, 'r')
#ax.semilogy(La1**2, 'g')
# ax.plot(np.ones(nx)**2, 'g')
# ax = plt.subplot(222)
# ax.plot(u[0, :])
# c = ax.contourf(u, cmap='seismic')
# plt.colorbar(c)

####2. see if eigenvalue decomposition of P, R can provide this 
####relationship: wa^-1 = wb^-1 + wo^-1, with shared eigenvectors v.
####when localization is applied, the eigenvectors for Pa and Pb can be different
Pb = misc.error_covariance(xb) * rho
Pa = misc.error_covariance(xa) * rho
wb, v = np.linalg.eig(Pb)
wa, v = np.linalg.eig(Pa)
wo = np.ones(nx)
wa1 = np.real((wo**-1 + wb**-1)**-1)
#print(wb)
#print(wa)
#print(wa1)
Pa1 = np.real(np.dot(v, np.dot(np.diag(wa1), v.T)))
Pa2 = np.linalg.inv(np.linalg.inv(Pb) + np.linalg.inv(R))

####compare trace of P
print(np.sqrt(np.trace(Pb)/nx))
print(np.sqrt(np.trace(Pa)/nx))
print(np.sqrt(np.trace(Pa1)/nx))
print(np.sqrt(np.trace(Pa2)/nx))

###compare P from different estimates
ax = plt.subplot(231)
c = ax.contourf(Pb, np.arange(-2, 2, 0.1), cmap='seismic')
ax.set_title(r'$\rho \circ P_b = V \Lambda_b V^T$')
ax = plt.subplot(232)
c = ax.contourf(Pa, np.arange(-2, 2, 0.1), cmap='seismic')
ax.set_title(r'$\rho \circ P_a = V \Lambda_a V^T$')
ax = plt.subplot(234)
c = ax.contourf(Pa1, np.arange(-2, 2, 0.1), cmap='seismic')
ax.set_title(r'$V [\Lambda_b^{-1}+\Lambda_o^{-1}]^{-1} V^T$')
ax = plt.subplot(235)
c = ax.contourf(Pa2, np.arange(-2, 2, 0.1), cmap='seismic')
ax.set_title(r'$[(\rho \circ P_b)^{-1}+R^{-1}]^{-1}$')

###plot eigenvalue spectrum
ax = plt.subplot(233)
ax.semilogy(wb, 'k', label=r'$\Lambda_b$')
ax.semilogy(wa, 'r', label=r'$\Lambda_a$')
ax.semilogy(wa1, 'g', label=r'$(\Lambda_b^{-1}+\Lambda_o^{-1})^{-1}$')
ax.legend(fontsize=12)

###check ensemble in state space
# ax = plt.subplot(223)
# ax.plot(xb, 'c')
# ax.plot(xt, 'k')
# ax = plt.subplot(224)
# ax.plot(xa, 'c')
# ax.plot(xt, 'k')

plt.savefig('1.pdf')
