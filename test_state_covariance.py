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
truth = np.load("output/truth.npy")
prior = np.load("output/ensemble_prior.npy")
post = np.load("output/ensemble_post.npy")
obs = np.load("output/obs.npy")
nx, nens, nt = prior.shape
# print(prior.shape)
tt = 12
nt = 1
cp = 4
xt = np.zeros(nx*nt)
xb = np.zeros((nx*nt, nens))
xa = np.zeros((nx*nt, nens))
yo = np.zeros(nx*nt)
for t in range(nt):
  xt[t*nx:(t+1)*nx] = truth[:, tt+cp*t]
  yo[t*nx:(t+1)*nx] = obs[:, tt+cp*t]
  xb[t*nx:(t+1)*nx, :] = prior[:, :, tt+cp*t]
  xa[t*nx:(t+1)*nx, :] = post[:, :, tt+cp*t]

##loalization
ROI = p.ROI
ROIt = p.ROIt
rho = np.zeros((nx*nt, nx*nt))
ii, jj = np.mgrid[0:nx, 0:nx]
dist = np.sqrt((ii - jj)**2)
dist = np.minimum(dist, nx - dist)
rhoblock = dist.copy()
for i in range(nx):
  rhoblock[:, i] = DA.GC_local_func(dist[:, i], ROI)
for t in range(nt):
  for s in range(nt):
    tdist = np.abs(t-s)
    rho[t*nx:(t+1)*nx, :][:, s*nx:(s+1)*nx] = rhoblock * DA.GC_local_func(np.array([tdist]), ROIt)

###covariance matrices
nobs = p.obs_ind.size
L = 0
if L<=0:
  corr = np.eye(nx)
else:
  corr = np.exp(-dist/L)
  # corr = (1+dist/L) * np.exp(-dist/L)  #SOAR
  # corr = np.cos(np.pi*dist/L) * np.exp(-dist/L)
R = corr * p.obs_err**2
H = np.eye(nx*nt)[p.obs_ind, :]
HTRinvH = np.dot(H.T, np.dot(np.linalg.inv(R), H))
Pb = misc.error_covariance(xb) * rho
Pa = misc.error_covariance(xa) * rho
# Pb = np.zeros((nx, nx))
# Pa = np.zeros((nx, nx))
# for t in range(0, nt, 4):
  # Pb += misc.error_covariance(prior[:, :, t]) * rho
  # Pa += misc.error_covariance(post[:, :, t]) * rho
# Pb = Pb/nt
# Pa = Pa/nt
# w, v = np.linalg.eig(R)
# Rsqinv = np.dot(v, np.dot(np.diag(w**-0.5), v.T))
# w, v = np.linalg.eig(Pb)
# Pbsq = np.dot(v, np.dot(np.diag(w**0.5), v.T))
# Pbsqinv = np.dot(v, np.dot(np.diag(w**-0.5), v.T))
# M = np.dot(Rsqinv, np.dot(H, Pbsq))

#canonical correlation analysis
w, v = np.linalg.eig(np.exp(-dist/1))
# u, s, v = np.linalg.svd(M)
wb = np.diag(np.dot(v.T, np.dot(Pb, v)))
wa = np.diag(np.dot(v.T, np.dot(Pa, v)))
wo = np.diag(np.dot(v.T, np.dot(HTRinvH, v))) **-1
wa1 = np.real((wo**-1 + wb**-1)**-1)
# print(wa[0:nx])
# print(wa1[0:nx])

###compare P from different estimates
clevel = np.arange(-1, 1.02, 0.02)
ax = plt.subplot(231)
c = ax.contourf(Pb, clevel, cmap='seismic')
#ax.set_title(r'$P_b = U \Lambda_b^2 U^T$')
#ax.set_xticks(np.arange(0, nx*nt, nx))
#ax.set_yticks(np.arange(0, nx*nt, nx))
ax = plt.subplot(232)
c = ax.contourf(R, clevel, cmap='seismic')
# c = ax.contourf(Pa, clevel, cmap='seismic')
# ax.set_title(r'$P_a = U \Lambda_a^2 U^T$')
# ax.set_xticks(np.arange(0, nx*nt, nx))
# ax.set_yticks(np.arange(0, nx*nt, nx))
ax = plt.subplot(234)
c = ax.contourf(np.dot(v.T, np.dot(Pb, v)), clevel, cmap='seismic')
# ax.set_title(r'$U (\Lambda_b^{-2}+\Lambda_o^{-2})^{-1} U^T$')
# ax.set_xticks(np.arange(0, nx*nt, nx))
# ax.set_yticks(np.arange(0, nx*nt, nx))
ax = plt.subplot(235)
c = ax.contourf(np.dot(v.T, np.dot(R, v)), clevel, cmap='seismic')
# ax.set_title(r'$[P_b^{-1}+H^TR^{-1}H]^{-1}$')
# ax.set_xticks(np.arange(0, nx*nt, nx))
# ax.set_yticks(np.arange(0, nx*nt, nx))

###plot eigenvalue spectrum
ax = plt.subplot(233)
ax.plot(np.sqrt(wb), 'k', label=r'$\Lambda_b$')
ax.plot(np.sqrt(wo), 'c', label=r'$\Lambda_o$')
ax.plot(np.sqrt(wa), 'r', label=r'$\Lambda_a$')
ax.plot(np.sqrt(wa1), 'g', label=r'$(\Lambda_b^{-2}+\Lambda_o^{-2})^{-\frac{1}{2}}$')
# ax.legend(fontsize=12)
# ax.set_title('Eigenspectrum')
ax.set_ylim(0, 5)

###plot eigenvectors
ax = plt.subplot(236)
# ax.contourf(v.T, clevel, cmap='seismic')
ax.plot(v[:, 0:10])
# for t in range(nt):
  # ax.plot(v[t*nx:(t+1)*nx, 0]+t, 'k')
# for t in range(nt):
#   ax.plot(v[t*nx:(t+1)*nx, 1]+t, 'r')
# for t in range(nt):
#   ax.plot(v[t*nx:(t+1)*nx, 2]+t, 'b')
# ax.set_title('First 3 eigenvectors')

plt.savefig('1.pdf')
