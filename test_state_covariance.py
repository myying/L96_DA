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
print(prior.shape)
tt = 24
nt = 3
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

R = np.eye(nx*nt)

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

Pb = misc.error_covariance(xb) * rho
Pa = misc.error_covariance(xa) * rho
###common eigenvectors of Pa and Pb????
w, v = np.linalg.eig(Pb)
wb = np.diag(np.dot(v.T, np.dot(Pb, v)))
wa = np.diag(np.dot(v.T, np.dot(Pa, v)))
wo = np.ones(nx*nt)
wa1 = np.real((wo**-1 + wb**-1)**-1)
print(wa[0:nx])
print(wa1[0:nx])
Pa1 = np.real(np.dot(v, np.dot(np.diag(wa1), v.T)))
Pa2 = np.linalg.inv(np.linalg.inv(Pb) + np.linalg.inv(R))
print(np.sqrt(np.trace(Pa)/nt/nx))
print(np.sqrt(np.trace(Pa1)/nt/nx))

###compare P from different estimates
clevel = np.arange(-1, 1, 0.02)
ax = plt.subplot(231)
c = ax.contourf(Pb, clevel, cmap='seismic')
ax.set_title(r'$\rho \circ P_b = U \Lambda_b^2 U^T$')
ax = plt.subplot(232)
c = ax.contourf(Pa, clevel, cmap='seismic')
ax.set_title(r'$\rho \circ P_a = U \Lambda_a^2 U^T$')
ax = plt.subplot(234)
c = ax.contourf(Pa1, clevel, cmap='seismic')
ax.set_title(r'$U (\Lambda_b^{-2}+\Lambda_o^{-2})^{-1} U^T$')
ax = plt.subplot(235)
c = ax.contourf(Pa2, clevel, cmap='seismic')
ax.set_title(r'$[(\rho \circ P_b)^{-1}+R^{-1}]^{-1}$')

###plot eigenvalue spectrum
ax = plt.subplot(233)
ax.plot(np.sqrt(wb), 'k', label=r'$\Lambda_b$')
ax.plot(np.sqrt(wo), 'c', label=r'$\Lambda_o$')
ax.plot(np.sqrt(wa), 'r', label=r'$\Lambda_a$')
ax.plot(np.sqrt(wa1), 'g', label=r'$(\Lambda_b^{-2}+\Lambda_o^{-2})^{-\frac{1}{2}}$')
ax.legend(fontsize=12)
ax.set_title('Eigenspectrum')

###plot eigenvectors
ax = plt.subplot(236)
for t in range(nt):
  ax.plot(v[t*nx:(t+1)*nx, 0]+t, 'k')
for t in range(nt):
  ax.plot(v[t*nx:(t+1)*nx, 1]+t, 'r')
for t in range(nt):
  ax.plot(v[t*nx:(t+1)*nx, 2]+t, 'b')
ax.set_title('First 3 eigenvectors')

plt.savefig('1.pdf')
