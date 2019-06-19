#!/usr/bin/env python
import numpy as np
import misc
import data_assimilation as DA
import config as p
import matplotlib.pyplot as plt
import sys
from scipy.fftpack import fft
plt.switch_backend('Agg')
plt.figure(figsize=(3, 3))

outdir = sys.argv[1]
# read in obs and prior ensemble
truth = np.load(outdir+"/truth.npy")
prior = np.load(outdir+"/ensemble_prior.npy")
post = np.load(outdir+"/ensemble_post.npy")
obs = np.load(outdir+"/obs.npy")
nx, nens, nt1 = prior.shape
tt = 1
nt = 1
cp = p.cycle_period
# xt = np.zeros(nx*nt)
# xb = np.zeros((nx*nt, nens))
# xa = np.zeros((nx*nt, nens))
# yo = np.zeros(nx*nt)
# for t in range(nt):
#   xt[t*nx:(t+1)*nx] = truth[:, tt+cp*t]
#   yo[t*nx:(t+1)*nx] = obs[:, tt+cp*t]
#   xb[t*nx:(t+1)*nx, :] = prior[:, :, tt+cp*t]
#   xa[t*nx:(t+1)*nx, :] = post[:, :, tt+cp*t]

###covariance matrices
R = DA.R_matrix(nx, nt, p.obs_ind, p.obs_err, p.L, p.Lt, p.corr_kind)
Rt = DA.R_matrix(nx, nt, p.obs_ind, 1, 2, 0, 1)
H = DA.H_matrix(nx, nt, p.obs_ind)
HTRinvH = np.dot(H.T, np.dot(np.linalg.inv(R), H))
HTRtinvH = np.dot(H.T, np.dot(np.linalg.inv(Rt), H))
rho = DA.local_matrix(nx, nt, p.ROI, p.ROIt)  ##loalization
# Pb = misc.error_covariance(xb) * rho
# Pa = misc.error_covariance(xa) * rho
Pb = np.zeros((nx, nx))
Pa = np.zeros((nx, nx))
for t in range(0, nt1, cp):
  Pb += misc.error_covariance(prior[:, :, t]) * rho
  Pa += misc.error_covariance(post[:, :, t]) * rho
Pb = Pb/(nt1/cp)
Pa = Pa/(nt1/cp)
# w, v = np.linalg.eig(R)
# Rsqinv = np.dot(v, np.dot(np.diag(w**-0.5), v.T))
# w, v = np.linalg.eig(Pb)
# Pbsq = np.dot(v, np.dot(np.diag(w**0.5), v.T))
# Pbsqinv = np.dot(v, np.dot(np.diag(w**-0.5), v.T))
# M = np.dot(Rsqinv, np.dot(H, Pbsq))

##actual error matrices
eb = np.mean(prior, axis=1) - truth
ea = np.mean(post, axis=1) - truth
Qb = np.dot(eb[:, ::cp], eb[:, ::cp].T)/(nt1/cp)
Qa = np.dot(ea[:, ::cp], ea[:, ::cp].T)/(nt1/cp)
print('prior rmse = {}, sprd = {}'.format(np.sqrt(np.mean(np.diag(Qb))), np.sqrt(np.mean(np.diag(Pb)))))
print('post rmse = {}, sprd = {}'.format(np.sqrt(np.mean(np.diag(Qa))), np.sqrt(np.mean(np.diag(Pa)))))

##Fourier basis
v = misc.fourier_basis(nx)

Wb = np.dot(v.T, np.dot(Pb, v))
Wa = np.dot(v.T, np.dot(Pa, v))
wo = np.diag(np.dot(v.T, np.dot(HTRinvH, v))) **-1
wot = np.diag(np.dot(v.T, np.dot(HTRtinvH, v))) **-1
Wb1 = np.dot(v.T, np.dot(Qb, v))
Wa1 = np.dot(v.T, np.dot(Qa, v))
# Wa1 = np.linalg.inv(np.diag(wo**-1) + np.linalg.inv(Wb))

###compare P from different estimates
# clevel = np.arange(-3, 3.1, 0.1)
# ax = plt.subplot(231)
# c = ax.contourf(Pb, clevel, cmap='seismic')
# ax.set_title(r'$P_b = U \Lambda_b^2 U^T$')
# ax = plt.subplot(232)
# c = ax.contourf(Qb, clevel, cmap='seismic')
# ax.set_title(r'$\tilde{P}_b = U \tilde{\Lambda}_b^2 U^T$')
# ax = plt.subplot(234)
# c = ax.contourf(Pa, clevel, cmap='seismic')
# ax.set_title(r'$P^a = U \Lambda_a^2 U^T$')
# ax = plt.subplot(235)
# c = ax.contourf(Qa, clevel, cmap='seismic')
# ax.set_title(r'$\tilde{P}_a = U \tilde{\Lambda}_a^2 U^T$')
# ax.set_title(r'$[P_b^{-1}+H^TR^{-1}H]^{-1}$')
# ax.set_xticks(np.arange(0, nx*nt, nx))
# ax.set_yticks(np.arange(0, nx*nt, nx))

###plot eigenvalue spectrum
ax = plt.subplot(111)
ax.plot(np.sqrt(wot[::2]), 'k', label=r'$\Lambda^{o*}$')
ax.plot(np.sqrt(np.diag(Wb1)[::2]), 'b', label=r'$\Lambda^{b*}$')
ax.plot(np.sqrt(np.diag(Wa1)[::2]), 'r', label=r'$\Lambda^{a*}$')
ax.plot(np.sqrt(wo[::2]), 'k:', label=r'$\Lambda^o$')
ax.plot(np.sqrt(np.diag(Wb)[::2]), 'c', label=r'$\Lambda^b$')
ax.plot(np.sqrt(np.diag(Wa)[::2]), 'y', label=r'$\Lambda^a$')
# ax.plot(np.sqrt(np.diag(Wa1)), 'g', label=r'$(\Lambda_b^{-2}+\Lambda_o^{-2})^{-\frac{1}{2}}$')
ax.legend(fontsize=13, ncol=2)
ax.set_ylim(0, 5)
# ax.set_xlabel('wavenumber')

###plot eigenvectors
# ax = plt.subplot(236)
# ax.contourf(v[:, ::2].T, np.arange(-0.3, 0.31, 0.01), cmap='jet')
# for t in range(nt):
#   ax.plot(v[t*nx:(t+1)*nx, 0]+t, 'k')
# for t in range(nt):
#   ax.plot(v[t*nx:(t+1)*nx, 1]+t, 'r')
# for t in range(nt):
#   ax.plot(v[t*nx:(t+1)*nx, 2]+t, 'b')
# ax.set_title('First 3 eigenvectors')

plt.savefig(outdir+'/spec.pdf')
