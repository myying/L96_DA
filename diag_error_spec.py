#!/usr/bin/env python
import numpy as np
import misc
import data_assimilation as DA
import config as p
import matplotlib.pyplot as plt
import sys
from scipy.fftpack import fft
plt.switch_backend('Agg')
plt.figure(figsize=(4, 3))

outdir = sys.argv[1]
t1 = 1000
t2 = 100000
truth = np.load("output/truth.npy")[:, t1:t2]
prior = np.load(outdir+"/ensemble_prior.npy")[:, :, t1:t2]
post = np.load(outdir+"/ensemble_post.npy")[:, :, t1:t2]
obs = np.load("output/obs.npy")[:, t1:t2]
nx, nens, nt1 = prior.shape
tt = 1
nt = 1
cp = p.cycle_period
L = float(sys.argv[2])
obs_err = float(sys.argv[3])

dk = int(sys.argv[4])
if dk < int(nx/2) :
  krange = np.arange(dk, int(nx/2), dk)
  R = DA.R_matrix(nx, p.obs_ind, np.array([0]), 1, 5, 0)
  Lo = misc.matrix_spec(R)
  obs_err_inf = np.ones(krange.size+1)
  obs_err_inf[0] = np.sqrt(np.mean(Lo[0:krange[0]+1]**2))
  obs_err_inf[krange.size] = np.sqrt(np.mean(Lo[krange[-1]+1:]**2))
  for i in range(1, krange.size):
    obs_err_inf[i] = np.sqrt(np.mean(Lo[krange[i-1]+1:krange[i]+1]**2))

###covariance matrices
t_ind = np.array([0])
R = DA.R_matrix(p.nx, p.obs_ind, t_ind, obs_err, L, 0)
Rt = DA.R_matrix(p.nx, p.obs_ind, t_ind, 1, 5, 0)
H = DA.H_matrix(p.nx, p.obs_ind, t_ind, 0)
HTRinvH = np.dot(H.T, np.dot(np.linalg.inv(R), H))
HTRtinvH = np.dot(H.T, np.dot(np.linalg.inv(Rt), H))
rho = DA.local_matrix(p.nx, t_ind, p.ROI, p.ROIt)  ##loalization
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
Qb = misc.Q_out(prior, truth)
Qa = misc.Q_out(post, truth)
print('prior rmse = {}, sprd = {}'.format(misc.rmse(Qb), misc.sprd(Pb)))
print('post rmse = {}, sprd = {}'.format(misc.rmse(Qa), misc.sprd(Pa)))
np.save(outdir+"/RMSEb", misc.rmse(Qb))
np.save(outdir+"/RMSEa", misc.rmse(Qa))
np.save(outdir+"/SPRDb", misc.sprd(Pb))
np.save(outdir+"/SPRDa", misc.sprd(Pa))
print('post CR = {}'.format(misc.rmse(Pa)/misc.sprd(Qa)))

##spectrum
Lb = misc.matrix_spec(Pb)
Lbt = misc.matrix_spec(Qb)
La = misc.matrix_spec(Pa)
Lat = misc.matrix_spec(Qa)
Lo = misc.matrix_spec(HTRinvH) ** -1
Lot = misc.matrix_spec(HTRtinvH) ** -1

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
ax.plot(Lbt, 'b', label=r'$\Lambda^{b*}$', linewidth=2)
ax.plot(Lat, 'r', label=r'$\Lambda^{a*}$', linewidth=2)
ax.plot(Lb, 'c', label=r'$\Lambda^b$', linewidth=2)
ax.plot(La, 'y', label=r'$\Lambda^a$', linewidth=2)
ax.plot(Lot, 'k', label=r'$\Lambda^{o*}$', linewidth=2)
if dk < int(nx/2) :
  print(krange)
  print(obs_err_inf)
  wn = np.arange(0, Lo.size)
  ns = krange.size+1
  for s in range(ns):
    if s == 0:
      ax.plot([0, krange[s]], obs_err_inf[s] * obs_err * np.ones(2), '.7', linestyle=':', linewidth=2)
    if s == ns-1:
      ax.plot([krange[s-1], nx/2-1], obs_err_inf[s] * obs_err * np.ones(2), '.7', linestyle=':', linewidth=2)
    if s > 0 and s < ns-1:
      ax.plot([krange[s-1], krange[s]-0.2], obs_err_inf[s] * obs_err * np.ones(2), '.7', linestyle=':', linewidth=2)
else:
  ax.plot(Lo, '.7', linestyle=':', label=r'$\Lambda^o$', linewidth=2)

###reference from a best case:
post = np.load("/glade/scratch/mying/L96_DA/EnSRF/dk20/L5.0_s1.0/N40_F8.0/ROI55_inf1.04/ensemble_post.npy")[:, :, t1:t2]
Qar = misc.Q_out(post, truth)
Lar = misc.matrix_spec(Qar)
ax.plot(Lar, 'k', linestyle=':', linewidth=1)

# ax.plot(np.sqrt(np.diag(Wa1)), 'g', label=r'$(\Lambda_b^{-2}+\Lambda_o^{-2})^{-\frac{1}{2}}$')
# ax.legend(fontsize=13, ncol=2)
ax.set_ylim(0, 2)
ax.set_xlim(-1, nx/2)
ax.tick_params(labelsize=12)
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
