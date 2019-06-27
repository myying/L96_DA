#!/usr/bin/env python
import numpy as np
import misc
import data_assimilation as DA
import config as p
import matplotlib.pyplot as plt
import sys
from scipy.fftpack import fft
plt.switch_backend('Agg')
plt.figure(figsize=(8, 6))

outdir = sys.argv[1]
tt = int(sys.argv[2])
truth = np.load(outdir+"/truth.npy")
prior = np.load(outdir+"/ensemble_forecast.npy")
nx, nens, nt = prior.shape

###covariance matrices
# Pb = np.zeros((nx, nx))
# for t in range(nt):
#   Pb += misc.error_covariance(prior[:, :, t])
# Pb = Pb/nt
Pb = misc.error_covariance(prior[:, :, tt])

##actual error matrices
# Qb = misc.Q_out(prior, truth)
Qb = misc.Q_out(prior[:, :, tt:tt+1], truth[:, tt:tt+1])
print('prior rmse = {}, sprd = {}'.format(misc.rmse(Qb), misc.sprd(Pb)))

##spectrum
Lb = misc.matrix_spec(Pb)
Lbt = misc.matrix_spec(Qb)

###plot eigenvalue spectrum
ax = plt.subplot(221)
ax.plot(Lbt, 'b', label=r'$\Lambda^{b*}$')
ax.plot(Lb, 'c', label=r'$\Lambda^b$')
# ax.legend(fontsize=13, ncol=2)
ax.set_ylim(0, 1)
ax.set_xlim(-1, p.nx/2)

# plt.savefig('{:04d}.png'.format(tt), dpi=200)
plt.savefig('1.pdf')
