#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import misc
from scipy.fftpack import fft2, ifft2, fftshift
plt.switch_backend('Agg')

###diagnose spectral space error/spread variances for prior/posterior/obs errors
x = np.load("output/truth.npy")
y = np.load("output/obs.npy")
xens = np.load("output/ensemble_prior.npy")
xens1 = np.load("output/ensemble_post.npy")

x_int = 1
t_int = 4
truth = x[::x_int, ::t_int]
obs = y[::x_int, ::t_int]
prior = xens[::x_int, :, ::t_int]
post = xens1[::x_int, :, ::t_int]
nx, nens, nt = prior.shape

prior_err = np.mean(prior, axis=1) - truth
post_err = np.mean(post, axis=1) - truth
obs_err = obs - truth

#errors
x = obs_err
xh = fft2(x - np.mean(x))/nt/nx
pwr = np.real(xh * np.conj(xh))
print(1e5*np.mean(pwr[5:15, 0:20]))

x = prior_err
xh = fft2(x - np.mean(x))/nt/nx
pwr = np.real(xh * np.conj(xh))
print(1e5*np.mean(pwr[5:15, 0:20]))

x = post_err
xh = fft2(x - np.mean(x))/nt/nx
pwr = np.real(xh * np.conj(xh))
print(1e5*np.mean(pwr[5:15, 0:20]))

#prior/post spread
mean = np.mean(prior, axis=1)
xh = np.zeros(prior.shape, dtype=complex)
for m in range(nens):
  xh[:, m, :] = fft2(prior[:, m, :] - mean)/nt/nx
pwr = np.real(np.mean(xh * np.conj(xh), axis=1))
print(1e5*np.mean(pwr[5:15, 0:20]))

mean = np.mean(post, axis=1)
xh = np.zeros(post.shape, dtype=complex)
for m in range(nens):
  xh[:, m, :] = fft2(post[:, m, :] - mean)/nt/nx
pwr = np.real(np.mean(xh * np.conj(xh), axis=1))
print(1e5*np.mean(pwr[5:15, 0:20]))

# ax = plt.subplot(221)
# #c = ax.contourf(x.T, np.arange(-15, 20, 1), cmap='seismic')
# c = ax.contourf(x.T, np.arange(-2, 2, 0.1), cmap='seismic')
# plt.colorbar(c)
# ax = plt.subplot(222)
# #c = ax.contourf((pwr[:, 0:100]).T, np.arange(0, 0.3, 0.01), cmap='bone_r')
# c = ax.contourf((pwr[:, 0:100]).T, np.arange(0, 0.001, 0.0001), cmap='bone_r')
# plt.colorbar(c)
# ax.set_xticks(np.arange(-1, 40, 10))
# ax.set_xticklabels(np.arange(-20, 21, 10))

# plt.savefig('1.pdf')

