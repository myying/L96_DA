#!/usr/bin/env python
import numpy as np
import misc
import data_assimilation as DA
import config as p
import matplotlib.pyplot as plt
plt.switch_backend('Agg')
plt.figure(figsize=(8, 4))

L = np.array([0, 1, 2, 5, 8, 15])
Lcolor = ('k', 'r', 'y', 'c', 'g', 'b')

ax = plt.subplot(121)
for i in range(L.size):
  R = DA.R_matrix(p.nx, p.obs_ind, np.array([0]), 1, L[i], 0)
  ax.plot(R[0, :], Lcolor[i], linewidth=2)
ax.set_ylim(-0.18, 1.18)
ax.set_xlim(-1, p.nx/2)
ax.tick_params(labelsize=12)
ax.set_xlabel('distance', fontsize=15)
ax.set_title(r'(a) Correlation     ', fontsize=18)

ax = plt.subplot(122)
for i in range(L.size):
  R = DA.R_matrix(p.nx, p.obs_ind, np.array([0]), 1, L[i], 0)
  Lo = misc.matrix_spec(R)
  ax.plot(Lo, Lcolor[i], linewidth=2, label=r'$L$={}'.format(L[i]))
ax.set_ylim(0, 4)
ax.set_xlim(-1, p.nx/2)
ax.tick_params(labelsize=12)
ax.legend(loc=1, fontsize=14)
ax.set_xlabel('wavenumber', fontsize=15)
ax.set_title(r'(b) Spectrum ($\Lambda^o$)', fontsize=18)

plt.tight_layout()
plt.savefig('1.pdf')
