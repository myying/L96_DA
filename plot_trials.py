#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt

plt.switch_backend('Agg')
plt.figure(figsize=(10, 3))

outdir = "L5_N20_F8"
param1 = np.arange(2, 50, 2)
param2 = np.arange(1.0, 1.2, 0.01)



RMSEa = np.load(outdir+"/RMSEa.npy")
SPRDa = np.load(outdir+"/SPRDa.npy")
CRa = SPRDa/RMSEa
# CRa[np.where(CRa>2)] = 2
# RMSEa[np.where(RMSEa>2)] = 2

###plots for two parameters
ax = plt.subplot(121)
cs = ax.contourf(RMSEa, np.arange(0, 2.05, 0.05), cmap='jet')
# cs = ax.contourf(RMSEa, np.arange(0.14, 0.16, 0.001), cmap='jet')
plt.colorbar(cs)
ax.set_yticks(np.arange(0, param1.size, 6))
ax.set_yticklabels(np.round(param1[::6], 2))
ax.set_xticks(np.arange(0, param2.size, 5))
ax.set_xticklabels(np.round(param2[::5], 2))
ax = plt.subplot(122)
cs = ax.contourf(CRa, np.arange(0, 2.05, 0.05), cmap='seismic')
plt.colorbar(cs)
ax.set_yticks(np.arange(0, param1.size, 6))
ax.set_yticklabels(np.round(param1[::6], 2))
ax.set_xticks(np.arange(0, param2.size, 5))
ax.set_xticklabels(np.round(param2[::5], 2))

###plots for one parameter
# ax = plt.subplot(121)
# ax.semilogx(params, RMSEa, 'ko-')
# ax = plt.subplot(122)
# ax.semilogx(params, CRa, 'ko-')
# ax.semilogx(params, np.ones(params.size), color='0.7')

plt.savefig(outdir+'/rmse_cr.pdf')
