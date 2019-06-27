#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt

plt.switch_backend('Agg')
plt.figure(figsize=(10, 3))

outdir = "output"
param1 = np.array([10, 20, 40, 80, 160, 2000])
param2 = np.array([2, 5, 8, 10, 15, 20, 30, 50])

RMSEa = np.load(outdir+"/RMSEa.npy")
SPRDa = np.load(outdir+"/SPRDa.npy")
CRa = SPRDa/RMSEa
# CRa[np.where(CRa>2)] = 2
# RMSEa[np.where(RMSEa>2)] = 2

###plots for two parameters
ax = plt.subplot(121)
c = ax.contourf(RMSEa, np.arange(0, 2.05, 0.05), cmap='jet')
plt.colorbar(c)
ax.set_xlim(0, 0.8)
ax.set_yticks(np.arange(param1.size))
ax.set_yticklabels(param1)
ax.set_xticks(np.arange(param2.size))
ax.set_xticklabels(param2)
ax = plt.subplot(122)
c = ax.contourf(CRa, np.arange(0, 2.05, 0.05), cmap='seismic')
plt.colorbar(c)
ax.set_xlim(0, 0.8)
ax.set_yticks(np.arange(param1.size))
ax.set_yticklabels(param1)
ax.set_xticks(np.arange(param2.size))
ax.set_xticklabels(param2)

###plots for one parameter
# ax = plt.subplot(121)
# ax.semilogx(params, RMSEa, 'ko-')
# ax = plt.subplot(122)
# ax.semilogx(params, CRa, 'ko-')
# ax.semilogx(params, np.ones(params.size), color='0.7')

plt.savefig(outdir+'/rmse_cr.pdf')
