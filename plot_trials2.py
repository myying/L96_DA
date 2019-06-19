#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt

plt.switch_backend('Agg')
plt.figure(figsize=(10, 3))

# outdir = "/home/yingyue/Google_Drive/data_assimilation/observation_error/Ying_2019_obserr/output/N20_L2s1_L0_F7"
outdir = "output"

# param1 = np.array([2, 5, 8, 10, 15, 20, 30, 50])
param1 = np.array([0, 0.2, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0])
param2 = np.arange(0.0, 1.0, 0.1)
# param2 = np.array([1.0, 1.05, 1.1, 1.2, 1.5, 2.0])

RMSEa = np.load(outdir+"/RMSEa.npy")
SPRDa = np.load(outdir+"/SPRDa.npy")
CRa = SPRDa/RMSEa
CRa[np.where(CRa>2)] = 2
RMSEa[np.where(RMSEa>2)] = 2

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

plt.savefig(outdir+'/rmse_cr.pdf')
