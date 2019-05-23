#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt

plt.switch_backend('Agg')
plt.figure(figsize=(10, 3))

outdir = "/home/yingyue/Google_Drive/data_assimilation/observation_error/Ying_2019_obserr/output/N20_L2s1_L0_1"

ROI = np.array([1, 2, 5, 8, 10, 15, 20, 30, 50])
alpha = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])

RMSEa = np.load(outdir+"/RMSEa.npy")
SPRDa = np.load(outdir+"/SPRDa.npy")
CRa = SPRDa/RMSEa
CRa[np.where(CRa>2)] = 2
RMSEa[np.where(RMSEa>2)] = 2

ax = plt.subplot(121)
c = ax.contourf(RMSEa, np.arange(0, 2.05, 0.05), cmap='jet')
plt.colorbar(c)
ax.set_xlim(0, 0.8)
ax.set_yticks(np.arange(ROI.size))
ax.set_yticklabels(ROI)
ax.set_xticks(np.arange(alpha.size))
ax.set_xticklabels(alpha)
ax = plt.subplot(122)
c = ax.contourf(CRa, np.arange(0, 2.05, 0.05), cmap='seismic')
plt.colorbar(c)
ax.set_xlim(0, 0.8)
ax.set_yticks(np.arange(ROI.size))
ax.set_yticklabels(ROI)
ax.set_xticks(np.arange(alpha.size))
ax.set_xticklabels(alpha)

plt.savefig(outdir+'/rmse_cr.pdf')
