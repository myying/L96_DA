#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt

plt.switch_backend('Agg')
plt.figure(figsize=(10, 3))

# outdir = "/home/yingyue/Google_Drive/data_assimilation/observation_error/Ying_2019_obserr/output/N20_L2s1_L0_F7"
outdir = "output"

# params = np.array([0.1, 0.2, 0.5, 0.8, 1.0, 1.2, 1.5, 1.8, 2.0, 3.0, 5.0])
params = np.array([0, 0.2, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0])

RMSEa = np.load(outdir+"/RMSEa.npy")
SPRDa = np.load(outdir+"/SPRDa.npy")
CRa = SPRDa/RMSEa
# CRa[np.where(CRa>2)] = 2
# RMSEa[np.where(RMSEa>2)] = 2

ax = plt.subplot(121)
ax.semilogx(params, RMSEa, 'ko-')
ax = plt.subplot(122)
ax.semilogx(params, CRa, 'ko-')
ax.semilogx(params, np.ones(params.size), color='0.7') 

plt.savefig(outdir+'/rmse_cr.pdf')
