#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import sys

plt.switch_backend('Agg')
plt.figure(figsize=(8, 3))

outdir = "/glade/scratch/mying/L96_DA/"
filter_kind = "EnSRF"
L = 0.0
obs_err = 1.0
# param1 = np.arange(0, 9)
# param2 = np.array([0.5, 0.8, 1.0, 1.2, 1.5])
nens = 20 #int(sys.argv[1])
# param1 = np.array([10, 20, 40, 80, 160])
F = 8.0
# ROI = 0
# alpha = 0.0
param2 = np.arange(5, 51, 5)
param1 = np.arange(0, 1, 0.1)

RMSEa = np.zeros((param1.size, param2.size))
CRa = np.zeros((param1.size, param2.size))

for i in range(param1.size):
  for j in range(param2.size):
    # casename = filter_kind+"/L{:3.1f}_s{:3.1f}".format(param1[i], param2[j])+"/N{}_F{}".format(nens, F)+"/ROI{}".format(ROI)+"_relax{:4.2f}".format(alpha)
    casename = filter_kind+"/L{:3.1f}_s{:3.1f}".format(L, obs_err)+"/N{}_F{}".format(nens, F)+"/ROI{}".format(param2[j])+"_relax{:4.2f}".format(param1[i])

    RMSEa[i, j] = np.load(outdir+casename+"/RMSEa.npy")
    CRa[i, j] = np.load(outdir+casename+"/SPRDa.npy") / np.load(outdir+casename+"/RMSEa.npy")

RMSEa[np.where(np.isnan(RMSEa))] = 2.0
CRa[np.where(np.isnan(CRa))] = 2.0
RMSEa[np.where(RMSEa>2)] = 2.0
CRa[np.where(CRa>2)] = 2.0
print(param1)
print(param2)
print(RMSEa)

print(np.min(RMSEa))
ind = np.where(RMSEa==np.min(RMSEa))
print(param1[ind[0]])
print(param2[ind[1]])
# print(CRa)

###plots for two parameters
ii = np.tile(param1, (param2.size, 1)).T
jj = np.tile(param2, (param1.size, 1))
ax = plt.subplot(121)
cs = ax.contourf(ii, jj, RMSEa, np.arange(0, 2.05, 0.05), cmap='jet')
plt.colorbar(cs)
ax = plt.subplot(122)
cs = ax.contourf(ii, jj, CRa, np.arange(0, 2.05, 0.05), cmap='seismic')
plt.colorbar(cs)
# ax.set_yticks(np.arange(0, param2.size, 1))
# ax.set_yticklabels(np.round(param2[::1], 2))
# ax.set_xticks(np.arange(0, param1.size, 1))
# ax.set_xticklabels(np.round(param1[::1], 2))

###plots for one parameter
# ax = plt.subplot(121)
# ax.semilogx(params, RMSEa, 'ko-')
# ax = plt.subplot(122)
# ax.semilogx(params, CRa, 'ko-')
# ax.semilogx(params, np.ones(params.size), color='0.7')

plt.savefig("1.pdf")
# plt.savefig(outdir+'/rmse_cr.pdf')
