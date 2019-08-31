#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import sys

plt.switch_backend('Agg')
plt.figure(figsize=(8, 3))

outdir = "/glade/scratch/mying/L96_DA/"
filter_kind = "EnSRF" #sys.argv[1] #"EnSRF"
# L = 3.0 #float(sys.argv[2]) ##5.0
# obs_err = 1.0 #float(sys.argv[3]) ##1.0
param1 = np.array([0, 0.1, 0.2, 0.5, 0.8, 1, 2, 3, 5, 7, 8, 10])
param2 = np.arange(0.4, 1.7, 0.2)
nens = 40 #int(sys.argv[4]) ##20
# param1 = np.array([10, 20, 40, 80, 160])
F = 8.0 #float(sys.argv[5]) ##8.0
ROI = 60
inflate = 1.04
# param2 = np.arange(30, 61, 5)
# param1 = np.arange(1.0, 1.21, 0.02)
dk = 20

RMSEa = np.zeros((param1.size, param2.size))
CRa = np.zeros((param1.size, param2.size))

for i in range(param1.size):
  for j in range(param2.size):
    casename = filter_kind+"/dk20/L{:3.1f}_s{:3.1f}".format(param1[i], param2[j])+"/N{}_F{}".format(nens, F)+"/ROI{}".format(ROI)+"_inf{:4.2f}".format(inflate)
    # casename = filter_kind+"/dk{}".format(dk)+"/L{:3.1f}_s{:3.1f}".format(L, obs_err)+"/N{}_F{}".format(nens, F)+"/ROI{}".format(param2[j])+"_inf{:4.2f}".format(param1[i])

    RMSEa[i, j] = np.load(outdir+casename+"/RMSEa.npy")
    CRa[i, j] = np.load(outdir+casename+"/SPRDa.npy") / np.load(outdir+casename+"/RMSEa.npy")

RMSEa[np.where(np.isnan(RMSEa))] = 1.0
RMSEa[np.where(RMSEa>1)] = 1.0
CRa[np.where(np.isnan(CRa))] = 2.0
CRa[np.where(CRa>2)] = 2.0
print(param1)
print(param2)
print(RMSEa)
print(CRa)
ind = np.where(RMSEa==np.min(RMSEa))
# ind = np.where(np.abs(CRa-1)==np.min(np.abs(CRa-1)))
print('    RMSEa = ', RMSEa[ind[0], ind[1]])
print('      CRa = ', CRa[ind[0], ind[1]])
print('      inf = ', param1[ind[0]])
print('      ROI = ', param2[ind[1]])

###plots for two parameters
param1[0] = 0.07
ii = np.tile(param1, (param2.size, 1)).T
jj = np.tile(param2, (param1.size, 1))
ax = plt.subplot(121)
cs = ax.contourf(ii, jj, RMSEa, np.arange(0, 1.01, 0.05), cmap='jet')

plt.colorbar(cs)
ax.set_xscale("log")
ax = plt.subplot(122)
cs = ax.contourf(ii, jj, CRa, np.arange(0, 2.01, 0.2), cmap='bwr')
plt.colorbar(cs)
ax.set_xscale("log")
# ax.set_yticks(np.arange(0, param2.size, 1))
# ax.set_yticklabels(np.round(param2[::1], 2))
# ax.set_xticks(np.arange(0, param1.size, 1))
# ax.set_xticklabels(np.round(param1[::1], 2))

##plots for one parameter
# ax = plt.subplot(121)
# ax.semilogx(params, RMSEa, 'ko-')
# ax = plt.subplot(122)
# ax.semilogx(params, CRa, 'ko-')
# ax.semilogx(params, np.ones(params.size), color='0.7')

plt.savefig("1.pdf")
# plt.savefig(outdir+'/rmse_cr.pdf')
