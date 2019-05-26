#!/usr/bin/env python
import numpy as np

nens = 20
r = np.arange(-1, 1.01, 0.01)
nk = r.size
r_data = np.zeros((nk, 2))
nm = 10000

def sample_corr(a, b):
  n = a.size
  vara = np.sum(a**2)/(n-1)
  varb = np.sum(b**2)/(n-1)
  cov = np.sum(a*b)/(n-1)
  corr = cov/np.sqrt(vara*varb)
  return corr

rs = np.zeros((nm, nk))
for k in range(nk):
  print(r[k])
  cov = np.eye(2)
  cov[0, 1] = r[k]
  cov[1, 0] = r[k]
  ##sample correlation
  for m in range(nm):
    rens = np.random.multivariate_normal(np.zeros(2), cov, size=nens)
    rs[m, k] = sample_corr(rens[:, 0], rens[:, 1])

for k in range(nk):
  # rs_sub = rs[np.where(np.logical_and(rs>=r[k]-0.005, rs<r[k]+0.005))]
  rs_sub = rs[:, k]
  nr = rs_sub.size
  rs_mean = np.mean(rs_sub)
  r_data[k, 0] = rs_mean
  r_data[k, 1] = np.sum((rs_sub-rs_mean)**2)/(nr-1)

np.save("sec_table", r_data)
