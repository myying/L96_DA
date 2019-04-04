#!/usr/bin/env python3
import numpy as np
n = 100
dat = np.zeros((5, n))
for i in range(n):
  dat[:, i] = np.loadtxt('trials/{:03d}'.format(i+1))
# print(dat[2, :])
out = np.mean(dat, axis=1)
print(out)
print(1/(1/out[0]+1/out[1]))
print(1/(1/out[0]+1/out[3]))

