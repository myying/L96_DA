import numpy as np


def forward(x, nx, F, dt):
  for n in range(int(dt/0.005)):
    x = x + dxdt_func(x, nx, F)*0.005
  return x


def dxdt_func(x, nx, F):
  dxdt = x*0.0
  for i in np.arange(2, nx-1):
    dxdt[i] = (x[i+1] - x[i-2]) * x[i-1] - x[i] + F
  dxdt[0]  = (x[1] - x[-2]) * x[-1] - x[0] + F
  dxdt[1]  = (x[2] - x[-1]) * x[0]  - x[1] + F
  dxdt[-1] = (x[0] - x[-3]) * x[-2] - x[-1]+ F
  return dxdt
