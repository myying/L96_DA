import numpy as np

##nonlinear forward model
def M_nl(x, nx, F, dt):
  for n in range(int(dt/0.005)):
    x = x + dxdt_func(x, nx, F)*0.005
  return x


def dxdt_func(x, nx, F):
  dxdt = np.zeros(x.shape)
  if x.ndim == 1:
    for i in np.arange(2, nx-1):
      dxdt[i] = (x[i+1] - x[i-2]) * x[i-1] - x[i] + F
    dxdt[0]  = (x[1] - x[-2]) * x[-1] - x[0] + F
    dxdt[1]  = (x[2] - x[-1]) * x[0]  - x[1] + F
    dxdt[-1] = (x[0] - x[-3]) * x[-2] - x[-1]+ F
  if x.ndim == 2:
    for i in np.arange(2, nx-1):
      dxdt[i, :] = (x[i+1, :] - x[i-2, :]) * x[i-1, :] - x[i, :] + F
    dxdt[0, :]  = (x[1, :] - x[-2, :]) * x[-1, :] - x[0, :] + F
    dxdt[1, :]  = (x[2, :] - x[-1, :]) * x[0, :]  - x[1, :] + F
    dxdt[-1, :] = (x[0, :] - x[-3, :]) * x[-2, :] - x[-1, :]+ F
  return dxdt


##tangent linear and adjoint models
def M_tl():
  return dx


def M_ad():
  return dx
