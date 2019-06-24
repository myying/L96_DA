import numpy as np
from scipy.fftpack import fft, ifft, fftshift

def grid2spec(x):
  nx = x.size
  nk = int(nx / 2)
  tmp = fft(x)
  xh = tmp[0:nk+1] / nx
  return xh

def spec2grid(xh):
  nk = xh.size - 1
  nx = 2 * nk
  tmp = np.zeros([nx], dtype=complex)
  tmp[0:nk+1] = xh
  tmp[-1:-nk-1:-1] = np.conj(xh[1:nk+1])
  tmp = tmp * nx
  x = np.real(ifft(tmp))
  return x

def spec_convolve(a, b):
  nk = a.size - 1
  nx = 2 * nk
  af = np.zeros([nx], dtype=complex)
  bf = np.zeros([nx], dtype=complex)
  af[0:nk+1] = a
  af[-1:-nk-1:-1] = np.conj(a[1:nk+1])
  af = fftshift(af)
  bf[0:nk+1] = b
  bf[-1:-nk-1:-1] = np.conj(b[1:nk+1])
  bf = fftshift(bf)
  cf = np.convolve(af, bf, 'same')
  return np.conj(cf[nk+1:0:-1])

def spec_bandpass(x, krange, s):
  nx = x.size
  ns = krange.size + 1
  xh = grid2spec(x)
  nk = xh.size
  wn = np.arange(0, nk)
  f = np.zeros(nk)
  if s == 0:
    f[np.where(wn<=krange[s])] = 1.0
  if s == ns-1:
    f[np.where(wn>krange[s-1])] = 1.0
  if s > 0 and s < ns-1:
    f[np.where(np.logical_and(wn>krange[s-1], wn<=krange[s]))] = 1.0
  x_f = spec2grid(xh*f)
  return x_f

###spatial operators
def warp(A, di, dt):
  A1 = A.copy()
  ni, nt = A.shape
  for i in range(ni):
    for t in range(nt):
      A1[i, t] = interp2d(A, (i+di[i, t], t+dt[i, t]))
  return A1

def interp1d(A, loc):
  nx = A.size
  io = loc
  io1 = int(np.floor(io))
  io2 = io1+1
  di = io - io1
  Ao = (1-di)*A[io1] + di*A[io2]
  return Ao

def interp2d(A, loc):
  ni, nt = A.shape
  io = loc[0]
  to = loc[1]
  io1 = int(np.floor(io)) % ni
  io2 = int(np.floor(io+1)) % ni
  # to1 = int(max(0, np.floor(to)))
  # to2 = int(min(nt-1, np.floor(to+1)))
  to1 = int(np.floor(to)) % nt
  to2 = int(np.floor(to+1)) % nt
  di = io - np.floor(io)
  dt = to - np.floor(to)
  Ao = (1-di)*(1-dt)*A[io1, to1] + di*(1-dt)*A[io2, to1] + (1-di)*dt*A[io1, to2] + di*dt*A[io2, to2]
  return Ao

def deriv_x(f):
  fx = 0.5*(np.roll(f, -1, axis=0) - np.roll(f, 1, axis=0))
  return fx

def deriv_t(f):
  ft = 0.5*(np.roll(f, -1, axis=1) - np.roll(f, 1, axis=1))
  # ft = f.copy()
  # nx, nt = ft.shape
  # ft[:, 1:nt] = 0.5*(np.roll(f[:, 1:nt], -1, axis=1) - np.roll(f[:, 1:nt], 1, axis=1))
  # ft[:, 0] = f[:, 1] - f[:, 0]
  # ft[:, nt-1] = f[:, nt-1] - f[:, nt-2]
  return ft

def deriv_xt(f):
  return deriv_x(deriv_t(f))

def deriv_xx(f):
  return deriv_x(deriv_x(f))

def deriv_tt(f):
  return deriv_t(deriv_t(f))

def laplacian(f):
  return deriv_xx(f) + deriv_tt(f)

def coarsen(f, level):
  for k in range(level):
    nx, nt = f.shape
    f1 = 0.25*(f[0:nx:2, :][:, 0:nt:2] + f[1:nx:2, :][:, 0:nt:2] + f[0:nx:2, :][:, 1:nt:2] + f[1:nx:2, :][:, 1:nt:2])
    f = f1
  return f

def sharpen(f, level):
  for k in range(level):
    nx, nt = f.shape
    f1 = np.zeros((nx*2, nt))
    f1[0:nx*2:2, :] = f
    f1[1:nx*2:2, :] = 0.5*(np.roll(f, -1, axis=0) + f)
    f2 = np.zeros((nx*2, nt*2))
    f2[:, 0:nt*2:2] = f1
    f2[:, 1:nt*2:2] = 0.5*(np.roll(f1, -1, axis=1) + f1)
    f = f2
  return f

###Fourier basis
def fourier_basis(nx):
  v = np.zeros((nx, nx))
  v[:, 0] = 1.0/np.sqrt(nx)
  ii = np.mgrid[0:nx]
  for n in range(1, nx, 2):
    v[:, n] = np.cos(np.pi*(n+1)*ii/nx)/np.sqrt(nx/2)
  for n in range(2, nx, 2):
    v[:, n] = -np.sin(np.pi*n*ii/nx)/np.sqrt(nx/2)
  v[:, -1] = np.cos(np.pi*ii)/np.sqrt(nx)
  return v


###ensemble estimated error covariance
def ens_pert(xens):
  nx, nens = xens.shape
  xp = np.zeros((nx, nens))
  xm = np.mean(xens, axis=1)
  for m in range(nens):
    xp[:, m] = xens[:, m] - xm
  return xp

def error_covariance(xens):
  nx, nens = xens.shape
  xp = ens_pert(xens)
  P = np.dot(xp, xp.T) / (nens-1)
  return P


###diagnostics
def P_out(xens):
  nx, nens, nt = xens.shape
  P = np.zeros((nx, nx))
  for t in range(nt):
    P += error_covariance(xens[:, :, t])
  P = P/nt
  return P

def sprd(P):
  return np.sqrt(np.mean(np.diag(P)))

def Q_out(xens, xt):
  nx, nens, nt = xens.shape
  error = np.mean(xens, axis=1) - xt
  Q = np.dot(error, error.T) / nt
  return Q

def rmse(Q):
  return np.sqrt(np.mean(np.diag(Q)))

def matrix_spec(P):
  n, n = P.shape
  U = fourier_basis(n)
  W = np.dot(U.T, np.dot(P, U))
  d = np.diag(W)
  m = int(n/2)*2
  d1 = np.sqrt( (d[0:m:2] + d[1:m:2])/2 )
  return d1
