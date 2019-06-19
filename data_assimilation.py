import numpy as np
import misc

##filters, full matrix version with perturbed obs
def EnKF(prior, yo, H, R, rho):
  post = prior.copy()
  nx, nens = prior.shape
  nobs = yo.size
  obs = np.zeros((nobs, nens))
  for k in range(nens):
    yo_pert = np.random.multivariate_normal(np.zeros(nobs), R)
    obs[:, k] = yo + yo_pert
  Pb = misc.error_covariance(prior)
  K = np.dot(np.dot(Pb*rho, H.T), np.linalg.inv(np.dot(np.dot(H, Pb*rho), H.T) + R))
  for k in range(nens):
    post[:, k] = prior[:, k] + np.dot(K, (obs[:, k] - np.dot(H, prior[:, k])))
  return post

##serial EnKF, assimilates obs one at a time
def EnKF_serial(prior, yo, H, R, D, ROI):
  post = prior.copy()
  nx, nens = prior.shape
  nobs = yo.size
  ens_mean = np.mean(post, axis=1)
  ens_pert = misc.ens_pert(post)
  for j in range(nobs):
    hxm = np.dot(H[j, :], ens_mean)
    hx = np.dot(H[j, :], ens_pert)
    varb = np.sum(hx * hx) / (nens - 1)
    varo = R[j, j]
    obs = yo[j]
    rho = GC_func(D[j, :], ROI) ##Gaspari-Cohn
    SRfac = 1.0 / (1.0 + np.sqrt(varo / (varb + varo)))
    cov = np.dot(ens_pert, hx) / (nens - 1)
    K = cov / (varb + varo)
    ens_mean = ens_mean + rho * K * (obs - hxm)
    for k in range(nens):
      ens_pert[:, k] = ens_pert[:, k] - SRfac * rho * K * hx[k]
  for k in range(nens):
    post[:, k] = ens_pert[:, k] + ens_mean
  return post

def H_matrix(nx, obs_ind, t_ind):
  nobs, nt = obs_ind.shape
  H = np.zeros((nobs * t_ind.size, nx * t_ind.size))
  for i in range(t_ind.size):
    for j in range(nobs):
      ind = obs_ind[j, t_ind[i]]
      ##linear interpolation
      ind_left = int(ind) % nx
      ind_right = int(ind + 1) % nx
      shift = ind - int(ind)
      H[i*nobs+j, i*nx+ind_left] = 1 - shift
      H[i*nobs+j, i*nx+ind_right] = shift
  return H

def D_matrix(nx, obs_ind, t_ind, t_ana, time_space_ratio):
  nobs, nt = obs_ind.shape
  D = np.zeros((nobs * t_ind.size, nx * t_ind.size))
  for i in range(t_ind.size):
    for j in range(nobs):
      dist = np.abs(np.arange(0, nx) - obs_ind[j, t_ind[i]])
      dist = np.minimum(dist, nx - dist)
      for k in range(t_ind.size):
        dist_time = np.abs(t_ind[k] - t_ana) * time_space_ratio
        D[i*nobs+j, k*nx:(k+1)*nx] = dist + dist_time
  return D

def R_matrix(nx, obs_ind, t_ind, obs_err, L, Lt):
  nobs, nt = obs_ind.shape
  n = nobs * t_ind.size
  R = np.zeros((n, n))
  for i in range(t_ind.size):
    for j in range(t_ind.size):
      #space component
      ind_i = np.tile(obs_ind[:, t_ind[i]], (nobs, 1))
      ind_j = np.tile(obs_ind[:, t_ind[j]], (nobs, 1)).T
      dist = np.abs(ind_i - ind_j)
      dist = np.minimum(dist, nx - dist)
      if L <= 0:
        corr = np.eye(nobs)
      else:
        corr = np.exp(-dist/L)
      #time component
      tdist = np.abs(i - j)
      if Lt <= 0:
        if tdist > 0:
          corr = corr * 0.0
      else:
        corr = corr * np.exp(-tdist/Lt)
      #block of R:
      R[i*nobs:(i+1)*nobs, j*nobs:(j+1)*nobs] = obs_err**2 * corr
  return R

def GC_func(dist, ROI):
  n = dist.size
  coef = np.zeros(n)
  if ROI <= 0:
    coef[:] = 1.0
  else:
    for i in range(n):
      r = dist[i] / (0.5 * ROI)
      if (r >= 2):
        coef[i] = 0.0
      elif (r >= 1 and r < 2):
        coef[i] = ((((r / 12.0 - 0.5) * r + 0.625) * r + 5.0 / 3.0) * r - 5.0) * r + 4.0 - 2.0 / (3.0 * r)
      else:
        coef[i] = (((-0.25 * r + 0.5) * r + 0.625) * r - 5.0 / 3.0) * (r ** 2) + 1.0
  return coef

def local_matrix(nx, t_ind, ROI, ROIt):
  n = nx * t_ind.size
  rho = np.zeros((n, n))
  for i in range(t_ind.size):
    for j in range(t_ind.size):
      #space component
      ii, jj = np.mgrid[0:nx, 0:nx]
      dist = np.abs(ii - jj)
      dist = np.minimum(dist, nx - dist)
      corr = np.zeros(dist.shape)
      for k in range(nx):
        corr[:, k] = GC_func(dist[:, k], ROI)
      #time component
      corr = corr * GC_func(np.array([np.abs(i-j)]), ROIt)
      #block of rho:
      rho[i*nx:(i+1)*nx, j*nx:(j+1)*nx] = corr
  return rho

def SEC_func(corr, varb, varb1, nens):
  nx = corr.size
  coef = np.zeros(nx)
  r_data = np.load("sec_table.npy")
  for i in range(nx):
    s_ratio = np.sqrt(varb1[i] / varb)
    r = np.floor(corr[i]*100)/100
    k = int(100*(r+1))
    r_mean = r_data[k, 0]
    r_std = np.sqrt(r_data[k, 1])
    b_mean = r_mean * s_ratio
    b_std = r_std * s_ratio
    Q = b_mean / b_std
    coef[i] = Q**2 / (Q**2 + 1)
    # coef[i] = coef[i] * r_mean / corr[i]
  return coef

###adaptive inflation
def adaptive_inflation(xb, ind, yo, obserr):
  nx, nens = xb.shape
  xb_inf = xb.copy()
  xbmean = np.mean(xb, axis=1)
  hxbmean = np.mean(xb[ind, :], axis=1)
  omb = yo - hxbmean
  varb = np.zeros(nx)
  for k in range(nens):
    varb += (xb[ind, k] - hxbmean)**2
  varb = varb/(nens-1)
  varo = obserr**2
  inf_factor = np.maximum(1.0, np.sqrt((np.maximum(0.0, np.mean(omb*omb) - varo))/np.mean(varb)))
  #print('inflation = {}'.format(inf_factor))
  for k in range(nens):
    xb_inf[:, k] = inf_factor*(xb[:, k] - xbmean) + xbmean
  return xb_inf


###Variational methods
def Var4d(prior):
  return post

###Displacement
def optical_flow(f1, f2, nlevel):
  nx, tw = f1.shape
  u = np.zeros((nx, tw))
  q = np.zeros((nx, tw))
  for lev in range(nlevel, -1, -1):
    f1warp = misc.warp(f1, -u, -q)
    f1c = misc.coarsen(f1warp, lev)
    f2c = misc.coarsen(f2, lev)
    niter = 200
    w = 100
    fx = 0.5*(misc.deriv_x(f1c) + misc.deriv_x(f2c))
    ft = 0.5*(misc.deriv_t(f1c) + misc.deriv_t(f2c))
    df = f2c - f1c
    du = np.zeros(fx.shape)
    dq = np.zeros(fx.shape)
    for k in range(niter):
      ubar = misc.laplacian(du) + du
      qbar = misc.laplacian(dq) + dq
      du = ubar - fx*(fx*ubar + ft*qbar + df)/(w + fx**2 + ft**2)
      dq = qbar - ft*(fx*ubar + ft*qbar + df)/(w + fx**2 + ft**2)
      #enforce boundary condition
      dq[:, 0] = 0.0
      dq[:, -1] = 0.0
    u += misc.sharpen(du*2**lev, lev)
    q += misc.sharpen(dq*2**lev, lev)
  return u, q
