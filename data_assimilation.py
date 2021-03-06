import numpy as np
import misc

##filters, full matrix version with perturbed obs
def EnKF_perturbed_obs(prior, yo, H, R, rho, seed):
  np.random.seed(seed)
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

##full matrix EnKF, deterministic version (square root filter)
def EnSRF(prior, yo, H, R, rho):
  nx, nens = prior.shape
  nobs = yo.size
  prior_pert = prior.copy()
  prior_mean = np.mean(prior, axis=1)
  for k in range(nens):
    prior_pert[:, k] = prior[:, k] - prior_mean
  post = prior.copy()
  Pb = misc.error_covariance(prior)
  A = np.dot(np.dot(H, Pb*rho), H.T) + R
  u, s, vh = np.linalg.svd(A)
  Ainv = np.dot(u, np.dot(np.diag(s**-1), vh))
  Asqrt = np.dot(u, np.dot(np.diag(s**0.5), vh))
  Asqrtinv = np.dot(u, np.dot(np.diag(s**-0.5), vh))
  u, s, vh = np.linalg.svd(R)
  Rsqrt = np.dot(u, np.dot(np.diag(s**0.5), vh))
  ARinv = np.linalg.inv(Asqrt + Rsqrt)
  cov = np.dot(Pb*rho, H.T)
  innov = yo - np.dot(H, prior_mean)
  post_mean = prior_mean + np.dot(np.dot(cov, Ainv), innov)
  for k in range(nens):
    post[:, k] = post_mean + prior_pert[:, k] - np.dot(np.dot(cov, np.dot(Asqrtinv.T, ARinv)), np.dot(H, prior_pert[:, k]))
  return post

##serial EnKF (square root filter), assimilates obs one at a time
def EnSRF_serial(prior, obs_prior, obs, obs_err, obs_ind, ROI):
  post = prior.copy()
  obs_post = obs_prior.copy()
  nx, nens = prior.shape
  nobs = obs.size
  mean = np.mean(post, axis=1)
  pert = misc.ens_pert(post)
  obs_mean = np.mean(obs_post, axis=1)
  obs_pert = misc.ens_pert(obs_post)
  for j in range(nobs):
    varb = np.sum(obs_pert[j, :] * obs_pert[j, :]) / (nens - 1)
    varo = obs_err ** 2
    SRfac = 1.0 / (1.0 + np.sqrt(varo / (varb + varo)))
    #update state
    rho = local_func(nx, np.arange(nx), obs_ind[j], ROI) #localization
    K_gain = np.dot(pert, obs_pert[j, :]) / (nens - 1) / (varb + varo)
    mean = mean + rho * K_gain * (obs[j] - obs_mean[j])
    for k in range(nens):
      pert[:, k] = pert[:, k] - SRfac * rho * K_gain * obs_pert[j, k]
    #update obs_state
    K_gain = np.dot(obs_pert, obs_pert[j, :]) / (nens - 1) / (varb + varo)
    obs_mean = obs_mean + rho * K_gain * (obs[j] - obs_mean[j])
    for k in range(nens):
      obs_pert[:, k] = obs_pert[:, k] - SRfac * rho * K_gain * obs_pert[j, k]
  for k in range(nens):
    post[:, k] = pert[:, k] + mean
  return post

def H_matrix(nx, obs_ind, t_ind, smooth):
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
      ##smoothing
      if smooth > 0:
        dist = np.abs(np.arange(nx) - ind)
        dist = np.minimum(dist, nx-dist)
        weight = np.zeros(nx)
        weight = np.exp(-(dist/smooth)**2)
        # weight[np.where(dist < smooth)] = 1.0
        weight = weight / np.sum(weight)
        H[i*nobs+j, i*nx:(i+1)*nx] = weight
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
        corr = np.exp(-1.0*dist/L)
      #time component
      tdist = np.abs(i - j)
      if Lt <= 0:
        if tdist > 0:
          corr = corr * 0.0
      else:
        corr = corr * np.exp(-1.0*tdist/Lt)
      #block of R:
      R[i*nobs:(i+1)*nobs, j*nobs:(j+1)*nobs] = obs_err**2 * corr
      R = R + np.eye(n) * 1e-8  ##avoid singularity
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

def local_func(nx, ind, obs_ind, ROI):
  coef = np.zeros(ind.shape)
  dist = np.abs(ind - obs_ind)
  dist = np.minimum(dist, nx - dist)
  coef = GC_func(dist, ROI)
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
def adaptive_prior_inflation(yo, yb, obserr):
  ny, nens = yb.shape
  ybmean = np.mean(yb, axis=1)
  for k in range(nens):
    yb[:, k] = yb[:, k] - ybmean
  omb2 = np.mean((yo - ybmean)**2)
  varb = np.mean(np.sum(yb**2, axis=1)/(nens-1))
  varo = obserr**2
  if(omb2-varo < 0):
    inflate_coef = 1.0
  else:
    inflate_coef = np.sqrt((omb2-varo)/varb)
  # print('omb2=', omb2, ' varb=', varb)
  # print('lambda=', inflate_coef)
  return inflate_coef

def adaptive_relaxation(yo, yb, ya, obserr):
  ny, nens = yb.shape
  ybmean = np.mean(yb, axis=1)
  yamean = np.mean(ya, axis=1)
  for k in range(nens):
    yb[:, k] = yb[:, k] - ybmean
    ya[:, k] = ya[:, k] - yamean
  omaamb = np.mean((yo - yamean)*(yamean - ybmean))
  varb = np.mean(np.sum(yb**2, axis=1)/(nens-1))
  vara = np.mean(np.sum(ya**2, axis=1)/(nens-1))
  varo = obserr**2
  if(omaamb < 0):
    inflate_coef = 1.0
  else:
    inflate_coef = np.sqrt(omaamb/vara)
  beta = np.sqrt(varb/vara)
  relax_coef = (inflate_coef - 1)/(beta - 1)
  if(relax_coef>2):
    relax_coef = 2.0
  if(relax_coef<-1):
    relax_coef = -1.0
  if(beta < 1):
    relax_coef = 0.0
  # print('omaamb=', omaamb, ' varb=', varb, ' vara=', vara)
  # print('alpha=', relax_coef)
  return relax_coef

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
