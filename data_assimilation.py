import numpy as np
import misc

def R_matrix(nx, nt, obs_ind, obs_err, L, Lt, corr_kind):
  nobs = obs_ind.size
  R = np.zeros((nobs*nt, nobs*nt))
  ii, jj = np.mgrid[0:nx, 0:nx]
  dist = np.sqrt((ii[obs_ind, :][:, obs_ind]-jj[obs_ind, :][:, obs_ind])**2)
  dist = np.minimum(dist, nx-dist)
  if L <= 0:
    corr = np.eye(nobs)
  else:
    if corr_kind == 1:
      corr = np.exp(-dist/L)
    if corr_kind == 2:
      corr = np.cos(np.pi*dist/L) * np.exp(-dist/L)
    else:
      corr = np.exp(-dist/L)
  if Lt <= 0:
    for t in range(nt):
      R[t*nobs:(t+1)*nobs, :][:, t*nobs:(t+1)*nobs] = obs_err**2 * corr
  else:
    for t in range(nt):
      for s in range(nt):
        tdist = np.abs(t-s)
        R[t*nobs:(t+1)*nobs, :][:, s*nobs:(s+1)*nobs] = obs_err**2 * corr * np.exp(-tdist/Lt)
  return R

def H_matrix(nx, nt, obs_ind):
  nobs = obs_ind.size
  H = np.zeros((nobs*nt, nx*nt))
  for t in range(nt):
    for j in range(nobs):
      H[t*nobs+j, t*nx+obs_ind[j]] = 1.0
  return H

def GC_func(dist, ROI):
  nx = dist.size
  coef = np.zeros(nx)
  if ROI <= 0:
    coef[:] = 1.0
  else:
    for i in np.arange(nx):
      r = dist[i] / (0.5 * ROI)
      if (r >= 2):
        coef[i] = 0.0
      elif (r >= 1 and r < 2):
        coef[i] = ((((r / 12.0 - 0.5) * r + 0.625) * r + 5.0 / 3.0) *
               r - 5.0) * r + 4.0 - 2.0 / (3.0 * r)
      else:
        coef[i] = (((-0.25 * r + 0.5) * r + 0.625) * r -
               5.0 / 3.0) * (r ** 2) + 1.0
  return coef

def local_matrix(nx, nt, ROI, ROIt):
  rho = np.zeros((nx*nt, nx*nt))
  ii, jj = np.mgrid[0:nx, 0:nx]
  dist = np.sqrt((ii - jj)**2)
  dist = np.minimum(dist, nx - dist)
  rhoblock = dist.copy()
  for i in range(nx):
    rhoblock[:, i] = GC_func(dist[:, i], ROI)
  for t in range(nt):
    for s in range(nt):
      tdist = np.abs(t-s)
      rho[t*nx:(t+1)*nx, :][:, s*nx:(s+1)*nx] = rhoblock * GC_func(np.array([tdist]), ROIt)
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

##filters, full matrix version with perturbed obs
def EnKF(prior, yo, H, R, rho):
  post = prior.copy()
  nx, nens = prior.shape
  nobs = yo.size
  obs = np.zeros((nobs, nens))
  for k in range(nens):
    obs[:, k] = yo + np.random.multivariate_normal(np.zeros(nobs), R)
  Pb = misc.error_covariance(prior)
  K = np.dot(np.dot(Pb*rho, H.T), np.linalg.inv(np.dot(np.dot(H, Pb*rho), H.T) + R))
  for k in range(nens):
    post[:, k] = prior[:, k] + np.dot(K, (obs[:, k] - np.dot(H, prior[:, k])))
  return post

##serial EnKF, assimilates obs one at a time
def EnKF_serial(prior, yo, H, R, rho1, filter_kind):
  post = prior.copy()
  nx, nens = prior.shape
  nobs = yo.size
  prior_mean = np.mean(prior, axis=1)
  x = xens.copy()
  for k in np.arange(nens):
    x[:, k] = x[:, k] - xm
  xb = x.copy()
  for j in np.arange(nobs):
    hxm = xm[ind[j]]
    hx = np.array(x[ind[j], :])
    varb = np.sum(hx * hx) / (nens - 1)
    varo = obserr ** 2
    obs_prior = hx + hxm
    obs = yo[ind[j]]
    #localization
    # dist = np.abs(np.arange(nx) - ind[j])
    # dist = np.minimum(dist, nx - dist)
    # rho = GC_func(dist, ROI) ##Gaspari-Cohn
    # cov = np.array(np.matrix(x) * np.matrix(hx).T) / (nens - 1)
    # varb1 = np.sum(x**2, axis=1) / (nens-1)
    # corr = cov[:, 0] / np.sqrt(varb * varb1)
    # rho = SEC_func(corr, varb, varb1, nens)  ##Sampling Error Correction (Anderson 2012)
    # print(corr)
    # print(rho)
    if filter_kind == 1:  #EAKF
      obs_inc = obs_inc_EAKF(obs_prior, varb, obs, varo)
      for i in np.arange(nx):
        state = x[i, :] + xm[i]
        obs_state_cov = np.sum(x[i, :]*hx)/(nens-1)
        state = state + rho[i] * obs_state_cov / varb * obs_inc
        new_mean = np.mean(state)
        state_pert = state - new_mean
        x[i, :] = state_pert
        xm[i] = new_mean
    if filter_kind == 2:  #EnSRF
      SRfac = 1.0 / (1.0 + np.sqrt(varo / (varb + varo)))
      cov = np.array(np.matrix(x) * np.matrix(hx).T) / (nens - 1)
      K = cov[:, 0] / (varb + varo)
      x = x - SRfac * np.array(np.matrix(rho * K).T * np.matrix(hx))
      xm = xm + rho * K * (obs - hxm)
  for k in np.arange(nens):
    xens1[:, k] = x[:, k] + xm
  return xens1

def obs_inc_EAKF(prior, prior_var, obs, obs_var):
  prior_mean = np.mean(prior)
  var_ratio = obs_var / (prior_var + obs_var)
  new_mean = var_ratio * (prior_mean + obs * prior_var / obs_var)
  obs_inc = np.sqrt(var_ratio) * (prior - prior_mean) + new_mean - prior
  return obs_inc

# def EnSRF_spec(xens, ind, yo, obserr, ROI):  ##old version
#   xens1 = np.copy(xens)
#   nx, nens = xens.shape
#   nk = int(nx/2+1)
#   xhens = np.zeros([nk, nens], dtype=complex)
#   for k in range(nens):
#     xhens[:, k] = misc.grid2spec(xens[:, k])
#   nobs = ind.size
#   x = np.zeros([nx, nens])
#   xm = np.zeros([nx])
#   xhm = np.mean(xhens, axis=1)
#   xh = xhens
#   for k in np.arange(nens):
#     xh[:, k] = xh[:, k] - xhm
#   # assimilation cycle
#   for j in np.arange(nobs):
#     # get a copy of state space values
#     for k in np.arange(nens):
#       x[:, k] = misc.spec2grid(xh[:, k])
#     xm = misc.spec2grid(xhm)
#     dist = np.abs(np.arange(nx) - ind[j])
#     dist = np.minimum(dist, nx - dist)
#     hxm = xm[ind[j]]
#     hx = np.array(x[ind[j], :])
#     varb = np.sum(hx * hx) / (nens - 1)
#     varo = obserr ** 2
#     SRfac = 1.0 / (1.0 + np.sqrt(varo / (varb + varo)))
#     cov = np.array(np.matrix(xh) * np.matrix(hx).T) / (nens - 1)
#     K = cov[:, 0] / (varb + varo)
#     rho = GC_func(dist, ROI)
#     rhoh = misc.grid2spec(rho)
#     xh = xh - SRfac * np.array(np.matrix(misc.spec_convolve(rhoh, K)).T * np.matrix(hx))
#     xhm = xhm + misc.spec_convolve(rhoh, K) * (yo[ind[j]] - hxm)
#   for k in np.arange(nens):
#     xhens[:, k] = xh[:, k] + xhm
#     xens1[:, k] = misc.spec2grid(xhens[:, k])
#   return xens1

# def EnSRF_specspec(xens, ind, yo, obserr, L, corr_kind):
#   xens1 = np.copy(xens)
#   nx, nens = xens.shape
#   v = misc.fourier_basis(nx)
#   xhens = np.dot(v.T, xens)
#   xm = np.mean(xhens, axis=1)
#   x = xhens
#   for k in np.arange(nens):
#     x[:, k] = x[:, k] - xm
#   obs = np.dot(v.T, yo)
#   nobs = obs.size
#   R = R_matrix(nx, 1, ind, obserr, L, 0, corr_kind)
#   wo = np.dot(v.T, np.dot(R, v))
#   # assimilation cycle
#   for j in np.arange(nobs):
#     hx = x[j, :]  ###all states observed
#     hxm = xm[j]
#     varb = np.sum(hx**2) / (nens - 1)
#     varo = wo[j, j]
#     SRfac = 1.0 / (1.0 + np.sqrt(varo / (varb + varo)))
#     cov = np.dot(x, hx) / (nens-1)
#     varb1 = np.sum(x**2, axis=1) / (nens-1)
#     corr = cov / np.sqrt(varb*varb1)
#     K = cov / (varb + varo)
#     rho = SEC_func(corr, varb, varb1, nens)
#     # rho = np.ones(nx)
#     for k in range(nens):
#       x[:, k] = x[:, k] - SRfac * rho * K * hx[k]
#     xm = xm + rho * K * (obs[j] - hxm)
#     for k in range(nens):
#       xens1[:, k] = np.dot(v, x[:, k]+xm)
#   return xens1

# def EnSRF_multiscale(xens, ind, yo, obserr, ROI, krange):
#   xens1 = xens.copy()
#   ns = krange.size + 1
#   nobs = ind.size
#   nx, nens = xens.shape
#   x_ms = np.zeros([nx, ns, nens])
#   xm_ms = np.zeros([nx, ns])
#   for s in range(ns):
#     for k in range(nens):
#       x_ms[:, s, k] = misc.spec_bandpass(xens[:, k], krange, s)
#     xm_ms[:, s] = np.mean(x_ms[:, s, :], axis=1)
#     for k in range(nens):
#       x_ms[:, s, k] = x_ms[:, s, k] - xm_ms[:, s]
#   for s in range(ns):
#     for j in range(nobs):
#       dist = np.abs(np.arange(nx) - ind[j])
#       dist = np.minimum(dist, nx - dist)
#       rho = GC_func(dist, ROI[s])
#       hxm = np.sum(xm_ms[ind[j], :])
#       hx = np.sum(x_ms[ind[j], :, :], axis=0)
#       varb = np.sum(hx * hx) / (nens - 1)
#       varo = obserr[s] ** 2
#       obs_prior = hx + hxm
#       obs = yo[ind[j]]
#       SRfac = 1.0 / (1.0 + np.sqrt(varo / (varb + varo)))
#       cov = np.array(np.matrix(x_ms[:, s, :]) * np.matrix(hx).T) / (nens - 1)
#       K = cov[:, 0] / (varb + varo)
#       x_ms[:, s, :] = x_ms[:, s, :] - SRfac * np.array(np.matrix(rho * K).T * np.matrix(hx))
#       xm_ms[:, s] = xm_ms[:, s] + rho * K * (obs - hxm)
#   for k in range(nens):
#     xens1[:, k] = np.sum(x_ms[:, :, k] + xm_ms, axis=1)
#   return xens1

###Smoothers
def EnKS(xens, analysis_ind, obs_ind, yo, obserr, L, Lt, ROI, ROIt):
  nx, nens, nt = xens.shape
  nobs = obs_ind.size
  ##time-space state
  prior = np.zeros((nx*nt, nens))
  for t in range(nt):
    prior[t*nx:(t+1)*nx, :] = xens[:, :, t]
  ###observation, perturbed for each member
  R = np.zeros((nobs*nt, nobs*nt))
  ii, jj = np.mgrid[0:nx, 0:nx]
  dist = np.sqrt((ii[obs_ind, :][:, obs_ind]-jj[obs_ind, :][:, obs_ind])**2)
  dist = np.minimum(dist, nx-dist)
  if L <= 0:
    corr = np.eye(nobs)
  else:
    corr = np.exp(-dist/L)
  if Lt <= 0:
    for t in range(nt):
      R[t*nobs:(t+1)*nobs, :][:, t*nobs:(t+1)*nobs] = obserr**2 * corr
  else:
    for t in range(nt):
      for s in range(nt):
        tdist = np.abs(t-s)
        R[t*nobs:(t+1)*nobs, :][:, s*nobs:(s+1)*nobs] = obserr**2 * corr * np.exp(-tdist/Lt)
  H = np.zeros((nobs*nt, nx*nt))
  for t in range(nt):
    for j in range(nobs):
      H[t*nobs+j, t*nx+obs_ind[j]] = 1.0
  np.random.seed(0)
  #perturb observation for members
  obs0 = np.zeros((nobs*nt))
  for t in range(nt):
    obs0[t*nobs:(t+1)*nobs] = yo[:, t]
  obs = np.zeros((nobs*nt, nens))
  for k in range(nens):
    obs[:, k] = obs0 + np.random.multivariate_normal(np.zeros(nobs*nt), R)
  ###prior error covariance
  priormean = np.mean(prior, axis=1)
  Pb = misc.error_covariance(prior)
  ###localization
  rho = np.zeros((nx*nt, nx*nt))
  ii, jj = np.mgrid[0:nx, 0:nx]
  dist = np.sqrt((ii - jj)**2)
  dist = np.minimum(dist, nx-dist)
  rhoblock = dist.copy()
  for i in range(nx):
    rhoblock[:, i] = GC_func(dist[:, i], ROI)
  for t in range(nt):
    for s in range(nt):
      tdist = np.abs(t-s)
      rho[t*nx:(t+1)*nx, :][:, s*nx:(s+1)*nx] = rhoblock * GC_func(np.array([tdist]), ROIt)
  ###Kalman gain
  K = np.dot(np.dot(Pb*rho, H.T), np.linalg.inv(np.dot(np.dot(H, Pb*rho), H.T) + R))
  ###update
  post = prior.copy()
  for k in range(nens):
    post[:, k] = prior[:, k] + np.dot(K, (obs[:, k] - np.dot(H, prior[:, k])))
  return post[analysis_ind*nx:(analysis_ind+1)*nx, :]

def EnKS_serial(xens, analysis_ind, obs_ind, yo, obserr, ROI, ROIt):
  nx, nens, nt = xens.shape
  nobs, nt = yo.shape
  xens1 = xens[:, :, analysis_ind].copy()
  xm = np.mean(xens, axis=1)
  x = xens.copy()
  for k in np.arange(nens):
    x[:, k, :] = x[:, k, :] - xm
  xb = x.copy()
  for t in np.arange(nt):
    for j in np.arange(nobs):
      dist = np.abs(np.arange(nx) - obs_ind[j])
      dist = np.minimum(dist, nx - dist)
      rhox = GC_func(dist, ROI)
      dist = np.abs(np.arange(nt) - t)
      rhot = GC_func(dist, ROIt)
      rho = np.array(np.dot(np.matrix(rhox).T, np.matrix(rhot)))
      hxm = xm[obs_ind[j], t]
      hx = np.array(x[obs_ind[j], :, t])
      varb = np.sum(hx * hx) / (nens - 1)
      varo = obserr ** 2
      obs_prior = hx + hxm
      obs = yo[obs_ind[j], t]
      SRfac = 1.0 / (1.0 + np.sqrt(varo / (varb + varo)))
      cov = np.zeros((nx, nt))
      for k in np.arange(nens):
        cov += x[:, k, :] * hx[k]
      cov = cov/(nens - 1)
      K = cov / (varb + varo)
      for k in np.arange(nens):
        x[:, k, :] = x[:, k, :] - SRfac * rho * K * hx[k]
      xm = xm + rho * K * (obs - hxm)
  for k in np.arange(nens):
    xens1[:, k] = x[:, k, analysis_ind] + xm[:, analysis_ind]
  return xens1

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


###Displacement
def optical_flow(f1, f2, nlevel):
  nx, nt = f1.shape
  u = np.zeros((nx, nt))
  q = np.zeros((nx, nt))
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
