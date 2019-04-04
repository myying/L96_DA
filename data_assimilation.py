import numpy as np
import misc

##perturbed obs EnKF, full matrix version
def EnKF(xens, ind, yo, obserr, ROI, alpha):
  xens1 = xens.copy()
  nx, nens = xens.shape
  nobs = ind.size
  ###observation, perturbed for each member
  R = np.eye(nobs) * (obserr**2)
  H = np.zeros((nobs, nx))
  for j in range(nobs):
    H[j, ind[j]] = 1.0
  np.random.seed(0)
  obs = np.zeros((nobs, nens))
  for k in range(nens):
    obs[:, k] = yo + np.random.normal(0, obserr, nobs) #perturb observation for members
  ###prior error covariance
  prior = xens.copy()
  post = xens.copy()
  priormean = np.mean(prior, axis=1)
  postmean = np.mean(post, axis=1)
  Pb = misc.error_covariance(prior)
  ###localization
  rho = np.zeros((nx, nx))
  ii, jj = np.mgrid[0:nx, 0:nx]
  dist = np.sqrt((ii - jj)**2)
  dist = np.minimum(dist, nx-dist)
  for i in range(nx):
    rho[:, i] = GC_local_func(dist[:, i], ROI)
  ###Kalman gain
  K = np.dot(np.dot(Pb*rho, H.T), np.linalg.inv(np.dot(np.dot(H, Pb*rho), H.T) + R))
  ###update
  for k in range(nens):
    post[:, k] = prior[:, k] + np.dot(K, (obs[:, k] - np.dot(H, prior[:, k])))
  for k in range(nens):  #relaxation
    xens1[:, k] = alpha*(prior[:, k]-priormean) + (1-alpha)*(post[:, k]-postmean) + postmean
  return xens1

##serial EnKF, assimilates obs one at a time
def EnKF_serial(xens, ind, yo, obserr, ROI, alpha, filter_kind):
  xens1 = xens.copy()
  [nx, nens] = xens.shape
  nobs = ind.size
  xm = np.mean(xens, axis=1)
  x = xens.copy()
  for k in np.arange(nens):
    x[:, k] = x[:, k] - xm
  xb = x.copy()
  for j in np.arange(nobs):
    dist = np.abs(np.arange(nx) - ind[j])
    dist = np.minimum(dist, nx - dist)
    rho = GC_local_func(dist, ROI)
    hxm = xm[ind[j]]
    hx = np.array(x[ind[j], :])
    varb = np.sum(hx * hx) / (nens - 1)
    varo = obserr ** 2
    obs_prior = hx + hxm
    obs = yo[ind[j]]
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
    xens1[:, k] = alpha*xb[:, k] + (1-alpha)*x[:, k] + xm
  return xens1

def obs_inc_EAKF(prior, prior_var, obs, obs_var):
  prior_mean = np.mean(prior)
  var_ratio = obs_var / (prior_var + obs_var)
  new_mean = var_ratio * (prior_mean + obs * prior_var / obs_var)
  obs_inc = np.sqrt(var_ratio) * (prior - prior_mean) + new_mean - prior
  return obs_inc

def EnSRF_spec(xhens, ind, yo, obserr, ROI):
  xhens1 = np.copy(xhens)
  [nk, nens] = xhens.shape
  nx = (nk-1)*2
  nobs = ind.size
  x = np.zeros([nx, nens])
  xm = np.zeros([nx])
  xhm = np.mean(xhens, axis=1)
  xh = xhens
  for k in np.arange(nens):
    xh[:, k] = xh[:, k] - xhm
  # assimilation cycle
  for j in np.arange(nobs):
    # get a copy of state space values
    for k in np.arange(nens):
      x[:, k] = misc.spec2grid(xh[:, k])
    xm = misc.spec2grid(xhm)
    dist = np.abs(np.arange(nx) - ind[j])
    dist = np.minimum(dist, nx - dist)
    hxm = xm[ind[j]]
    hx = np.array(x[ind[j], :])
    varb = np.sum(hx * hx) / (nens - 1)
    varo = obserr ** 2
    SRfac = 1.0 / (1.0 + np.sqrt(varo / (varb + varo)))
    cov = np.array(np.matrix(xh) * np.matrix(hx).T) / (nens - 1)
    K = cov[:, 0] / (varb + varo)
    rho = GC_local_func(dist, ROI)
    rhoh = misc.grid2spec(rho)
    xh = xh - SRfac * np.array(np.matrix(misc.spec_convolve(rhoh, K)).T *
                   np.matrix(hx))
    xhm = xhm + misc.spec_convolve(rhoh, K) * (yo[ind[j]] - hxm)
  for k in np.arange(nens):
    xhens1[:, k] = xh[:, k] + xhm
  return xhens1


def GC_local_func(dist, ROI):
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

###Smoothers
def EnKS_serial(xens, analysis_ind, obs_ind, yo, obserr, ROI, ROIt, alpha):
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
      rhox = GC_local_func(dist, ROI)
      dist = np.abs(np.arange(nt) - t)
      rhot = GC_local_func(dist, ROIt)
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
    xens1[:, k] = alpha*xb[:, k, analysis_ind] + (1-alpha)*x[:, k, analysis_ind] + xm[:, analysis_ind]
  return xens1


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

