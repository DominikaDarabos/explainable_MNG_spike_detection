"""
PyTorch implementation of the weighted Hermite functions.
"""

import torch
import math

def polyval(mu, x, device=None):
    m = len(mu)
    Q = torch.ones_like(x, device=device)
    for i in range(m):
        Q = Q * (x-mu[i])
    return Q

def r_hermite(N, mu=0, device=None):
    if N <= 0 or mu <= -0.5:
        raise ValueError('parameter(s) out of range')

    m0 = math.gamma(mu + 0.5)
    if N == 1:
        return torch.tensor([[0.0, m0]], device=device)

    N = N - 1
    n = torch.arange(1, N + 1, dtype=torch.float32, device=device)
    nh = 0.5 * n
    nh[0::2] = nh[0::2] + mu

    A = torch.zeros(N + 1, device=device)
    B = torch.cat([torch.tensor([m0], device=device), nh])

    ab = torch.stack([A, B], dim=1)
    return ab

def rec_coeffs(ab, n, mu, dprevmua=None, dprevmub=None, device=None):
    # Generate the recurrence coefficients corresponding to the linear modifier (x-mu).
    # The function also returns the partial derivatives of the new a and b coefficients
    # with respect to mu (needed to calculate the polynomial partial derivatives).
    # ASSUMPTION: mu is independent of any variables in dprevmua or dprevmub

    if dprevmua is None:
        dprevmua = torch.empty(0, device=device)
    if dprevmub is None:
        dprevmub = torch.empty(0, device=device)

    # Constants
    m = dprevmua.shape[1] if dprevmua.nelement() > 0 else 0

    abhat = torch.zeros((n, 2), device=device)  # Holder for recurrence coeffs
    damu = torch.zeros((n, m+1), device=device)  # Holders for derivatives
    dbmu = torch.zeros((n, m+1), device=device)

    r = torch.zeros(n+1, device=device)  # Holder for helper values
    dr = torch.zeros((n+1, m+1), device=device)  # ... and corresponding derivatives

    # k = 0
    r[0] = mu - ab[0, 0]  # r0
    r[1] = mu - ab[1, 0] - ab[1, 1] / r[0]

    # Alpha 0 and Beta 0
    abhat[0, 0] = ab[1, 0] + r[1] - r[0]
    abhat[0, 1] = -r[0] * ab[0, 1]

    # Derivatives w.r.t. mu
    dr[0, 0] = 1
    dr[1, 0] = 1 + ab[1, 1] / (r[0]**2)
    damu[0, 0] = dr[1, 0] - dr[0, 0]
    dbmu[0, 0] = -ab[0, 1]

    # Derivatives w.r.t. previous modifying terms
    for j in range(1, m+1):
        dr[0, j] = -dprevmua[0, j-1]
        dr[1, j] = -dprevmua[1, j-1] - (dprevmub[1, j-1] / r[0] - ab[1, 1] * r[0]**(-2) * dr[0, j])
        damu[0, j] = dprevmua[1, j-1] + dr[1, j] - dr[0, j]
        dbmu[0, j] = -(dr[0, j] * ab[0, 1] + r[0] * dprevmub[0, j-1])

    # k >= 1
    for k in range(1, n):
        # Alphas and betas
        r[k+1] = mu - ab[k+1, 0] - ab[k+1, 1] / r[k]
        abhat[k, 0] = ab[k+1, 0] + r[k+1] - r[k]
        abhat[k, 1] = ab[k, 1] * r[k] / r[k-1]

        # Derivatives
        dr[k+1, 0] = 1 + ab[k+1, 1] / (r[k]**2) * dr[k, 0]
        damu[k, 0] = dr[k+1, 0] - dr[k, 0]
        dbmu[k, 0] = ab[k, 1] * (dr[k, 0] / r[k-1] - r[k] / (r[k-1]**2) * dr[k-1, 0])

        # Derivatives w.r.t. previous modifying terms
        for j in range(1, m+1):
            dr[k+1, j] = -dprevmua[k+1, j-1] - (
                dprevmub[k+1, j-1] / r[k] - ab[k+1, 1] * r[k]**(-2) * dr[k, j]
            )
            damu[k, j] = dprevmua[k+1, j-1] + dr[k+1, j] - dr[k, j]
            dbmu[k, j] = (
                dprevmub[k, j-1] * r[k] / r[k-1]
                + ab[k, 1] * dr[k, j] / r[k-1]
                - ab[k, 1] * r[k] / (r[k-1]**2) * dr[k-1, j]
            )

    return abhat, damu, dbmu

def rec_coeffs_double_root(ab, n, mu, dprevmua, dprevmub, device=None):
    # Generate the recurrence coefficients corresponding to the linear modifier (x-mu).
    # The function also returns the partial derivatives of the new a and b coefficients
    # with respect to mu (needed to calculate the polynomial partial derivatives).

    if dprevmua is None or dprevmub is None:
        raise ValueError('This requires at least one previous weight modification')

    # Constants
    m = dprevmua.size(1)

    abhat = torch.zeros((n, 2), device=device)  # Holder for recurrence coeffs
    damu = torch.zeros((n, m), device=device)   # Holders for derivatives
    dbmu = torch.zeros((n, m), device=device)   # Holders for derivatives

    r = torch.zeros(n + 1, device=device)       # Holder for helper values
    dr = torch.zeros((n + 1, m), device=device) # ... and corresponding derivatives

    # k = 0
    r[0] = mu - ab[0, 0]         # r0
    r[1] = mu - ab[1, 0] - ab[1, 1] / r[0]

    # Alpha 0 and Beta 0
    abhat[0, 0] = ab[1, 0] + r[1] - r[0]
    abhat[0, 1] = -r[0] * ab[0, 1]

    # Derivatives w.r.t. mu
    dr[0, 0] = 1 - dprevmua[0, 0]
    dr[1, 0] = 1 - dprevmua[1, 0] - (dprevmub[1, 0] / r[0] - ab[1, 1] * r[0] ** -2 * dr[0, 0])
    damu[0, 0] = dprevmua[1, 0] + dr[1, 0] - dr[0, 0]
    dbmu[0, 0] = -(r[0] * dprevmub[0, 0] + dr[0, 0] * ab[0, 1])

    # Derivatives w.r.t. previous modifying terms
    for j in range(1, m):
        dr[0, j] = -dprevmua[0, j]
        dr[1, j] = -dprevmua[1, j] - (dprevmub[1, j] / r[0] - ab[1, 1] * r[0] ** -2 * dr[0, j])
        damu[0, j] = dprevmua[1, j] + dr[1, j] - dr[0, j]
        dbmu[0, j] = -(dr[0, j] * ab[0, 1] + r[0] * dprevmub[0, j])

    # k >= 1
    for k in range(1, n):
        # Alphas and betas
        r[k + 1] = mu - ab[k + 1, 0] - ab[k + 1, 1] / r[k]
        abhat[k, 0] = ab[k + 1, 0] + r[k + 1] - r[k]
        abhat[k, 1] = ab[k, 1] * r[k] / r[k - 1]

        # Derivatives
        dr[k + 1, 0] = 1 - dprevmua[k + 1, 0] - (dprevmub[k + 1, 0] / r[k] - ab[k + 1, 1] * r[k] ** -2 * dr[k, 0])
        damu[k, 0] = dprevmua[k + 1, 0] + dr[k + 1, 0] - dr[k, 0]
        dbmu[k, 0] = dprevmub[k, 0] * r[k] / r[k - 1] + ab[k, 1] * dr[k, 0] / r[k - 1] - ab[k, 1] * r[k] / r[k - 1] ** 2 * dr[k - 1, 0]

        # Derivatives w.r.t. previous modifying terms
        for j in range(1, m):
            dr[k + 1, j] = -dprevmua[k + 1, j] - (dprevmub[k + 1, j] / r[k] - ab[k + 1, 1] * r[k] ** -2 * dr[k, j])
            damu[k, j] = dprevmua[k + 1, j] + dr[k + 1, j] - dr[k, j]
            dbmu[k, j] = dprevmub[k, j] * r[k] / r[k - 1] + ab[k, 1] * dr[k, j] / r[k - 1] - ab[k, 1] * r[k] / r[k - 1] ** 2 * dr[k - 1, j]

    return abhat, damu, dbmu

def orthopoly(ab, dmua, dmub, t, device=None):
    n = ab.shape[0]  # Number of basis functions
    N = len(t)  # Number of sampling points
    m = dmua.shape[1]  # Number of weight modifier parameters

    tt = t.reshape(N, 1)
    P = torch.zeros(N, n, device=device)
    dP = torch.zeros(N, n, device=device)
    dPmu = torch.zeros(m, N, n, device=device)  # For each weight modifying parameter we have a matrix of derivatives

    P[:, 0] = 1  # zero order orthogonal polynomial
    P[:, 1] = tt.squeeze() - ab[0, 0]  # first order orthogonal polynomial

    dP[:, 0] = 0  # derivative of the zero order orthogonal polynomial
    dP[:, 1] = 1  # derivative of the first order orthogonal polynomial

    dPmu[:, :, 0] = 0  # Derivative of zero order orthogonal polynomial w.r.t. mu
    for j in range(m):
        dPmu[j, :, 1] = -dmua[0, j]  # Derivative of first order orthogonal polynomial w.r.t mu

    # Orthogonal polynomials by recursion
    for i in range(1, n - 1):
        P[:, i + 1] = (tt.squeeze() - ab[i, 0]) * P[:, i] - ab[i, 1] * P[:, i - 1]
        dP[:, i + 1] = P[:, i] + (tt.squeeze() - ab[i, 0]) * dP[:, i] - ab[i, 1] * dP[:, i - 1]

        # Now the partial derivatives w.r.t. the weight modifying terms
        for j in range(m):
            dPmu[j, :, i + 1] = -dmua[i, j] * P[:, i] + (tt.squeeze() - ab[i, 0]) * dPmu[j, :, i] - dmub[i, j] * P[:, i - 1] - ab[i, 1] * dPmu[j, :, i - 1]

    return P, dP, dPmu

def weighted_Hermite_system(n, mu, x, device=None):
    # Constants and holders
    m = len(mu)
    N = len(x)
    x = x.reshape(-1, 1)

    Phi = torch.zeros(N, n, device=device)  # Holder for the weighted Hermite functions
    dPhix = torch.zeros(N, n, device=device)  # Holder for derivatives w.r.t. x
    dPhimu = torch.zeros(m, N, n, device=device)  # Holder for derivatives w.r.t. mu

    # First generate the weighted orthogonal polynomials and corresponding derivatives
    ab = r_hermite(n + 2 * m, device=device)
    damu = None
    dbmu = None

    # Recurrence coeffs for modified system
    for k in range(m - 1, -1, -1):
        ab, damu, dbmu = rec_coeffs(ab, n + 2 * (k + 1) - 1, mu[k], damu, dbmu, device=device)
        ab, damu, dbmu = rec_coeffs_double_root(ab, n + 2 * (k + 1) - 2, mu[k], damu, dbmu, device=device)

    # Calculate Polynomials
    P, dPx, dPmu = orthopoly(ab, damu, dbmu, x, device=device)

    # Calculate root of weight function and its derivatives
    # Construct weight function and needed derivatives
    Q = polyval(mu, x, device=device)
    # Q2 = torch.tensor(np.polyval(np.poly(mu.cpu()), x.cpu()), device=device)
    # torch.testing.assert_close(Q, Q2)
    v_sqrt = Q * torch.exp(-x ** 2 / 2)

    Phi = P * v_sqrt

    # Derivatives w.r.t. x
    dQx = torch.zeros(N, 1, device=device)
    # dQx2 = np.zeros((N, 1))
    for k in range(m):
        mu_curr = mu.clone().detach()
        mu_curr = torch.cat((mu_curr[0:k], mu_curr[k + 1:]))
        dQx += polyval(mu_curr, x, device=device)
        # dQx2 += np.polyval(np.poly(mu_curr.cpu()), x.cpu())
        # torch.testing.assert_close(dQx, torch.tensor(dQx2, dtype=torch.float32, device=device))

    # Derivatives w.r.t. mu
    dQmu = torch.zeros(N, m, device=device)
    # dQmu2 = np.zeros((N, m))
    for k in range(m):
        mu_curr = mu.clone().detach()
        mu_curr = torch.cat((mu_curr[0:k], mu_curr[k + 1:]))
        dQmu[:, k] = torch.squeeze(-polyval(mu_curr, x, device=device))
        # dQmu2[:, k] = np.squeeze(-np.polyval(np.poly(mu_curr.cpu()), x.cpu()))
        # torch.testing.assert_close(dQmu, torch.tensor(dQmu2, dtype=torch.float32, device=device))

    # Derivatives of weighted Hermite functions
    # Derivatives w.r.t. x and mu
    for k in range(n):
        dPhix[:, k] = torch.squeeze(dPx[:, k] * v_sqrt.squeeze() + P[:, k] * ((dQx - x * Q) * torch.exp(-x ** 2 / 2)).T)

        for j in range(m):
            dpmu = torch.squeeze(dPmu[j, :, k])
            dPhimu[j, :, k] = torch.squeeze(dpmu * v_sqrt.squeeze() + P[:, k] * torch.exp(-x ** 2 / 2).T * dQmu[:, j])

    return Phi, dPhix, dPhimu
