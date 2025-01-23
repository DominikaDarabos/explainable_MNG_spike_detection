import tensorflow as tf
import math
import numpy as np
from tensorflow import keras

tf.compat.v1.disable_eager_execution()


def r_hermite_tf(N, mu=0):
    if N <= 0 or mu <= -0.5:
        raise ValueError('parameter(s) out of range')

    m0 = math.gamma(mu + 0.5)
    if N == 1:
        return tf.convert_to_tensor([[0.0, m0]], dtype=tf.float32)

    N = N - 1
    n = tf.range(1, N + 1, dtype=tf.float32)
    nh = 0.5 * n
    indices = tf.range(0, N, 2)
    nh = tf.tensor_scatter_nd_add(nh, tf.expand_dims(indices, axis=1), tf.ones_like(indices, dtype=tf.float32) * mu)

    A = tf.zeros(N + 1)
    B = tf.concat([tf.convert_to_tensor([m0], dtype=tf.float32), nh], axis=0)

    ab = tf.stack([A, B], axis=1)
    return ab

def rec_coeffs_tf(ab, n, mu, dprevmua=None, dprevmub=None):
    if dprevmua is None:
        dprevmua = tf.zeros(0)
    if dprevmub is None:
        dprevmub = tf.zeros(0)

    m = dprevmua.shape[1] if dprevmua.shape[0] > 0 else 0

    abhat = tf.zeros((n, 2))
    damu = tf.zeros((n, m+1))
    dbmu = tf.zeros((n, m+1))

    r = tf.zeros(n+1)
    dr = tf.zeros((n+1, m+1))


    r = tf.tensor_scatter_nd_update(r, [[0]], mu - ab[0, 0])
    r = tf.tensor_scatter_nd_update(r, [[1]], mu - ab[1, 0] - ab[1, 1] / r[0])
    abhat = tf.tensor_scatter_nd_update(abhat, [[0, 0]], [ab[1, 0] + r[1] - r[0]])
    abhat = tf.tensor_scatter_nd_update(abhat, [[0, 1]], [-r[0] * ab[0, 1]])

    dr = tf.tensor_scatter_nd_update(dr, [[0, 0]], [1])
    dr = tf.tensor_scatter_nd_update(dr, [[1, 0]], [1 + ab[1, 1] / (r[0]**2)])
    damu = tf.tensor_scatter_nd_update(damu, [[0, 0]], [dr[1, 0] - dr[0, 0]])
    dbmu = tf.tensor_scatter_nd_update(dbmu, [[0, 0]], [-ab[0, 1]])

    for j in range(1, m + 1):
        dr = tf.tensor_scatter_nd_update(dr, [[0, j]], [-dprevmua[0, j-1]])
        dr = tf.tensor_scatter_nd_update(dr, [[1, j]], [-dprevmua[1, j-1] - (dprevmub[1, j-1] / r[0] - ab[1, 1] * r[0]**(-2) * dr[0, j])])
        damu = tf.tensor_scatter_nd_update(damu, [[0, j]], [dprevmua[1, j-1] + dr[1, j] - dr[0, j]])
        dbmu = tf.tensor_scatter_nd_update(dbmu, [[0, j]], [-(dr[0, j] * ab[0, 1] + r[0] * dprevmub[0, j-1])])

    # Loop over k
    for k in range(1, n):
        r = tf.tensor_scatter_nd_update(r, [[k + 1]], mu - ab[k+1, 0] - ab[k+1, 1] / r[k])
        abhat = tf.tensor_scatter_nd_update(abhat, [[k, 0]], [ab[k+1, 0] + r[k+1] - r[k]])
        abhat = tf.tensor_scatter_nd_update(abhat, [[k, 1]], [ab[k, 1] * r[k] / r[k-1]])

        dr = tf.tensor_scatter_nd_update(dr, [[k + 1, 0]], [1 + ab[k+1, 1] / (r[k]**2) * dr[k, 0]])
        damu = tf.tensor_scatter_nd_update(damu, [[k, 0]], [dr[k+1, 0] - dr[k, 0]])
        dbmu = tf.tensor_scatter_nd_update(dbmu, [[k, 0]], [ab[k, 1] * (dr[k, 0] / r[k-1] - r[k] / (r[k-1]**2) * dr[k-1, 0])])

        for j in range(1, m + 1):
            dr = tf.tensor_scatter_nd_update(dr, [[k + 1, j]], [-dprevmua[k+1, j-1] - (
                dprevmub[k+1, j-1] / r[k] - ab[k+1, 1] * r[k]**(-2) * dr[k, j]
            )])
            damu = tf.tensor_scatter_nd_update(damu, [[k, j]], [dprevmua[k+1, j-1] + dr[k+1, j] - dr[k, j]])
            dbmu = tf.tensor_scatter_nd_update(dbmu, [[k, j]], [(
                dprevmub[k, j-1] * r[k] / r[k-1]
                + ab[k, 1] * dr[k, j] / r[k-1]
                - ab[k, 1] * r[k] / (r[k-1]**2) * dr[k-1, j])
            ])

    return abhat, damu, dbmu

def rec_coeffs_double_root_tf(ab, n, mu, dprevmua, dprevmub):
    if dprevmua is None or dprevmub is None:
        raise ValueError('This requires at least one previous weight modification')

    m = dprevmua.shape[1]

    abhat = tf.zeros((n, 2), dtype=ab.dtype)  # Holder for recurrence coeffs
    damu = tf.zeros((n, m), dtype=ab.dtype)   # Holders for derivatives
    dbmu = tf.zeros((n, m), dtype=ab.dtype)   # Holders for derivatives

    r = tf.zeros(n + 1, dtype=ab.dtype)       # Holder for helper values
    dr = tf.zeros((n + 1, m), dtype=ab.dtype) # ... and corresponding derivatives
    # k = 0
    r = tf.tensor_scatter_nd_update(r, [[0]], mu - ab[0, 0])  # r0
    r = tf.tensor_scatter_nd_update(r, [[1]], mu - ab[1, 0] - ab[1, 1] / r[0])

    # Alpha 0 and Beta 0
    abhat = tf.tensor_scatter_nd_update(abhat, [[0, 0]], [ab[1, 0] + r[1] - r[0]])
    abhat = tf.tensor_scatter_nd_update(abhat, [[0, 1]], [-r[0] * ab[0, 1]])

    # Derivatives w.r.t. mu
    dr = tf.tensor_scatter_nd_update(dr, [[0, 0]], [1 - dprevmua[0, 0]])
    dr = tf.tensor_scatter_nd_update(dr, [[1, 0]], [1 - dprevmua[1, 0] - (dprevmub[1, 0] / r[0] - ab[1, 1] * r[0] ** -2 * dr[0, 0])])
    damu = tf.tensor_scatter_nd_update(damu, [[0, 0]], [dprevmua[1, 0] + dr[1, 0] - dr[0, 0]])
    dbmu = tf.tensor_scatter_nd_update(dbmu, [[0, 0]], [-(r[0] * dprevmub[0, 0] + dr[0, 0] * ab[0, 1])])

    # Derivatives w.r.t. previous modifying terms
    for j in range(1, m):
        dr = tf.tensor_scatter_nd_update(dr, [[0, j]], [-dprevmua[0, j]])
        dr = tf.tensor_scatter_nd_update(dr, [[1, j]], [-dprevmua[1, j] - (dprevmub[1, j] / r[0] - ab[1, 1] * r[0] ** -2 * dr[0, j])])
        damu = tf.tensor_scatter_nd_update(damu, [[0, j]], [dprevmua[1, j] + dr[1, j] - dr[0, j]])
        dbmu = tf.tensor_scatter_nd_update(dbmu, [[0, j]], [-(dr[0, j] * ab[0, 1] + r[0] * dprevmub[0, j])])

    # k >= 1
    for k in range(1, n):
        # Alphas and betas
        r = tf.tensor_scatter_nd_update(r, [[k + 1]], mu - ab[k + 1, 0] - ab[k + 1, 1] / r[k])
        abhat = tf.tensor_scatter_nd_update(abhat, [[k, 0]], [ab[k + 1, 0] + r[k + 1] - r[k]])
        abhat = tf.tensor_scatter_nd_update(abhat, [[k, 1]], [ab[k, 1] * r[k] / r[k - 1]])

        # Derivatives
        dr = tf.tensor_scatter_nd_update(dr, [[k + 1, 0]], [1 - dprevmua[k + 1, 0] - (dprevmub[k + 1, 0] / r[k] - ab[k + 1, 1] * r[k] ** -2 * dr[k, 0])])
        damu = tf.tensor_scatter_nd_update(damu, [[k, 0]], [dprevmua[k + 1, 0] + dr[k + 1, 0] - dr[k, 0]])
        dbmu = tf.tensor_scatter_nd_update(dbmu, [[k, 0]], [dprevmub[k, 0] * r[k] / r[k - 1] + ab[k, 1] * dr[k, 0] / r[k - 1] - ab[k, 1] * r[k] / r[k - 1] ** 2 * dr[k - 1, 0]])

        # Derivatives w.r.t. previous modifying terms
        for j in range(1, m):
            dr = tf.tensor_scatter_nd_update(dr, [[k + 1, j]], [-dprevmua[k + 1, j] - (dprevmub[k + 1, j] / r[k] - ab[k + 1, 1] * r[k] ** -2 * dr[k, j])])
            damu = tf.tensor_scatter_nd_update(damu, [[k, j]], [dprevmua[k + 1, j] + dr[k + 1, j] - dr[k, j]])
            dbmu = tf.tensor_scatter_nd_update(dbmu, [[k, j]], [dprevmub[k, j] * r[k] / r[k - 1] + ab[k, 1] * dr[k, j] / r[k - 1] - ab[k, 1] * r[k] / r[k - 1] ** 2 * dr[k - 1, j]])

    return abhat, damu, dbmu

def orthopoly_tf(ab, dmua, dmub, t):
    n = ab.shape[0]
    N = t.shape[0]
    m = dmua.shape[1]

    tt = tf.reshape(t, [N, 1])
    P = tf.zeros([N, n])
    dP = tf.zeros([N, n])
    dPmu = tf.zeros([m, N, n])

    indices = tf.range(0, N)
    indices_with_zeros = tf.stack([indices, tf.zeros_like(indices)], axis=1)
    indices_with_ones = tf.stack([indices, tf.ones_like(indices)], axis=1)

    P = tf.tensor_scatter_nd_update(P, indices_with_zeros, tf.fill(indices.shape, 1.0))
    P = tf.tensor_scatter_nd_update(P, indices_with_ones, tf.squeeze(tt) - ab[0, 0])
    dP = tf.tensor_scatter_nd_update(dP, indices_with_zeros, tf.fill(indices.shape, 0.0))
    dP = tf.tensor_scatter_nd_update(dP, indices_with_ones, tf.fill(indices.shape, 1.0))

    for j in range(m):
        indices = tf.stack([tf.fill([N], j), tf.range(N), tf.fill([N], 1)], axis=1)
        updates = tf.fill([N], -dmua[0, j])
        dPmu = tf.tensor_scatter_nd_update(dPmu, indices, updates)

    for i in range(1, n - 1):
        indices = tf.stack([tf.range(0, N), tf.fill([N], i+1)], axis=1)
        P = tf.tensor_scatter_nd_update(P, indices, (tf.squeeze(tt) - ab[i, 0]) * P[:, i] - ab[i, 1] * P[:, i - 1])
        dP = tf.tensor_scatter_nd_update(dP, indices, P[:, i] + (tf.squeeze(tt) - ab[i, 0]) * dP[:, i] - ab[i, 1] * dP[:, i - 1])


        for j in range(m):
            indices = tf.stack([tf.fill([N], j), tf.range(N), tf.fill([N], i + 1)], axis=1)
            updates = -dmua[i, j] * P[:, i] + (tf.squeeze(tt) - ab[i, 0]) * dPmu[j, :, i] - dmub[i, j] * P[:, i - 1] - ab[i, 1] * dPmu[j, :, i - 1]
            dPmu = tf.tensor_scatter_nd_update(dPmu, indices, updates)
    return P, dP, dPmu


def weighted_Hermite_system_tf(n, mu, x):
    # Constants and holders
    if not tf.is_tensor(mu):
        mu = tf.convert_to_tensor(mu)
    m = mu.shape[0]
    N = x.shape[0]
    x = tf.reshape(x, (N, 1))

    Phi = tf.zeros((N, n))  # Holder for the weighted Hermite functions
    dPhix = tf.zeros((N, n))  # Holder for derivatives w.r.t. x
    dPhimu = tf.zeros((m, N, n))  # Holder for derivatives w.r.t. mu

    # First generate the weighted orthogonal polynomials and corresponding derivatives
    ab = r_hermite_tf(n + 2 * m)
    damu = None
    dbmu = None

    # Recurrence coeffs for modified system
    for k in range(m-1, -1, -1):
        ab, damu, dbmu = rec_coeffs_tf(ab, n + 2 * (k + 1) - 1, mu[k], damu, dbmu)
        ab, damu, dbmu = rec_coeffs_double_root_tf(ab, n + 2 * (k + 1) - 2, mu[k], damu, dbmu)
    # Calculate Polynomials
    P, dPx, dPmu = orthopoly_tf(ab, damu, dbmu, x)

    val = tf_poly(mu)
    Q = tf.convert_to_tensor(tf.math.polyval(val, x))
    v_sqrt = Q * tf.exp(-x ** 2 / 2)

    Phi = P * v_sqrt

    # Derivatives w.r.t. x
    dQx = tf.zeros((N, 1), dtype=tf.float32)
    for k in range(m):
        mu_curr = tf.stop_gradient(tf.identity(mu))
        mu_curr = tf.concat([mu_curr[:k], mu_curr[k+1:]], axis=0)
        dQx += tf.math.polyval(tf_poly(mu_curr), x)
    
    # Derivatives w.r.t. mu
    dQmu = tf.zeros((N, m), dtype=tf.float32)
    for k in range(m):
        mu_curr = tf.stop_gradient(tf.identity(mu))
        mu_curr = tf.concat([mu_curr[:k], mu_curr[k+1:]], axis=0)
        indices = tf.stack([tf.range(N), tf.fill([N], k)], axis=1)
        dQmu = tf.tensor_scatter_nd_update(dQmu, indices, tf.squeeze(-tf.math.polyval(tf_poly(mu_curr), x)))

    # Derivatives of weighted Hermite functions
    for k in range(n):
        dPhix_k = tf.squeeze(dPx[:, k] * tf.squeeze(v_sqrt) + P[:, k] * tf.transpose((dQx - x * Q) * tf.exp(-x ** 2 / 2)))
        indices = tf.stack([tf.range(N), tf.fill([N], k)], axis=1)
        dPhix = tf.tensor_scatter_nd_update(dPhix, indices, dPhix_k)

        for j in range(m):
            dpmu = tf.squeeze(dPmu[j, :, k])
            dPhimu_jk = tf.squeeze(dpmu * tf.squeeze(v_sqrt) + P[:, k] * tf.transpose(tf.exp(-x ** 2 / 2)) * dQmu[:, j])
            indices = tf.stack([tf.fill([N], j), tf.range(N), tf.fill([N], k)], axis=1)
            dPhimu = tf.tensor_scatter_nd_update(dPhimu, indices, dPhimu_jk)

    return Phi, dPhix, dPhimu

def tf_poly(roots):
    coefficients = tf.constant([1.0], dtype=tf.float32)

    for root in tf.unstack(roots):
        coefficients = tf.concat([coefficients, tf.constant([0.0], dtype=tf.float32)], axis=0)
        coefficients = coefficients - root * tf.pad(coefficients[:-1], [[1, 0]], constant_values=0.0)
    coefficients_list = tf.unstack(coefficients)

    return coefficients_list

def ada_weighted_Hermite_tf(N, n, alpha):
    """
    INPUT:
    n: number of basis functions
    x: domain
    alpha: trainable parameters. These are:
        - alpha[0]: dilation
        - alpha[1]: translation
        - alpha[2:]: weight modifiers (mu)
    OUTPUT:
    Phi: basis functions
    dPhi: derivatives w.r.t. alpha
    Ind: derivative map as usual
    """
    x = tf.linspace(-10.0, 10.0, N)
    dilat = alpha[0]
    trans = alpha[1]
    t = dilat * (x - trans)
    N = tf.shape(t)[0]
    m = len(alpha) - 2
    # Generate functions + derivatives
    Phi, dPhix, dPhimu = weighted_Hermite_system_tf(n, alpha[2:], t)
    # Apply dilation
    sqrt_dilat = tf.sqrt(dilat)
    Phi = sqrt_dilat * Phi
    dPhimu = sqrt_dilat * dPhimu

    # Reshape dPhi and fill up Ind
    dPhi = tf.zeros((N, n * (m + 2)), dtype=Phi.dtype)
    Ind = tf.zeros((2, n * (m + 2)), dtype=tf.int64)

    for k in range(n):  # For every basis function
        # Derivatives w.r.t. dilat and trans
        indices = tf.stack([tf.range(N), tf.fill([N], (k * (m + 2)))], axis=1)
        dPhi = tf.tensor_scatter_nd_update(dPhi, indices, 0.5 * tf.math.pow(dilat, -1) * Phi[:, k] + sqrt_dilat * dPhix[:, k] * (x - trans))
        Ind = tf.tensor_scatter_nd_update(Ind, [[0, k * (m + 2)]], [k])
        Ind = tf.tensor_scatter_nd_update(Ind, [[1, k * (m + 2)]], [0])

        indices = tf.stack([tf.range(N), tf.fill([N], (k * (m + 2) + 1))], axis=1)
        dPhi = tf.tensor_scatter_nd_update(dPhi, indices, -tf.math.pow(dilat, (3 / 2)) * dPhix[:, k])
        Ind = tf.tensor_scatter_nd_update(Ind, [[0, k * (m + 2) + 1]], [k])
        Ind = tf.tensor_scatter_nd_update(Ind, [[1, k * (m + 2) + 1]], [1])

        # Derivatives of the k-th basis function w.r.t. weight modifiers
        for j in range(m):
            indices = tf.stack([tf.range(N), tf.fill([N], (k * (m + 2) + 2 + j))], axis=1)
            dPhi = tf.tensor_scatter_nd_update(dPhi, indices, dPhimu[j, :, k])
            Ind = tf.tensor_scatter_nd_update(Ind, [[0, k * (m + 2) + 2 + j]], [k])
            Ind = tf.tensor_scatter_nd_update(Ind, [[1, k * (m + 2) + 2 + j]], [j + 2])

    return Phi, dPhi, Ind