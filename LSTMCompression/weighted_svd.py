import sympy
import numpy as np
import contexttimer

__all__ = ['compress']


def optimalAlpha(x0, eps):
    x0symb = sympy.Symbol('x0')
    xsymb = sympy.Symbol('x')
    sigmoidsymb = (sympy.E ** xsymb) / (1 + sympy.E ** xsymb)
    alphasymb = (sigmoidsymb - sigmoidsymb.subs({xsymb: x0symb})) ** 2 / (xsymb - x0symb) ** 2
    alphafunc = sympy.lambdify((xsymb, x0symb), alphasymb)

    x0 = np.array(x0).astype(np.float64).reshape(-1)
    very_small = np.abs(x0) < np.finfo(np.float32).eps * 16
    x0[very_small] = np.finfo(np.float64).eps * 16 * np.sign(x0[very_small])

    high = np.array(np.abs(x0) - np.finfo(np.float64).eps)
    low = -high
    while True:
        mid = (low + high) / 2
        lower_quartile = (low + mid) / 2
        upper_quartile = (high + mid) / 2
        lower_quartile_vals = alphafunc(lower_quartile, x0)
        upper_quartile_vals = alphafunc(upper_quartile, x0)

        # if lower_quartile is higher, then we take the lower, otherwise take higher
        lower_region = lower_quartile_vals > upper_quartile_vals
        upper_region = ~lower_region
        high[lower_region] = upper_quartile[lower_region]
        low[upper_region] = lower_quartile[upper_region]
        if np.linalg.norm(low - high, np.inf) < eps:
            break

    return alphafunc(mid, x0)


def weightedsvd(W, A, rank, err, U=None, V=None):
    if U is None:
        U = np.random.randn(A.shape[0], rank).astype(A.dtype)
    if V is None:
        V = np.random.randn(A.shape[1], rank).astype(A.dtype)
    Wsqrt = np.sqrt(W)
    oldApprox = A.copy()
    with contexttimer.Timer() as timer:
        while True:
            for row_index in range(A.shape[0]):
                Wrow = np.diag(Wsqrt[row_index, :])
                U[row_index, :] = np.linalg.pinv(Wrow @ V) @ (Wrow @ A[row_index, :])
            for row_index in range(A.shape[1]):
                Wrow = np.diag(Wsqrt[:, row_index])
                V[row_index, :] = np.linalg.pinv(Wrow @ U) @ (Wrow @ A[:, row_index])

            newApprox = U @ V.T
            change = np.linalg.norm(oldApprox - newApprox)
            oldApprox = newApprox

            print("Change:", change, "Elapsed:", timer.elapsed)

            if change <= err:
                return U, V


def weightedError(A, B, W):
    return np.sum((A - B) ** 2 * W)


def compress(inputs, weights, bias, rank, block_sizes, eps=1e-8, convergence_movement=1):
    inpt_weights = inputs @ weights
    alphas = optimalAlpha((inpt_weights + bias).flatten(), eps).reshape(inpt_weights.shape)
    submatrices = np.split(inpt_weights, block_sizes, axis=1)
    subalphas = np.split(alphas, block_sizes, axis=1)
    inverse = np.linalg.pinv(inputs)
    new_weights = []
    for submatrix, subalpha in zip(submatrices, subalphas):
        XU, V = weightedsvd(subalpha, submatrix, rank, convergence_movement)
        U = inverse @ XU
        new_weights.append((U, V))
    return new_weights

