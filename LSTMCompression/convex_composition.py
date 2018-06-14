import numpy as np

import contexttimer
import itertools
import scipy.linalg as splinalg

from .utils import sigmoid

__all__ = ['compress']


def computeGradientAB(A, B):
    # computes gradient of AB in terms of B.
    # A is mxk, B is kxn.
    # The gradient is then an m x n x k x n Tensor
    m, k = A.shape
    _, n = B.shape
    print(m * n * k * n)
    gradient = np.zeros((m, n, k, n))

    for nind, kind in np.ndindex(n, k):
        gradient[:, nind, kind, nind] = A[:, kind]

    return gradient.reshape((m * n, k * n))


def computeGradientABC(A, B, C):
    # computes gradient of ABC in terms of vec(B).
    # A is mxk, B is kxn, and C is nxl
    # We have d(ABC_ad)/d(B_bc) = A_ab C_cd
    m, n = A.shape
    k, l = C.shape
    print(m * n * k * l)
    grad = np.einsum("ab, cd -> adbc", A, C)
    return grad.reshape((m * l, k * n))


def compress(Xs, weights, bias, rank, elements_per_batch, snapshot_num=None):

    # Chosen Line Search Parameters
    alpha = 0.1
    beta = 0.5

    print("Setting up Problem")
    if snapshot_num is not None:
        f = np.load("snapshot_{}.npz".format(snapshot_num))
        U = f.get('arr_0')
        V = f.get('arr_1')
    else:
        # Seed the initial U, V with the SVD.
        Ufull, Sfull, Vfull = np.linalg.svd(weights, full_matrices=False)
        U = Ufull[:, :rank] @ np.diag(Sfull[:rank])
        V = Vfull[:rank, :]
        np.savez("snapshot_SVD.npz", U, V)

    for iteration in itertools.count(snapshot_num or 0):
        # Due to memory constraints, we must sub-sample the inputs.
        X = Xs[np.random.choice(Xs.shape[0], elements_per_batch)]
        print("Starting Iteration", iteration)
        gold = sigmoid(X @ weights + bias)
        current_guess = sigmoid(X @ U @ V + bias)
        diff = current_guess - gold
        current_value = np.linalg.norm(diff, 'fro') ** 2
        print("Current Status:", current_value)

        # applying the gradient of the sigmoid
        sig_gradient = (current_guess * (1 - current_guess)).ravel()

        with contexttimer.Timer() as tv:
            partialGradV = (sig_gradient * computeGradientAB(X @ U, V).T).T
        print("GradV:", tv.elapsed)
        print(np.linalg.norm(partialGradV, np.inf))
        with contexttimer.Timer() as tu:
            partialGradU = (sig_gradient * computeGradientABC(X, U, V).T).T
        print("GradU:", tu.elapsed)
        print(np.linalg.norm(partialGradU, np.inf))
        # stack into one large matrix
        stacked = np.concatenate([partialGradU, partialGradV], axis=1)

        with contexttimer.Timer() as tup:
            print("Starting LSTSQ")
            assert np.isfinite(diff).all()
            assert np.isfinite(stacked).all()
            update = splinalg.lstsq(stacked, -diff.ravel())[0]
            print("Finished LSTSQ")

            expected_value = np.linalg.norm(diff.ravel() + stacked @ update) ** 2
            expected_improvement = expected_value - current_value  # This is negative, as we expect an improvement
            assert expected_improvement < 0
            print("Expected Improvement:", expected_improvement)
            t = 1
            while True:
                Uupdate = t * update[:U.size].reshape(U.shape)
                Vupdate = t * update[U.size:].reshape(V.shape)
                new_value = np.linalg.norm(sigmoid(X @ (U + Uupdate) @ (V + Vupdate) + bias) - gold, 'fro') ** 2

                if new_value <= current_value + alpha * t * expected_improvement:
                    print("Line search T:", t)
                    print("New Value:", new_value)
                    step_size = 1 / np.sqrt(iteration + 1)
                    U += step_size * Uupdate
                    V += step_size * Vupdate
                    np.savez("snapshot_{}.npz".format(iteration), U, V)
                    if np.linalg.norm(t * step_size * update) <= 1e-5:
                        return U, V
                    break
                t *= beta

        print("Update:", tup.elapsed)
