import numpy as np
import scipy as sp

import itertools

from .convex_composition import computeGradientAB, computeGradientABC
from .utils import sigmoid, LSTMCell_prestacked, scale_rows


def rank_factorize(W, rank):
    """
    :param W: Matrix to be compressed
    :param rank: Rank of factorization
    :return: U, V low rank factors such that UV = W
    """
    U, S, Vt = sp.linalg.svd(W, full_matrices=False)
    return U[:, :rank] @ np.diag(S[:rank]), Vt[:rank, :]


def low_rank_LSTMCell(inputs, C, UVs, B):
    i, j, f, o = [inputs @ U @ V for U, V in UVs]
    new_c = sigmoid(i) * np.tanh(j) + C * sigmoid(f)
    new_h = np.tanh(new_c) * sigmoid(o)
    return new_c, new_h


def compress(Xs, C, W, B, ranks, alpha, beta, sample_size):
    """
    :param Xs: Stacked row-input matrix
    :param C: Carried input matrix
    :param W: Weight Matrix
    :param B: Bias Vector
    :param ranks: Compression rank of U, V
    :param alpha: Line search improvement parameter
    :param beta: Line search shrinkage parameter
    :param sample_size: sampling rate in X
    :return:
    """
    # compute gold values
    # We express everything in terms of vectorized forms

    Ws = np.split(W, 4, axis=1)
    Bs = np.split(B, 4)
    UVs = [rank_factorize(Wi, rank_i) for Wi, rank_i in zip(Ws, ranks)]
    functions = [sigmoid, np.tanh, sigmoid, sigmoid]
    for iteration in itertools.count(start=1):

        X = Xs[np.random.choice(Xs.shape[0], sample_size)]
        gold_c, gold_h = LSTMCell_prestacked(X, C, W, B)
        gold = np.concatenate([gold_c.ravel(), gold_h.ravel()])

        i, j, f, o = [func(X @ U @ V + b) for func, (U, V), b in zip(functions, UVs, Bs)]
        computed_c = i * j + C * f
        tanh_c = np.tanh(computed_c)
        computed_h = tanh_c * o
        computed_vec = np.concatenate([computed_c.ravel(), computed_h.ravel()])
        diff = computed_vec - gold
        current_value = np.linalg.norm(diff)**2
        print("Current difference:", current_value)

        # c_scales only exist for i, j, f
        # for o we have a separate computation that's only in h_scales
        c_scales = [
            (i * (1 - i) * j).ravel(),
            i * (1 - j**2).ravel(),
            C * f * (1 - f).ravel()
        ]
        h_scales = [
            c_scale * (1 - tanh_c**2) for c_scale in c_scales
        ] + [o * (1 - o)]

        gradients = [(computeGradientABC(X, U, V), computeGradientAB(X @ U, V)) for U, V in UVs]

        # this will not include the gradient of o
        c_scaled_gradients = [[scale_rows(gradU, c_scale), scale_rows(gradV, c_scale)] for
                              (gradU, gradV), c_scale in zip(gradients, c_scales)]
        h_scaled_gradients = [[scale_rows(gradU, c_scale), scale_rows(gradV, c_scale)] for
                              (gradU, gradV), c_scale in zip(gradients, h_scales)]

        full_gradient = np.block([c_scaled_gradients + [np.zeros_like(h_scaled_gradients[-1])],
                                  h_scaled_gradients])

        update = sp.linalg.lstsq(full_gradient, -diff)

        print("Update Size:", np.linalg.norm(update)**2)
        if np.linalg.norm(update)**2 < 1e-5:
            return UVs

        expected_value = np.linalg.norm(diff + full_gradient @ update)**2
        expected_improvement = expected_value - current_value  # should be negative

        # perform line search
        t = 1
        while t > 1e-5:
            scaled_update = t * update
            UVupdates = np.split(scaled_update, 4) # contains the update for the UV vector for i, j, f, o
            new_UVs = [(U + UVupdate[:U.size].reshape(U.shape), V + UVupdates[-V.size:].reshape(V.shape)) for
                       (U, V), UVupdate in zip(UVs, UVupdates)]
            new_c, new_h = low_rank_LSTMCell(X, C, new_UVs, B)
            candidate = np.concatenate([new_c.ravel(), new_h.ravel()])
            new_value = np.linalg.norm(candidate - gold)**2

            if new_value <= current_value + alpha * t * expected_improvement:
                iteration_scale_factor = 1 / np.sqrt(iteration)

                print("New Value:", new_value)
                UVs = [(U + iteration_scale_factor*UVupdate[:U.size].reshape(U.shape),
                        V + iteration_scale_factor*UVupdates[-V.size:].reshape(V.shape)) for
                       (U, V), UVupdate in zip(UVs, UVupdates)]
                break
            t *= beta

