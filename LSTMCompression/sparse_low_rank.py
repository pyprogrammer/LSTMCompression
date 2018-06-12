import numpy as np


def slrm(C, gamma, beta):
    A = np.zeros_like(C)
    B = np.copy(C)
    Z = np.zeros_like(C)

    while True:
        # AStep
        tmp = 1 / beta * Z - B + C
        A = tmp - np.clip(tmp, -gamma / beta, gamma / beta)

        # BStep
        U, S, VT = np.linalg.svd(C - A + 1 / beta * Z, full_matrices=False)
        Snew = np.maximum(S - 1 / beta, 0)
        Brank = np.count_nonzero(Snew)
        B = U @ np.diag(Snew) @ VT

        # ZStep
        Z = Z - beta * (A + B - C)
        yield A, B, Brank