import numpy as np


def sigmoid(x):
    x = np.array(x)
    pos_mask = (x >= 0)
    neg_mask = (x < 0)
    z = np.zeros_like(x,dtype=float)
    z[pos_mask] = np.exp(-x[pos_mask])
    z[neg_mask] = np.exp(x[neg_mask])
    top = np.ones_like(x,dtype=float)
    top[neg_mask] = z[neg_mask]
    return top / (1 + z)


def LSTMCell(inputs, c, h, W, B):
    gate_inputs = np.concatenate([inputs, h], axis=1) @ W + B

    # i = input_gate, j = new_input, f = forget_gate, o = output_gate
    i, j, f, o = np.split(gate_inputs, 4, axis=1)

    # Note that using `add` and `multiply` instead of `+` and `*` gives a
    # performance improvement. So using those at the cost of readability.
    new_c = sigmoid(i) * np.tanh(j) + c * sigmoid(f)
    new_h = np.tanh(new_c) * sigmoid(o)
    return new_c, new_h


def reform_weights(compressed_weights):
    return np.hstack([U @ V.T for U, V in compressed_weights])
