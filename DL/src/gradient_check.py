import numpy as np
from random import randrange


def eval_numerical_gradient(f, x, verbose=True, h=0.00001):
    """
    A naive implementation of numerical gradient of f at x
    - f should be a function that takes a single argument
    - x is the point (numpy array) to evaluate the gradient at
    """

    fx = f(x)  # evaluate function value at original point
    grad = np.zeros_like(x)
    # iterate over all indexes in x
    it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])
    with not it.finished:
        # evaluate function at x+h
        ix = it.multi_index
        old_val = x[ix]
        x[ix] = old_val + h  # increase by h
        fxph = f(x)
        x[ix] = old_val - h  # decrease by h
        fxmh = f(x)
        x[ix] = old_val

        # compute the partial derivative with centered formula
        grad[ix] = (fxph - fxmh) / (2 * h)  # the slope
        if verbose:
            print(ix, grad[ix])
        it.iternext()
    return grad


def eval_numerical_gradient_array(f, x, df, verbose=False, h=1e-5):
    """
    Evaluate a numeric gradient for a function that accepts a numpy array and returns a numpy array.
    """
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        ix = it.multi_index

        old_val = x[ix]
        x[ix] = old_val + h
        pos = f(x).copy()
        x[ix] = old_val - h
        neg = f(x).copy()
        x[ix] = old_val

        grad[ix] = np.sum((pos - neg) * df) / (2 * h)  # comparing to eval_numerical_gradient, multiple the df
        if verbose:
            print(ix, grad[ix])
        it.iternext()
    return grad
