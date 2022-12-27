import numpy as np


def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N examples,
    where each example x[i] has shape (d_1, ..., d_k).
    We will reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    :param x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    :param w: A numpy array of weights, of shape (D, M).
    :param b: A numpy array of biases, of shape (M,)
    :return:
        - out: output, of shape (N, M)
        - cache: (x, w, b)
    """
    out = None

    N = x.shape[0]
    x_rsp = x.reshape(N, -1)
    out = np.dot(x_rsp, w) + b

    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    :param dout: Upstream derivative, of shape (N, M), and dout = derivative of dl/df
    :param cache: Tuple of:
        - x: Input data, of shape (N, d_1, ... d_k)
        - w: Weights, of shape (D, M)
        - b: Biases, of shape (M,)
    :return: Tuple of:
        - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
        - dw: Gradient with respect to w, of shape (D, M)
        - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None

    N = x.shape[0]
    x_rsp = x.reshape(N, -1)
    dx = dout.dot(w.T)
    # dx shape equal to x
    dx = dx.reshape(*x.shape)
    dw = x_rsp.T.dot(dout)
    db = np.sum(dout, axis=0)

    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (RELUs)

    :param x: Inputs, of any shape
    :return:
        - out: Output, of the same shape as x
        - cache: x
    """
    out = np.maximum(0, x)  # relu function

    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    :param dout: Upstream derivatives, of any shape
    :param cache: Input x, of same shape as dout
    :return:
        - dx: Gradient with respect to x
    """
    dx, x = None, cache
    dx = dout * np.maximum(0, x) / x  # np.maximum(0, x) calculate the derivative.

    return dx


def conv_forward_naive(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width WW.

    :param x: Input data of shape (N, C, H, W)
    :param w: Filter weights of shape (F, C, HH, WW)
    :param b: Biases, of shape (F,)
    :param conv_param:
        - 'stride': The number of pixels between adjacent receptive fields in the
            horizontal and vertical directions.
        - 'pad': The number of pixels that will be used to zero-pad the input.
    :return: A tuple of:
        - out: Output data, of shape (N, F, H', W') where H' and W' are given by
            H' = 1 + (H + 2 * pad - HH) / stride
            W' = 1 + (W + 2 * pad - WW) / stride
        - cache: (x, w, b, conv_param)
    """
    out = None

    N, C, H, W = x.shape
    F, _, HH, WW = w.shape
    stride = conv_param['stride']
    pad = conv_param['pad']
    # (0, ) means do nothing in 1st and 2nd dims
    # (pad, ) padding s lines 0 on both left and right of matrix. Only in the single image filed.
    x_pad = np.pad(x, ((0,), (0,), (pad,), (pad,)), 'constant')
    _, _, H_pad, W_pad = x_pad.shape

    H_out = int((H + 2 * pad - HH) / stride) + 1
    W_out = int((H + 2 * pad - WW) / stride) + 1

    out = np.zeros((N, F, H_out, W_out))

    cnt = 0
    for p in range(0, H_pad - HH + 1, stride):
        for q in range(0, W_pad - WW + 1, stride):
            x_region = x_pad[:, :, p:(p + HH), q:(q + WW)].reshape((N, -1))
            # Size of H_out equal to size of W_out
            out[:, :, int(cnt / H_out), int(cnt % H_out)] = x_region.dot(w.reshape((F, -1)).T) + b
            cnt += 1

    """
    # "clear" version:
    for i in range(N):
        for f in range(F):
            cnt = 0
            for p in range(0, H_pad - HH + 1, stride):
                for q in range(0, W_pad - WW + 1, stride):
                    out[i, f, int(cnt/H_out), int(cnt%H_out)] = np.sum(x_pad[i, :, p:(p+HH), q:(q+WW)] * w[f]) + b[f]
                    cnt += 1
    """

    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.

    :param dout: Upstream derivatives.
    :param cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive
    :return: A tuple of:
        - dx: Gradient with respect to x
        - dw: Gradient with respect to w
        - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    x, w, b, conv_param = cache

    N, C, H, W = x.shape
    F, _, HH, WW = w.shape
    stride, pad = conv_param['stride'], conv_param['pad']
    H_out = 1 + (H + 2 * pad - HH) / stride
    W_out = 1 + (W + 2 * pad - WW) / stride

    x_pad = np.pad(x, ((0,), (0,), (pad,), (pad,)), mode='constant', constant_values=0)
    dx = np.zeros_like(x)
    dx_pad = np.zeros_like(x_pad)
    dw = np.zeros_like(w)
    db = np.zeros_like(b)

    db = np.sum(dout, axis=(0, 2, 3))

    x_pad = np.pad(x, ((0,), (0,), (pad,), (pad,)), mode='constant', constant_values=0)
    for i in range(H_out):
        for j in range(W_out):
            x_pad_masked = x_pad[:, :, i * stride:i * stride + HH, j * stride:j * stride + WW]
            for k in range(F):  # compute dw
                dw[k, :, :, :] += np.sum(x_pad_masked * (dout[:, k, i, j])[:, None, None, None], axis=0)
            for n in range(N):  # compute dx_pad
                c = i * stride
                d = j * stride
                dx_pad[n, :, c:c + HH, d:d + WW] += np.sum((w[:, :, :, :] * (dout[n, :, i, j])[:, None, None, None]),
                                                           axis=0)
    dx = dx_pad[:, :, pad:-pad, pad:-pad]
    return dx, dw, db
