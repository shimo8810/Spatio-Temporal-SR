import numpy as np
import chainer.functions as F

if __name__ == '__main__':
    num_data = 1
    ch_input = 3
    ch_output = 6
    size = 1
    kernel = 1
    stride = 2
    pad = 1
    x = np.random.randn(num_data, ch_input, size, size)
    W = np.random.randn(ch_output, ch_input, kernel, kernel)
    print("x shape:", x.shape)
    print("W shape:", W.shape)

    z = F.convolution_2d(x, W, stride=stride, pad=pad)
    print("z shape:", z.shape)

    y = F.deconvolution_2d(z, W, stride=stride, pad=pad)
    print("y shape:", y.shape)
