import numpy as np

param_alpha = np.load('param_alpha.npy')

def y_n(n, T):
    alpha = param_alpha[0] + param_alpha[1] * T
    return ( 1 - alpha ) * alpha ** ( n - 1 )

