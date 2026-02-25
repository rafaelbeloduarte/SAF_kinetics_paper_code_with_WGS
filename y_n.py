import numpy as np

alpha = np.load('param_alpha.npy')

def y_n(n):
    return ( 1 - alpha ) * alpha ** ( n - 1 )

