import numpy as np

beta = np.load('params_upsilon.npy')

def upsilon_n(n, T):
    exponent = beta[0] + beta[1] * n + beta[2] * T + beta[3] * ( n == 2 )
    return 1 / ( 1 + ( 1 - ( ( n == 1 ) | ( n == 3 ) ) ) * np.exp( - exponent ) )