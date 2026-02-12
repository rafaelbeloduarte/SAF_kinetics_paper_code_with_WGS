import numpy as np

def upsilon_n(n):
    exponential = np.exp ( - ( -3.303 + 0.4139 * n + 3.391 * ( n == 2 ) + 21.01 * ( n == 3 ) ) )
    upsilon_n = 1 / ( 1 + ( 1 - ( n == 1 ) ) * exponential )
    return upsilon_n