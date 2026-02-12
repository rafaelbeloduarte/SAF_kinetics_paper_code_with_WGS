import numpy as np

def sigmoid(x, L, x0, k, b):
    y = b + L / (1 + np.exp(-k*(x - x0)))
    return y

def Kc(T):
    L_K_C2, x0_K_C2, k_K_C2, b_K_C2 = [12.877, 495.969, -0.267, 1.942]
    L_K_C3, x0_K_C3, k_K_C3, b_K_C3 = [5.167, 496.098, -0.557, -0.366]
    K_C2 = sigmoid(T, L_K_C2, x0_K_C2, k_K_C2, b_K_C2)
    K_C3 = sigmoid(T, L_K_C3, x0_K_C3, k_K_C3, b_K_C3)
    return K_C2, K_C3

def calc_alpha(T, H2_CO):
    beta = [39.79173658, -0.07452321, -0.73213305]
    alpha = 1 / ( 1 + np.exp( - ( beta[0] + beta[1] * T + beta[2] * H2_CO ) ) )
    return alpha

def calc_y_n(n, T, H2_CO):
    # K_C2, K_C3 = Kc(T)
    K_C2 = 3.12
    K_C3 = 0.809
    alpha = calc_alpha(T, H2_CO)
    y_n = ( 1 - alpha ) * alpha ** ( n - 1  + K_C2 * ( n == 2 ) + K_C3 * ( n == 3 ) )
    return y_n

def y_n_distribution(T, H2_CO, N):
    y_n = {}

    for n in N:
        y_n[n] = calc_y_n(n, T, H2_CO)

    # the chategorical variables change the distribution sum to be always less than 1
    # so we will normalize it before returning
    y_n = {key: value/sum(y_n.values()) for key, value in y_n.items()}
    return y_n

