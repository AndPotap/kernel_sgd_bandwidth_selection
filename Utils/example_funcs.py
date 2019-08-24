import numpy as np


def doppler(variable: np.array, obs_n: int):
    doppler_output = np.sqrt(variable * (1 - variable)) * np.sin(2.1 * np.pi / (variable + 0.05))
    return np.reshape(a=doppler_output, newshape=obs_n)
