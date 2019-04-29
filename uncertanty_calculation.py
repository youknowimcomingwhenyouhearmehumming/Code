import numpy as np

def f(error_rate):
    return np.sqrt((error_rate*(1-error_rate))/(number observations))