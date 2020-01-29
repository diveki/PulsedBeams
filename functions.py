import numpy as np
import scipy.constants as sc


def lambda2omega(x):
    return 2*np.pi*sc.c / x

def get_fft_delta_omega(t):
    T = t.max() - t.min()
    return 2*np.pi / T
    
def gaussian_shape(x, mean=0, sigma=1):
    return np.exp(-(x - mean)**2 / (2*sigma**2))

def lorentzian_shape(x, mean=0, gamma=1):
    return 1 / (1 + (x-mean)**2/gamma**2)