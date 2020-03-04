import numpy as np
import scipy.constants as sc
from scipy.interpolate import UnivariateSpline


def lambda2omega(x):
    return 2*np.pi*sc.c / x

def get_fft_delta_omega(t):
    T = t.max() - t.min()
    return 2*np.pi / T
    
def gaussian_shape(x, mean=0, sigma=1):
    return np.exp(-(x - mean)**2 / (2*sigma**2))

def lorentzian_shape(x, mean=0, gamma=1):
    return 1 / (1 + (x-mean)**2/gamma**2)


def find_FWHM(x, y):
    spline = UnivariateSpline(x, y-np.max(y)/2, s=0)
    r1, r2 = spline.roots() # find the roots
    return r1, r2