import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.constants as sc
import os


class Light:
    type_options = ['spectral', 'temporal']
    def __init__(self, params, type):
        if type not in self.type_options:
            raise AttributeError(f'`type` parameter has to be either "{self.type_options[0]}" or "{self.type_options[1]}"')
        if type == self.type_options[0]:
            self.initialize_spectrum(params)
        else:
            self.initialize_temporal_profile(params)

    def initialize_spectrum(self, params):
        print('I was in initialize spectrum')

    def initialize_temporal_profile(self, params):
        print('I was in initialize temporal profile')


class SpectralBeamInput_Gaussian:
    beam_name = 'gaussian'
    def __init__(self, **kwargs):
        pass

if __name__ == '__main__':
    lambda0 = 900e-9
    delta_lambda = 200e-9
    omega0 = 2*np.pi*sc.c / lambda0
    delta_omega = 2*np.pi*sc.c / delta_lambda
    N = 1000
    omega = np.linspace(0, 2*omega0, N)
    amp = np.exp(-(omega - omega0)**2 / (2*delta_omega**2)) / (np.sqrt(2*np.pi) * delta_omega)
    polarization = 'x'
    phase = np.zeros(len(omega))
    pars = {'amplitude' : amp,
            'phase'     : phase,
            'polarization' : polarization,
            'omega'     : omega}
    beam = Light(pars, type='spectral')