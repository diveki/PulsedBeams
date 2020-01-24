import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.constants as sc
import os


class Light:
    type_options = ['spectral', 'temporal']
    beam_type    = ['gaussian', 'lorentzian']
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
        self.amplitude = params.get('amplitude', self.error(KeyError, '`amplitude` key and value is missing!'))
        if type(self.amplitude) == str:
            print('Loading file')
        elif type(self.amplitude) == np.ndarray:
            print('Loading numpy array')
        elif type(self.amplitude) == float or type(self.amplitude) == int:
            self.t = params.get('t', self.error(KeyError, '`t` key and value is missing!'))
            self.create_beam_from_params(params)
            print(f'the amplitude is {self.amplitude}')
        else:
            raise TypeError('amplitude has to be a filename or numpy array or float or integer')

    def error(self, errortype, text):
        raise errortype(text)

    def create_beam_from_params(self, params):
        name = params.get('name', self.error(NameError, f'The key `name` has to be "{self.beam_type[0]}" or "{self.beam_type[1]}"!'))
        if name not in self.beam_type:
            raise NameError(f'The key `name` has to be "{self.beam_type[0]}" or "{self.beam_type[1]}"!')
        if name == self.beam_type[0]:
            self._create_beam_from_gaussian(params)
        elif name == self.beam_type[1]:
            self._create_beam_from_lorentzian(params)
        
    def _create_beam_from_gaussian(self, params):
        sigma  = params.get('sigma', None)
        FWHM   = params.get('FWHM', None)
        t0     = params.get('t0', 0)
        lambda0= params.get('lambda0', self.error(KeyError, f'The key `lambda0` has to be in the argument!'))
        pol    = params.get('polarization', 'x')
        beam   = Gaussian_temporal(self.amplitude, FWHM=FWHM, spread=sigma, t0=t0, lambda0=lambda0, pol=pol)

    def _create_beam_from_lorentzian(self, params):
        pass

    def get_spectral_intensity(self):
        pass

    def get_pulse_duration_FWHM(self):
        pass

    def get_central_wavelength(self):
        pass

    def get_envelope(self):
        pass

    def get_electric_field(self):
        pass

    def get_spectrum(self):
        pass

    def get_spectral_phase(self):
        pass


class Temporal_Envelope_Profile:
    def __init__(self, amp=1, FWHM=None, spread = None, t0=0, lambda0=None, pol='x'):
        self.amplitude = amp  
        if FWHM == None and spread == None:
            raise ValueError(f'FWHM or spread must be a number !!')
        self.FWHM      = FWHM 
        self.polarization = pol
        self.param     = spread
        self.t0        = t0
        self.lambda0   = lambda0
        self.set_params()
    
    def set_params(self):
        if self.FWHM == None:
            self.FWHM = self.param2FWHM(self.param)
        elif self.param == None:
            self.param = self.FWHM2param(self.FWHM)

    def FWHM2param(self, x):
        pass

    def param2FWHM(self, x):
        pass

    def create_envelope(self, t):
        pass


class Gaussian_temporal(Temporal_Envelope_Profile):
    def __init__(self, amp=1, FWHM=None, sigma=None, lambda0=None, t0=0, pol='x'):
        Temporal_Envelope_Profile.__init__(self, amp=amp, FWHM=FWHM, spread=sigma, t0=t0, pol=pol)
    
    def FWHM2param(self, x):
        return x / (2*np.sqrt(2*np.log(2)))

    def param2FWHM(self, x):
        return 2*x*np.sqrt(2*np.log(2))

    def create_envelope(self, t):
        env = self.amplitude * np.exp(-(t - self.t0)**2 / (2*self.param)) 
        return env

class Lorentzian_temporal(Temporal_Envelope_Profile):
    def __init__(self, amp=1, FWHM=None, sigma=None, lambda0=None, t0=0, pol='x'):
        Temporal_Envelope_Profile.__init__(self, amp=amp, FWHM=FWHM, spread=sigma, t0=t0, pol=pol)
    
    def FWHM2param(self, x):
        return x / 2

    def param2FWHM(self, x):
        return 2*x

    def create_envelope(self, t):
        env = self.amplitude / (1 + (t-self.t0)**2/self.param**2)
        return env


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