import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.constants as sc
from functions import *
import os

def error(errortype, text):
        raise errortype(text)


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
        import pdb
        # pdb.set_trace()
        self.amplitude = params['amplitude']#, error(KeyError, '`amplitude` key and value is missing!'))
        if type(self.amplitude) == str:
            print('Loading file')
        elif type(self.amplitude) == np.ndarray:
            print('Loading numpy array')
        elif type(self.amplitude) == float or type(self.amplitude) == int:
            self.t = params['t']#, self.error(KeyError, '`t` key and value is missing!'))
            self.create_beam_from_params(params)
            self.from_temporal2spectral()
            print(f'the amplitude is {self.amplitude}')
        else:
            raise TypeError('amplitude has to be a filename or numpy array or float or integer')

    def create_beam_from_params(self, params):
        name = params['name']#, self.error(NameError, f'The key `name` has to be "{self.beam_type[0]}" or "{self.beam_type[1]}"!'))
        if name not in self.beam_type:
            raise NameError(f'The key `name` has to be "{self.beam_type[0]}" or "{self.beam_type[1]}"!')
        if name == self.beam_type[0]:
            self._create_beam_from_gaussian(params)
        elif name == self.beam_type[1]:
            self._create_beam_from_lorentzian(params)
        
    def _create_beam_from_gaussian(self, params):
        sigma  = params.get('sigma', None)
        FWHM   = params.get('FWHM', None)
        self.t0     = params.get('t0', 0)
        self.lambda0= params['lambda0']#, self.error(KeyError, f'The key `lambda0` has to be in the argument!'))
        self.omega0 = self._lambda2omega(self.lambda0)
        self.polarization    = params.get('polarization', 'x')
        self.beam = Gaussian_temporal(self.amplitude, FWHM=FWHM, spread=sigma, t0=self.t0, lambda0=self.lambda0, pol=self.polarization)
        self.FWHM = self.beam.FWHM
        self.spread = self.beam.param
        self.temporal_envelope = self._calculate_temporal_envelope(self.t)
        self.temporal_electric_field = self._calculate_temporal_electric_field(self.amplitude, self.temporal_envelope, self.omega0, self.t)

    def _create_beam_from_lorentzian(self, params):
        pass
    
    def from_temporal2spectral(self):
        self.delta_omega = self._calculate_delta_omega(self.t)
        self.omega = self._calculate_omega(self.t, self.omega0, self.delta_omega)
        self.lambdas = self._calculate_lambdas(self.omega)
        self.spectrum = self._calculate_spectrum(self.temporal_electric_field)

    def _calculate_spectrum(self, ef):
        return np.fft.fft(ef, norm='ortho')


    def _calculate_lambdas(self, value):
        return lambda2omega(value)

    def _calculate_omega(self, t, omega0, dw):
        return np.fft.fftfreq(len(t))*len(t)*dw

    def _calculate_delta_omega(self, t):
        T = t.max() - t.min()
        return 2*np.pi / T

    def _lambda2omega(self, value):
        return 2*np.pi*sc.c / value

    def set_FWHM(self, value):
        pass
        # self.FWHM = value
        # self.beam = Gaussian_temporal(self.amplitude, FWHM=self.FWHM, spread=None, t0=self.t0, lambda0=self.lambda0, pol=self.polarization)
        # self.spread = self.beam.spread
        
    def get_spectral_intensity(self):
        pass

    def get_pulse_duration_FWHM(self):
        return self.FWHM

    def get_central_wavelength(self):
        return self.lambda0

    def get_envelope(self):
        pass

    def get_electric_field(self):
        pass

    def get_spectrum(self):
        pass

    def get_spectral_phase(self):
        pass

    def get_spectral_bandwidth(self, freq, spectrum):
        pass


class Temporal_Envelope_Profile:
    def __init__(self, amp=1, FWHM=None, spread = None, t=None, t0=0, phase=None, lambda0=None, pol='x'):
        self.amplitude = amp  
        if FWHM == None and spread == None:
            raise ValueError(f'FWHM or spread must be a number !!')
        self.FWHM      = FWHM 
        self.polarization = pol
        self.param     = spread
        self.t         = t
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
    
    def update_beam(self):
        pass

    def create_envelope(self, t):
        pass

    # def _calculate_temporal_envelope(self, t):
    #     return self.beam.create_envelope(t)

    def get_temporal_electric_field(self, amp, env, omega0, phase, t):
        # import pdb
        # pdb.set_trace()
        return amp * env * np.exp(1j*(omega0*t + phase))
    
    def set_amplitude(self, value, update=False):
        self.amplitude = value
        if update:
            self.update_beam()

    def set_phase(self, value, update=False):
        pass

    def set_FWHM(self, value, update=False):
        self.FWHM = value
        if update:
            self.update_beam()

    def set_spread(self, value, update=False):
        self.param = value
        if update:
            self.update_beam()

    def set_t0(self, value, update=False):
        self.t0 = value
        if update:
            self.update_beam()

    def set_lambda0(self, value, update=False):
        self.lambda0 = value
        if update:
            self.update_beam()

    def set_polarization(self, value, update=False):
        self.polarization = value
        if update:
            self.update_beam()


class Gaussian_temporal(Temporal_Envelope_Profile):
    def __init__(self, amp=1, FWHM=None, spread=None, lambda0=None, t=None, t0=0, phase=None, pol='x'):
        Temporal_Envelope_Profile.__init__(self, amp=amp, FWHM=FWHM, spread=spread, t=t, t0=t0, phase=phase, lambda0=lambda0, pol=pol)
        # import pdb
        # pdb.set_trace()
        self.set_phase(phase)
        self.envelope = self.create_envelope(t)
        self.omega0 = lambda2omega(lambda0)
        self.temporal_electric_field = self.get_temporal_electric_field(self.amplitude, self.envelope, self.omega0, self.phase, self.t)
    
    def FWHM2param(self, x):
        return x / (2*np.sqrt(2*np.log(2)))

    def param2FWHM(self, x):
        return 2*x*np.sqrt(2*np.log(2))

    def create_envelope(self, t):
        env = self.amplitude * gaussian_shape(t, mean=self.t0, sigma=self.param) # np.exp(-(t - self.t0)**2 / (2*self.param**2)) 
        return env
    
    def update_beam(self):
        Temporal_Envelope_Profile.__init__(self, amp=self.amplitude, FWHM=self.FWHM, spread=self.param, t=self.t, t0=self.t0, lambda0=self.lambda0, pol=self.polarization)
        self.envelope = self.create_envelope(self.t)
        self.omega0 = lambda2omega(self.lambda0)
        self.temporal_electric_field = self.get_temporal_electric_field(self.amplitude, self.envelope, self.omega0, self.phase, self.t)
            
    def set_phase(self, value, update=False):
        if value == None:
            self.phase = 0
        else:
            self.phase = value
        if update:
            self.update_beam()


class Lorentzian_temporal(Temporal_Envelope_Profile):
    def __init__(self, amp=1, FWHM=None, spread=None, lambda0=None, t0=0, pol='x'):
        Temporal_Envelope_Profile.__init__(self, amp=amp, FWHM=FWHM, spread=spread, t0=t0, pol=pol)
    
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
    t = np.linspace(-70, 70, 500)*1e-15
    sigma=20e-15
    delta_lambda = 200e-9
    omega0 = 2*np.pi*sc.c / lambda0
    delta_omega = 2*np.pi*sc.c / delta_lambda
    N = 1000
    omega = np.linspace(0, 2*omega0, N)
    amp = np.exp(-(omega - omega0)**2 / (2*delta_omega**2)) / (np.sqrt(2*np.pi) * delta_omega)
    polarization = 'x'
    phase = np.zeros(len(omega))
    pars = {
        't': t,
        'amplitude':1,
        'lambda0':lambda0,
        'FWHM':30e-15,
        'polarization':polarization,
        'name':'gaussian'
    }
    beam = Gaussian_temporal(amp=1, FWHM=30e-15, spread=None, lambda0=lambda0, t=t, t0=10e-15, phase=None, pol=polarization)