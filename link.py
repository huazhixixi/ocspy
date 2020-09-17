# AUTOGENERATED! DO NOT EDIT! File to edit: channel.ipynb (unless otherwise specified).

__all__ = ['FiberParam', 'SsfmFiber']

# Cell
from dataclasses import dataclass
from core import Signals



@dataclass()
class FiberParam:
    '''

    '''
    alpha: float = 0.2
    D: float = 16.7
    gamma: float = 1.3
    length: float = 80
    wavelength = 1550

    @property
    def beta2(self):
        pass


class SsfmFiber:

    def __init__(self, fiberparam: FiberParam, step_length: float):
        self.param = fiberparam
        self.step_length = step_length

    def __getattr__(self, item):
        if hasattr(self.param, item):
            return getattr(self.param, item)
        else:
            raise Exception("Attr not exist")

    def __call__(self, signal, device):
        signal.to(device)
        if device == 'cpu':
            self.prop_arrayfire_cpu(signal)
        if device == 'cuda':
            self.prop_cupy(signal)

    def prop_cupy(self, signal):

        def linear_prop(signal):
            return signal

        def nonlinear_prop(signal):
            return signal


class LinearFiber:
    pass


class SemiNonlinearFiber:
    pass


class Edfa:

    def __init__(self, gain_db, nf_db, mode):
        self.gain_db = gain_db
        self.nf_db = nf_db
        self.mode = mode
        self.device = 'cpu'

    def __call__(self, signal:Signals):
        # First amplify
        import numpy as np
        signal = np.sqrt(10 ** (self.gain_db/10)) * signal

        # ASE noise add to the signal






class Wss:

    def __init__(self, bw, otf):
        self.bw = bw
        self.otf = otf
        self.device = 'cpu'
        self.H = None

    def __call__(self, signal):
        pass


class Connector:
    pass