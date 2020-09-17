from typing import List
import numpy as np

from transimitter import *

from core import WdmSignal
from typing import  Union

class Mux:

    def __init__(self, modulators: List[QamModulator]
                 , grid_size: Union[List[float],np.ndarray]
                 , start_freq: float
                 , grid: List
                 ):
        '''
        Param:
            modulator:  Different modulators
            gird_size:  信道间隔，等于信号的个数
            start_freq: 信道的起始频率
            gird:       信道设置
        Param read onlyL
            frequency: read the laser frequency of each modulator
            end_freq: calc the
        '''
        self.modulators = modulators
        self.grid_size = grid_size
        self.nch = len(modulators)
        self.grid = grid

        self.frequency = np.array([ts.laser.center_frequency for ts in self.modulators])
        self.start_freq = start_freq
        self.end_freq = self.start_freq
        self.symbols = [md.signal.symbol for md in self.modulators]
        if 0 not in self.grid:
            self.center_freq = (np.min(self.frequency) + np.max(self.frequency)) / 2
            self.band_center_freq = self.frequency - self.center_freq
        else:
            for cnt, grid_size in enumerate(self.grid_size):
                try:
                    self.end_freq = self.end_freq + grid_size / 2 + self.grid_size[cnt + 1] / 2
                except IndexError:
                    pass

            self.center_freq = (self.start_freq + self.end_freq) / 2
            self.band_center_freq = self.frequency - self.center_freq

    def mux_signal(self,device)->WdmSignal:
        if device == 'cuda':
            from cupy import exp
            from cupy import arange

        elif device == 'cpu':
            from numpy import exp
            from numpy import arange
        else:
            raise NotImplementedError
        
        for md in self.modulators:
            md.signal.to(device)
        
        signal_res = None

        for md,freq in zip(self.modulators,self.band_center_freq):

            t = 1 / md.fs_in_fiber * arange(len(md.signal[0]))
            if signal_res is None:

                signal_res = md.signal * exp(-1j*2*np.pi*freq*t)
                continue
            signal_res = signal_res + md.signal * exp(-1j*2*np.pi*freq*t)
            
        signal_res.centerfreq = self.band_center_freq
        signal_res.symbols = self.symbols

        self.modulators = None

        return signal_res
    
    
def calc_sps_in_fiber(start_freq,end_freq):
    pass


def wdm_test():
    # grid_size = np.array([50, 50, 75, 50, 50, 75, 50, 100, 50, 75, 100, 75, 100])
    # grid = [0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1]
    # start_freq = 193.1e12
    grid_size = np.array([50e9,50e9,50e9,50e9])
    grid = [1,1,1,1]

    modu = []

    for i,g in enumerate(grid_size):
        modu.append(QamModulator(
                                Laser(193.1e12 + g*i,0,False),
                                 IQ(),
                                 PulseShaping(0.2,1024),
                                 DAC(),
                                 SignalParam(sps_in_fiber=14)
                                )
                    )

        modu[-1].modulate('cpu')

    wdm = Mux(modu,grid_size,193.1e12,grid=grid)
    wdm_signal = wdm.mux_signal('cpu')
    print(len(wdm_signal.symbols))
    import matplotlib.pyplot as plt
    plt.psd(wdm_signal.samples_in_fiber[0])
    plt.show()


if __name__ == '__main__':
    wdm_test()
