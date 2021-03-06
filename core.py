# AUTOGENERATED! DO NOT EDIT! File to edit: core.ipynb (unless otherwise specified).

__all__ = ['Signals', 'QamSignal', 'WdmSignal']

# Cell
# from fastcore.foundation import delegat/es
import numpy as np
from dataclasses import dataclass


@dataclass()
class SignalParam:
    qam_order: int = 16
    symbol_length: int = 65536
    baudrate: float = 35e9
    sps: int = 2
    sps_in_fiber: int = 4
    pol_number: int = 2


class WdmSignal:

    def __init__(self):

        self.samples_in_fiber = None
        self.symbols = []
        self.centerfreq = []
        self.__device = 'cpu'
        self.fs_in_fiber = None

    @property
    def device(self):
        return self.__device

    def __add__(self, other):
        if not isinstance(other, WdmSignal) and not isinstance(other, QamSignal):
            raise TypeError

        else:
            import copy
            new_signal = copy.deepcopy(self)
            new_signal.samples_in_fiber += other.samples_in_fiber

        return new_signal

    def __mul__(self, number):
        if not isinstance(number, float) and not isinstance(number, int):
            if self.__device == 'cpu':
                from numpy import ndarray
            else:
                from cupy import ndarray
        
            if isinstance(number,ndarray):

                if len(number) != len(self.samples_in_fiber[0]):
                    raise ValueError("The mutiplied ndarry must be the same size with the signal")
            else:
                raise TypeError("The mutiplied object must be int,float,or ndarry(numpy or cupy)")

        import copy
        new_signal = copy.deepcopy(self)
        new_signal.samples_in_fiber *= number
        return new_signal

    def __rmul__(self, other):
        self.__mul__(other)

    def __getitem__(self, item):
        return self.samples_in_fiber[item]

    def __setitem__(self, key, value):
        self.samples_in_fiber[key] = value

    def to(self,deivce:str):
        if deivce.lower()== self.__device:
            return

        if deivce.lower() == 'cuda':
            from cupy import array
            self.samples_in_fiber = array(self.samples_in_fiber)

        if deivce.lower() == 'cpu':
            from cupy import asnumpy
            self.samples_in_fiber = asnumpy(self.samples_in_fiber)


class Signals(object):

    def __init__(self, qam_order: int = 16, symbol_length: int = 65536, baudrate: float = 35e9
                 , sps: int = 2, sps_in_fiber: int = 4
                 , pol_number: int = 2):
        '''
        Param:
            qam_order: The order of QAM
            symbol_length: the length of symbol
            baudrate: [GHz] baudrate of the signal
            sps: The samples per symbol for DSP
            sps_in_fiber: The samples per symbol when the signal is transimitted in the fiber
            pol_number: Whether to use pol-demultiplex
            device: Where the signal is on cpu/gpu

        ------------------------------------------------------

        The param below is read only and should not be changed:
            symbol: The complex QAM symbol
            msg: The message between 0 ~ qam_order -1
            samples: The sample in DSP
            samples_in_fiber: The samples in fiber
            __device:Whether the signal is on CPU on GPU
        '''
        self.qam_order = qam_order
        self.symbol_length = symbol_length
        self.baudrate = baudrate
        self.sps = sps
        self.sps_in_fiber = sps_in_fiber
        self.pol_number = pol_number

        self.__device = 'cpu'
        self.symbol = None
        self.msg = None
        self.samples = None
        self.samples_in_fiber = None

    @property
    def device(self):
        '''
        Function:
            Get the signal position, on GPU or CPU
        '''
        return self.__device

    def to(self, device: str):
        '''
        Param:
            device [str]: 'cuda' or cpu
        ---------
        function:
            Move the signal to the specified device

        Return:
            None
        '''
        if device.lower() == self.__device:
            return

        if device == 'cuda':
            try:
                import cupy as cp
            except ImportError:
                raise Exception("CUDA not supported")
            if self.samples is not None:
                self.samples = cp.array(self.samples)
            if self.samples_in_fiber is not None:
                self.samples_in_fiber = cp.array(self.samples_in_fiber)
            self.__device = 'cuda'

        if device == 'cpu':
            import cupy as cp
            if self.samples is not None:
                self.samples = cp.asnumpy(self.samples)
            if self.samples_in_fiber is not None:
                self.samples_in_fiber = cp.asnumpy(self.samples_in_fiber)
            self.__device = 'cpu'

    def __mul__(self, number):
        '''
        Function:
            Multilply the signal by a number or
        '''
        if not isinstance(number, float) and not isinstance(number, int):

            if self.__device == 'cpu':
                from numpy import ndarray
            else:
                from cupy import ndarray


            if not isinstance(number, ndarray):
                raise TypeError
            elif len(number) != len(self.samples_in_fiber[0]):
                raise TypeError
        else:
            raise TypeError

        import copy
        new_signal = copy.deepcopy(self)
        new_signal.samples_in_fiber *= number
        return new_signal

    def __getitem__(self, item):
        return self.samples_in_fiber[item]

    def __setitem__(self, key, value):
        self.samples_in_fiber[key] = value

    def __rmul__(self, number):
        return self.__mul__(number)

    def __add__(self, other) -> WdmSignal:
        '''
        param:
            other: A signal object
        function:
            add two signal and create a wdm_signal
        '''
        if not isinstance(other, Signals):
            raise TypeError
        #
        import copy
        res = WdmSignal()
        res.samples_in_fiber = self.samples_in_fiber + other.samples_in_fiber
        return res

    #
    def inplace_normalise(self):
        pass

    def scatterplot(self, sps):
        from DensityPlot import density2d
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(nrows=1, ncols=self.pol_number)

        for cnt, ax in enumerate(axes):
            data = self.samples_in_fiber[cnt, ::sps]
            i = data.real
            q = data.imag

            xlim = (i.min() - 0.2, i.max() + 0.2)
            ylim = (q.min() - 0.2, q.max() + 0.2)

            density2d(x=i, y=q, ax=ax, xlim=xlim, ylim=ylim)
            ax.set_aspect('equal')
        plt.tight_layout()
        plt.show()

    @property
    def fs_in_fiber(self):
        return self.baudrate * self.sps_in_fiber


class QamSignal(Signals):

    def __init__(self, need_init=True, signal_param=SignalParam()):

        super().__init__(qam_order=signal_param.qam_order, symbol_length=signal_param.symbol_length,
                         baudrate=signal_param.baudrate, sps=signal_param.sps, sps_in_fiber=signal_param.sps_in_fiber,
                         pol_number=signal_param.pol_number)
        self.need_init = need_init
        self.const = None
        if self.need_init:
            self.__init()

    def __init(self):
        self.encoding, self.const = QamSignal._generate_mapping(self.qam_order)
        self.msg = np.random.randint(0, self.qam_order, (self.pol_number, self.symbol_length))

        self.symbol = np.zeros_like(self.msg, dtype=np.complex128)

        for row_cnt, row in enumerate(self.msg):
            for m_cnt, m in enumerate(row):
                self.symbol[row_cnt, m_cnt] = self.encoding[m]

        self.msg = np.atleast_2d(self.msg)
        self.symbol = np.atleast_2d(self.symbol)

    @staticmethod
    def _generate_mapping(M, dtype=np.complex128):
        Nbits = np.log2(M)
        symbols = QamSignal.cal_symbols_qam(M).astype(dtype)
        # check if this gives the correct mapping
        scale = np.sqrt(QamSignal.cal_scaling_factor_qam(M))
        symbols /= scale
        _graycode = QamSignal.gray_code_qam(M)
        coded_symbols = symbols[_graycode]
        #         bformat = "0%db" % Nbits
        encoding = dict([(_graycode[i], symbols[i])
                         for i in range(len(_graycode))])
        return encoding, symbols

    @staticmethod
    def cal_symbols_qam(M):
        """
        Generate the symbols on the constellation diagram for M-QAM
        """
        if np.log2(M) % 2 > 0.5:
            return QamSignal.cal_symbols_cross_qam(M)
        else:
            return QamSignal.cal_symbols_square_qam(M)

    @staticmethod
    def cal_scaling_factor_qam(M):
        """
        Calculate the scaling factor for normalising MQAM symbols to 1 average Power
        """
        bits = np.log2(M)
        if not bits % 2:
            scale = 2 / 3 * (M - 1)
        else:
            symbols = QamSignal.cal_symbols_qam(M)
            scale = (abs(symbols) ** 2).mean()
        return scale

    @staticmethod
    def cal_symbols_square_qam(M):
        """
        Generate the symbols on the constellation diagram for square M-QAM
        """
        qam = np.mgrid[-(2 * np.sqrt(M) / 2 - 1):2 * np.sqrt(
            M) / 2 - 1:1.j * np.sqrt(M), -(2 * np.sqrt(M) / 2 - 1):2 * np.sqrt(M) /
                                                                   2 - 1:1.j * np.sqrt(M)]
        return (qam[0] + 1.j * qam[1]).flatten()

    @staticmethod
    def cal_symbols_cross_qam(M):
        """
        Generate the symbols on the constellation diagram for non-square (cross) M-QAM
        """
        N = (np.log2(M) - 1) / 2
        s = 2 ** (N - 1)
        rect = np.mgrid[-(2 ** (N + 1) - 1):2 ** (N + 1) - 1:1.j * 2 ** (N + 1), -(
                2 ** N - 1):2 ** N - 1:1.j * 2 ** N]
        qam = rect[0] + 1.j * rect[1]
        idx1 = np.where((abs(qam.real) > 3 * s) & (abs(qam.imag) > s))
        idx2 = np.where((abs(qam.real) > 3 * s) & (abs(qam.imag) <= s))
        qam[idx1] = np.sign(qam[idx1].real) * (
                abs(qam[idx1].real) - 2 * s) + 1.j * (np.sign(qam[idx1].imag) *
                                                      (4 * s - abs(qam[idx1].imag)))
        qam[idx2] = np.sign(qam[idx2].real) * (
                4 * s - abs(qam[idx2].real)) + 1.j * (np.sign(qam[idx2].imag) *
                                                      (abs(qam[idx2].imag) + 2 * s))
        return qam.flatten()

    @staticmethod
    def gray_code_qam(M):
        """
        Generate gray code map for M-QAM constellations
        """
        Nbits = int(np.log2(M))
        if Nbits % 2 == 0:
            N = Nbits // 2
            idx = np.mgrid[0:2 ** N:1, 0:2 ** N:1]
        else:
            N = (Nbits - 1) // 2
            idx = np.mgrid[0:2 ** (N + 1):1, 0:2 ** N:1]
        gidx = QamSignal.bin2gray(idx)
        return ((gidx[0] << N) | gidx[1]).flatten()

    @staticmethod
    def bin2gray(value):
        """
        Convert a binary value to an gray coded value see _[1]. This also works for arrays.
        ..[1] https://en.wikipedia.org/wiki/Gray_code#Constructing_an_n-bit_Gray_code
        """
        return value ^ (value >> 1)

    def __str__(self):
        return f'QamSignal(sps = {self.sps},sps_in_fiber = {self.sps_in_fiber}, baudrate = {self.baudrate / 1e9} GHz,' \
               f'device = {type(self.samples), type(self.samples_in_fiber)})'

    def __repr__(self):
        return self.__str__()


# Cell


if __name__ == '__main__':
    signal = QamSignal(True, SignalParam())
