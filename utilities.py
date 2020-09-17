import numpy as np
import matplotlib.pyplot as plt
from core import QamSignal, SignalParam,WdmSignal
from numpy import ndarray


def load_qam_signal(name):
    pass


def save_qam_signal(name, signal: QamSignal, is_mat=False):
    signal.to('cpu')
    attr = dir(signal)

    res = {}
    for shuxing in attr:
        if shuxing.startswith('__'):
            continue

        else:
            value = getattr(signal, shuxing)
            if isinstance(value, ndarray) or isinstance(value, float) or isinstance(value, int):
                res[shuxing] = value

    if is_mat:
        from scipy.io import savemat
        savemat(file_name=name + '.mat', mdict=res)

    else:
        import joblib
        joblib.dump(res, name + '.data')


def load_qam_signal(name, is_mat=False):
    import joblib
    if is_mat:
        from scipy.io import loadmat
        data = loadmat(name)
        for key in data:
            if isinstance(data[key], ndarray) and data[key].shape == (1, 1):
                data[key] = data[key][0, 0]

    else:
        data = joblib.load(name)

    signal = SignalParam(data['qam_order'], data['symbol_length'], data['baudrate'], data['sps'],
                         data['sps_in_fiber'], data['pol_number'])
    signal = QamSignal(need_init=False, signal_param=signal)
    signal.samples_in_fiber = data['samples_in_fiber']
    signal.samples = data['samples']
    signal.symbol = data['symbol']
    signal.msg = data['msg']
    return signal


def save_wdm_signal(name,
                    wdm_signal:WdmSignal
                    ,is_mat = True):

    import joblib
    from scipy.io import savemat
    wdm_signal.to('cpu')
    data = {}
    data['samples_in_fiber'] = wdm_signal.samples_in_fiber
    data['symbols'] = wdm_signal.symbols
    data['centerfreq'] = wdm_signal.centerfreq

    if is_mat:
        savemat(name+'.mat',data)

    else:
        import joblib
        joblib.dump(data,name +'.data')


def load_wdm_signal(name,is_mat = True):
    pass


