# AUTOGENERATED! DO NOT EDIT! File to edit: sigdsp.ipynb (unless otherwise specified).

__all__ = ['conv']

def conv(h,sample,device):

    if device =='cuda':
        from cusignal import convolve
        from cupy import array,vstack
    if device =='cpu':
        from numpy import convolve,vstack
    from numpy import array,atleast_2d

    res = []
    for row_cnt,row in enumerate(sample):
        res.append(convolve(h[0],row))

    if len(res) == 2:
        return vstack(res)
    else:
        return atleast_2d(array(res))


def sync():
    pass

