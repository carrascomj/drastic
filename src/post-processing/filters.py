"""
Implementantion of filters to enhance predictions of the models, reducing
sparsity
"""

import numpy as np
from scipy.signal import butter, lfilter, freqz


def lowpass(cutoff, fs, order):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def lowpass_filter(data, cutoff, fs, order):
    b, a = lowpass(cutoff, fs, order)
    y = lfilter(b, a, data)
    return y
    
def cut_filter(output, cutoff):
    return [i >= 0.5 for i in output]