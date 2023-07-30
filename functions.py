import numpy as np
import scipy.signal as signal

from obspy.taup import TauPyModel
from obspy.geodetics import gps2dist_azimuth
from obspy.signal.trigger import classic_sta_lta

def quantize(A, dtype = np.int16):
    'quantize float data in range [-127,127]'
    m = np.max(np.abs(A), axis = (1, 2), keepdims = True)
    factors = np.iinfo(dtype).max / m 
    return (A * factors).astype(dtype = dtype)

def remove_small_amplitude(A, B, min_amp = 1e-8):
    to_keep = np.where(np.max(np.abs(A), axis=(1,2)) > min_amp)[0]
    return A[to_keep], B[to_keep]

def remove_large_amplitude(A, B, max_amp = 2.0 * 9.81):
    to_keep = np.where(np.max(np.abs(A), axis = (1,2)) < max_amp)[0]
    return A[to_keep], B[to_keep]

def detrend(X):
    'Remove mean and trend from data'
    N = X.shape[-1]
    # create linear trend matrix 
    A = np.zeros((2, N),dtype = X.dtype)
    A[1,:] = 1
    A[0,:] = np.linspace(0, 1, N)
    R = A @ np.transpose(A)
    Rinv = np.linalg.inv(R)
    factor = np.transpose(A) @ Rinv
    X -= (X @ factor) @ A
    return X

def taper(A, alpha):
    'taper signal'
    window = signal.tukey(A.shape[-1],alpha)
    A *= window
    return A

def shift_event(data, maxshift, rate, start, halfdim): 
    'Randomly rotate the array to shift the event location'
    
    if np.random.uniform(0, 1) < rate:
        start += int(np.random.uniform(-maxshift, maxshift))             
    return data[:, start-halfdim:start + halfdim]

def adjust_amplitude_for_multichannels(data):
    'Adjust the amplitude of multi-channel data'
    
    tmp = np.max(np.abs(data), axis = -1, keepdims = True)
    assert(tmp.shape[0] == data.shape[0])
    if np.count_nonzero(tmp) > 0:
        data *= data.shape[0] / np.count_nonzero(tmp)
    return data

def scale_amplitude(data, rate):
    'Scale amplitude or waveforms'
    
    tmp = np.random.uniform(0, 1)
    if tmp < rate:
        data *= np.random.uniform(1, 3)
    elif tmp < 2*rate:
        data /= np.random.uniform(1, 3)
    return data

def normalize(data):
    'Normalize waveforms over each event'
        
    max_data = np.max(data, axis=(1, 2), keepdims=True)
    assert(max_data.shape[0] == data.shape[0])
    max_data[max_data == 0] = 1
    data /= max_data              
    return data