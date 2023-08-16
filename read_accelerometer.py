import time

import numpy as np
import obspy
import scipy
import torch

from Phidget22.Phidget import *
from Phidget22.Devices.Accelerometer import *

import functions

sampling_rate = 125

def scale_amplitude(data, rate):
    'Scale amplitude or waveforms'
    
    tmp = np.random.uniform(0, 1)
    if tmp < rate:
        data *= np.random.uniform(1, 3)
    elif tmp < 2 * rate:
        data /= np.random.uniform(1, 3)
    return data


all_data = np.empty((0, 3))

def onAccelerationChange(self, acceleration, timestamp):
    global all_data

    acceleration_arr = np.array([acceleration])
    acceleration_arr = acceleration_arr * 9.81 * 100
    
    all_data = np.vstack((all_data, acceleration_arr))

ch = Accelerometer()
ch.setOnAccelerationChangeHandler(onAccelerationChange)

ch.openWaitForAttachment(1000)
ch.setDataRate(sampling_rate)

min_rate = ch.getMinDataRate()
max_rate = ch.getMaxDataRate()
print(f"Min Data Rate: {min_rate}, Max Data Rate: {max_rate}")

sampling_rate_actual = ch.getDataRate()
print("DataRate: " + str(sampling_rate_actual))

sampling_rate = sampling_rate_actual

start_time = time.perf_counter()
while time.perf_counter() - start_time < 10:
    last_1000_data = all_data[-250:]
    input_Stream = obspy.Stream()
    for i in range(3):
        trace = obspy.Trace(data = last_1000_data[:, i])
        trace.stats.sampling_rate = sampling_rate

        input_Stream.append(trace)
        input_Stream.detrend(type = "linear")
        input_Stream.taper(0.05)
        input_Stream.filter(type = "highpass", freq = 0.5)

        data = np.vstack([input_Stream[i].data for i in range(len(input_Stream))])

        fsin = sampling_rate
        fsout = 40.0
        wlen = 2.0
        alpha = 0.05
        freq = 0.5
        maxshift = 80
        shift_event_r = 0.995
        scale_amplitude_r = 0.3
        dim = int(wlen * fsout)

        data = scipy.signal.resample(data, dim, axis = -1)
        data = data.transpose()

ch.close()

last_1000_data = all_data[-250:]
print("All data collected:", all_data.shape)
print(last_1000_data.shape)
data = data.reshape(1, 80, 3, 1)

data_feed = functions.normalize(data)
data_feed = functions.quantize(data_feed)
data_feed = data_feed.astype(np.float32)
data_feed = functions.normalize(data_feed)
print(data_feed.shape)

model = torch.load('./model/EarthquakeCNN_finetuned.pth')
softmax = torch.nn.Softmax(dim = 1)

X = torch.from_numpy(data_feed).float()
X = X.permute(0, 3, 1, 2)
Y = model(X)
Y = softmax(Y)
Y = Y.detach().numpy()
print(Y)