import collections
import time

import numpy as np
import obspy
import scipy
import torch

from Phidget22.Phidget import *
from Phidget22.Devices.Accelerometer import *

import functions

class EarthquakeDataCollector:
    def __init__(self, sampling_rate = 125, sleep = 1):
        self.g_acc = 9.81
        self.fsout = 40.0
        self.wlen = 2
        self.dim = int(self.wlen * self.fsout)
        self.sampling_rate = sampling_rate
        self.data_length = int(self.wlen * self.sampling_rate)
        self.all_data = collections.deque(maxlen = self.data_length)
        self.ch = Accelerometer()
        self.ch.setOnAccelerationChangeHandler(self.onAccelerationChange)

        self.model_finetuned = torch.load('./model/EarthquakeCNN_finetuned.pth')
        self.model_pretrained = torch.load('./model/EarthquakeCNN.pth')
        self.model_finetuned.eval()
        self.model_pretrained.eval()

        self.model_TS = torch.jit.load('./model/EarthquakeCNN_TS.pt')

        self.softmax = torch.nn.Softmax(dim = 1)

        self.sleep = sleep
        self.trigger = False

    def onAccelerationChange(self, self_ch, acceleration, timestamp):
        acceleration_arr = np.array(acceleration) * self.g_acc # Convert to m/s2
        self.all_data.append(acceleration_arr)

    def beeper_trigger(self):
        if self.trigger == True:
            # print('triggered')
            print("\033[91mtriggered\033[0m")



        self.trigger = False

    def start_collecting(self):
        self.ch.openWaitForAttachment(1000)
        self.ch.setDataRate(self.sampling_rate)
        try:
            while True:
                time.sleep(self.sleep)

                self.feed_data = self.get_process_last_2_seconds_data()
                self.feed_data = self.feed_data.reshape(1, 80, 3, 1)

                self.feed_data = functions.normalize(self.feed_data)
                self.feed_data = functions.quantize(self.feed_data)
                self.feed_data = self.feed_data.astype(np.float32)
                self.feed_data = functions.normalize(self.feed_data)

                X = torch.from_numpy(self.feed_data).float()
                X = X.permute(0, 3, 1, 2)

                Y = self.model_TS(X)
                Y = self.softmax(Y)
                Y = Y.detach().numpy()

                Y = np.round(Y * 100).astype(int)
                print(Y, end = ' ')
                PGA = self.get_last_2_seconds_data_PGA()
                print(PGA)
                if PGA >= 2.0 and Y[0][2] <= 0.5:
                    self.trigger = True

                self.beeper_trigger()

        except KeyboardInterrupt:
            self.ch.close()
            print("\nData collection stopped.")

    def get_last_2_seconds_data(self):
        return np.array(self.all_data)
    
    def get_process_last_2_seconds_data(self, alpha = 0.05, freq = 0.5):
        data_array = self.get_last_2_seconds_data()
        stream = obspy.Stream()
        for i in range(3):
            trace = obspy.Trace(data = data_array[:, i])
            stream.append(trace)
            trace.stats.sampling_rate = self.sampling_rate

        stream.detrend(type = "linear")
        stream.taper(alpha)
        stream.filter(type = "highpass", freq = freq)

        feed_data = np.vstack([stream[i].data for i in range(len(stream))])
        feed_data = scipy.signal.resample(feed_data, self.dim, axis = -1)
        feed_data = feed_data.transpose()

        return feed_data
    
    def get_last_2_seconds_data_PGA(self):
        data_array = self.get_process_last_2_seconds_data()
        composite_acceleration = np.sqrt(data_array[:, 0]**2 + data_array[:, 1]**2 + data_array[:, 2]**2)
        PGA = np.max(composite_acceleration) * 100 # Convert to gal

        return PGA


collector = EarthquakeDataCollector(sleep = 0.0)
collector.start_collecting()