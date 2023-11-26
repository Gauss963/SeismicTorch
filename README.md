# SeismicTorch
Seismic Event Classification using Convolutional Neutral Network applied to time series data.

This machine learning application to seismology uses a CNN to classify seismic events into earthquakes, active seismic sources and noise based on raw waveform records with a duration of 480 seconds using MEMS.

Package Requirements:
The pre-processing of the seismic data requires the Obspy package which is easy to install with pip.
1) Numpy
2) Scipy
3) PyTorch
4) Matplotlib
5) Sklearn
6) Scikit-learn

# Tutorial
1) Download the code and data. In the folder "data_QSIS_Event" and "data_QSIS_Noise", put your .sac files in corresponding folders.
2) Run: "QSIS_dataprep_event.ipynb".
3) Run: "QSIS_dataprep_noise.ipynb".
4) Run: "QSIS_dataprep_merge.ipynb".
5) Run: "model_train_Torch.ipynb" to pre-train model with STanford EArthquake Dataset.
6) Run: "model_finetune_Torch.ipynb" to finetune model with QSIS Dataset.
7) Check the results in the "Result_Training" and "Result_Finetuing" folder.
8) Run: "convert_to_ONNX.py" to convert the PyTorch model into ONNX model.
 
The ANN architecture includes the following laters: Conv2D, MaxPool, DropOut, Conv2D, MaxPool, DropOut, Conv2D, MaxPool, DropOut, FC16(relu), FC3(softmax)
The code generates plots of loss function, validation curve and some examples of seismic waves and noise.
The overall accuracy for the pre-train model is 93%.

## Code of data preparation functions

```python
import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.signal as signal

from io import BytesIO
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

def stream_to_spectrogram_ndarray(input_Stream):
    xyz = ['x', 'y', 'z']
    array_list = []
    for i in range(3):
        trace = input_Stream[i]
        trace_acceleration = trace.data

        # 設置 Spectrogram 參數
        fs = trace.stats.sampling_rate  # 取樣率
        nperseg = 256                   # 每個段的數據點數
        noverlap = nperseg // 2         # 重疊的數據點數

        # Draw Spectrogram
        frequencies, times, Sxx = scipy.signal.spectrogram(trace_acceleration, fs=fs, nperseg=nperseg, noverlap=noverlap)

        plt.pcolormesh(times, frequencies, 10 * np.log10(Sxx), shading = 'auto', cmap = 'gray')
        plt.axis('off')

        # Use BytesIO to save Matplotlib image to ram
        img_stream = BytesIO()
        plt.savefig(img_stream, format = 'png', bbox_inches = 'tight', pad_inches = 0)
        img_stream.seek(0)

        # 使用 OpenCV 讀取並縮放圖片
        img = cv2.imdecode(np.frombuffer(img_stream.read(), dtype=np.uint8), 1)
        img_resized = cv2.resize(img, (150, 100))

        # 將縮放後的圖片轉換為 NumPy 陣列
        img_resized_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        img_resized_gray_array = np.asarray(img_resized_gray)

        # Plot all three channel (For test)
        # cv2.imwrite('./Spectrogram_test/' + xyz[i] + '.png', img_resized)

        array_list.append(img_resized_gray_array)

    # 合併成一張彩色圖片 (100, 150, 3)
    # Also, ues RGB instead of BGR. Easier to read
    color_image = np.stack([array_list[2], array_list[0], array_list[1]], axis = -1)

    return color_image
```

## Time Series Model Definition

```python
class EarthquakeCNN(nn.Module):
    def __init__(self):
        super(EarthquakeCNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 16, kernel_size = (3, 1), stride = (1, 1))
        self.bn1 = nn.BatchNorm2d(16)
        self.GELU1 = nn.GELU()
        self.maxpool1 = nn.MaxPool2d(kernel_size = (3, 1), stride = (3, 1))
        self.dropout1 = nn.Dropout(p = 0.1)

        self.conv2 = nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = (3, 1), stride = (1, 1))
        self.bn2 = nn.BatchNorm2d(32)
        self.GELU2 = nn.GELU()
        self.maxpool2 = nn.MaxPool2d(kernel_size = (3, 1), stride = (3, 1))
        self.dropout2 = nn.Dropout(p = 0.1)

        self.conv3 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = (3, 1), stride = (1, 1))
        self.bn3 = nn.BatchNorm2d(64)
        self.GELU3 = nn.GELU()
        self.maxpool3 = nn.MaxPool2d(kernel_size = (3, 1), stride = (3, 1))
        self.dropout3 = nn.Dropout(p = 0.1)

        self.fc1 = nn.Linear(384, 64)
        self.GELU4 = nn.GELU()
        self.fc2 = nn.Linear(64, 16)
        self.GELU5 = nn.GELU()
        self.fc3 = nn.Linear(16, 3)
        

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.GELU1(x)
        x = self.maxpool1(x)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.GELU2(x)
        x = self.maxpool2(x)
        x = self.dropout2(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.GELU3(x)
        x = self.maxpool3(x)
        x = self.dropout3(x)

        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = self.GELU4(x)
        x = self.fc2(x)
        x = self.GELU5(x)
        x = self.fc3(x)

        return x
```


## Spectrogram Model Definition

```python
class SpectrogramCNN(nn.Module):
    def __init__(self):
        super(SpectrogramCNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 48, kernel_size = (3, 3), stride = (1, 1))
        self.bn1 = nn.BatchNorm2d(48)
        self.GELU1 = nn.GELU()
        self.maxpool1 = nn.MaxPool2d(kernel_size = (2, 2), stride = (2, 2))
        self.dropout1 = nn.Dropout(p = 0.1)

        self.conv2 = nn.Conv2d(in_channels = 48, out_channels = 96, kernel_size = (3, 3), stride = (1, 1))
        self.bn2 = nn.BatchNorm2d(96)
        self.GELU2 = nn.GELU()
        self.maxpool2 = nn.MaxPool2d(kernel_size = (2, 2), stride = (2, 2))
        self.dropout2 = nn.Dropout(p = 0.1)

        self.conv3 = nn.Conv2d(in_channels = 96, out_channels = 192, kernel_size = (3, 3), stride = (1, 1))
        self.bn3 = nn.BatchNorm2d(192)
        self.GELU3 = nn.GELU()
        self.maxpool3 = nn.MaxPool2d(kernel_size = (2, 2), stride = (2, 2))
        self.dropout3 = nn.Dropout(p = 0.1)

        self.fc1 = nn.Linear(32640, 1280)
        self.GELU4 = nn.GELU()
        self.fc2 = nn.Linear(1280, 64)
        self.GELU5 = nn.GELU()
        self.fc3 = nn.Linear(64, 2)
        

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.GELU1(x)
        x = self.maxpool1(x)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.GELU2(x)
        x = self.maxpool2(x)
        x = self.dropout2(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.GELU3(x)
        x = self.maxpool3(x)
        x = self.dropout3(x)

        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = self.GELU4(x)
        x = self.fc2(x)
        x = self.GELU5(x)
        x = self.fc3(x)

        return x
```