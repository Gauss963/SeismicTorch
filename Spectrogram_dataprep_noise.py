import cv2
import matplotlib.pyplot as plt
import numpy as np
import obspy
import scipy

from io import BytesIO


import os
import glob
import random
import cv2
import obspy
import scipy
import torch

from io import BytesIO
from obspy import Stream
from obspy import read
from obspy.taup import TauPyModel
from obspy.geodetics import gps2dist_azimuth
from obspy.signal.trigger import classic_sta_lta

import numpy as np 
import scipy.signal as signal
import matplotlib.pyplot as plt 
import tqdm


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

if __name__ == "__main__":
    # 讀取 SAC 檔案
    datadir = './dada_QSIS_test'
    file1 = datadir + '/RCEC.09f.5AF30.TW.A1.HLX.2022.09.18.06.43.22.sac'
    file2 = datadir + '/RCEC.09f.5AF30.TW.A1.HLY.2022.09.18.06.43.22.sac'
    file3 = datadir + '/RCEC.09f.5AF30.TW.A1.HLZ.2022.09.18.06.43.22.sac'
    test_event = obspy.read(file1) + obspy.read(file2) + obspy.read(file3)
    print(type(test_event))
    print(type(test_event[0]))
    colored_image = stream_to_spectrogram_ndarray(test_event)

    print(colored_image.shape)

    # 保存為 PNG 檔案
    # cv2.imwrite('./Spectrogram_test/test_3channel_merged_image.png', colored_image)

    stream_list = []
    stream_list = []
    Spectrogram_train_data_X = np.array([])
    Spectrogram_train_data_list = []

    file_name_part = 'HLX'

    all = 0
    useable = 0

    datadir = './data_QSIS_Noise'
    file_name_part = 'HLX'

    sac_files_X = glob.glob(f'{datadir}/*{file_name_part}*.sac')
    sac_files_Y = [s.replace('X', 'Y') for s in sac_files_X]
    sac_files_Z = [s.replace('X', 'Z') for s in sac_files_X]

    # print(sac_files_X)
    # print(sac_files_Y)
    # print(sac_files_Z)

    for i in tqdm.trange(len(sac_files_X)):
        stream = Stream()
        st = read(sac_files_X[i])
        stream += st

        st = read(sac_files_Y[i])
        stream += st

        st = read(sac_files_Z[i])
        stream += st

        stream_list.append(stream)

    # for i in tqdm.trange(len(stream_list)):
    for i in tqdm.trange(4):

        Spectrogram_train_data_append = stream_to_spectrogram_ndarray(stream_list[i])
        Spectrogram_train_data_list.append(Spectrogram_train_data_append)

    Spectrogram_train_data_X = np.stack(Spectrogram_train_data_list)
    Spectrogram_train_data_X = np.expand_dims(Spectrogram_train_data_X, axis = -1)


    initial_array = np.array([[0., 1.]])
    Spectrogram_train_data_Y = np.tile(initial_array, (len(stream_list), 1))

    print(Spectrogram_train_data_X.shape)
    print(Spectrogram_train_data_Y.shape)

    torch.save({'Xnoise': Spectrogram_train_data_X, 'Ynoise': Spectrogram_train_data_Y}, './Spectrogram_data_pth/Spectrogram_noise.pth')

    datadir = './Spectrogram_data_pth'

    train_data_path = os.path.join(datadir, 'Spectrogram_noise.pth')
    train_data = torch.load(train_data_path)
    Xevent = train_data['Xnoise']
    Yevent = train_data['Ynoise']

    print(Xevent.shape)
    print(Yevent.shape)