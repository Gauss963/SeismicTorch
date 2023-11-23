import cv2
import matplotlib.pyplot as plt
import numpy as np
import obspy
import scipy

from io import BytesIO

# # 提取 x 軸加速度數據
# trace = test_event[0]
# x_acceleration = trace.data

# # 設置參數
# fs = trace.stats.sampling_rate  # 取樣率
# nperseg = 256                   # 每個段的數據點數
# noverlap = nperseg // 2         # 重疊的數據點數

# # 計算時頻圖
# frequencies, times, Sxx = spectrogram(x_acceleration, fs = fs, nperseg = nperseg, noverlap = noverlap)

# # 繪製時頻圖
# plt.pcolormesh(times, frequencies, 10 * np.log10(Sxx), shading = 'auto', cmap = 'gray')
# plt.axis('off')


# # 使用 BytesIO 將 Matplotlib 圖片保存到記憶體中
# img_stream = BytesIO()
# plt.savefig(img_stream, format = 'png', bbox_inches = 'tight', pad_inches = 0)
# img_stream.seek(0)

# # 使用 OpenCV 讀取並縮放圖片
# img = cv2.imdecode(np.frombuffer(img_stream.read(), dtype = np.uint8), 1)
# img_resized = cv2.resize(img, (150, 100))

# # 顯示原始圖片（可省略）
# plt.show()

# # 將縮放後的圖片轉換為 NumPy 陣列
# img_resized_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

# img_resized_array = np.asarray(img_resized_gray)

# print(type(img_resized_array))
# print(img_resized_array.shape)

# # 保存縮放後的圖片
# cv2.imwrite('./Spectrogram_test/spectrogram_test_resized.png', img_resized)


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
        cv2.imwrite('./Spectrogram_test/' + xyz[i] + '.png', img_resized)

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
    colored_image = stream_to_spectrogram_ndarray(test_event)

    print(colored_image.shape)

    # 保存為 PNG 檔案
    cv2.imwrite('./Spectrogram_test/test_3channel_merged_image.png', colored_image)