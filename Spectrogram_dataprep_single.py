import numpy as np
import matplotlib.pyplot as plt
import obspy
from obspy import Trace, UTCDateTime
from scipy.signal import spectrogram
import cv2

from io import BytesIO

# 讀取 SAC 檔案
datadir = './dada_QSIS_test'
file1 = datadir + '/RCEC.09f.5AF30.TW.A1.HLX.2022.09.18.06.43.22.sac'
file2 = datadir + '/RCEC.09f.5AF30.TW.A1.HLY.2022.09.18.06.43.22.sac'
file3 = datadir + '/RCEC.09f.5AF30.TW.A1.HLZ.2022.09.18.06.43.22.sac'
test_event = obspy.read(file1) + obspy.read(file2) + obspy.read(file3)

# 提取 x 軸加速度數據
trace = test_event[0]
x_acceleration = trace.data

# 設置參數
fs = trace.stats.sampling_rate  # 取樣率
nperseg = 256  # 每個段的數據點數
noverlap = nperseg // 2  # 重疊的數據點數

# 計算時頻圖
frequencies, times, Sxx = spectrogram(x_acceleration, fs=fs, nperseg=nperseg, noverlap=noverlap)

# 繪製時頻圖
plt.pcolormesh(times, frequencies, 10 * np.log10(Sxx), shading='auto', cmap='gray')
plt.axis('off')  # 隱藏座標軸


# plt.savefig('./Spectrogram_test/spectrogram_test.png', bbox_inches = 'tight', pad_inches = 0)

# plt.show()
# plt.close()  # 關閉 Matplotlib 的顯示

# # 使用 OpenCV 讀取和縮放圖片
# img = cv2.imread('./Spectrogram_test/spectrogram_test.png')
# img_resized = cv2.resize(img, (150, 100))

# # 保存縮放後的圖片
# cv2.imwrite('./Spectrogram_test/spectrogram_test_resized.png', img_resized)



# 使用 BytesIO 將 Matplotlib 圖片保存到記憶體中
img_stream = BytesIO()
plt.savefig(img_stream, format = 'png', bbox_inches = 'tight', pad_inches = 0)
img_stream.seek(0)

# 使用 OpenCV 讀取並縮放圖片
img = cv2.imdecode(np.frombuffer(img_stream.read(), dtype=np.uint8), 1)
img_resized = cv2.resize(img, (150, 100))

# 顯示原始圖片（可省略）
plt.show()

# 保存縮放後的圖片
cv2.imwrite('./Spectrogram_test/spectrogram_test_resized.png', img_resized)