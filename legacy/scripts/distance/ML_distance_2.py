import os
import glob
import random
import obspy

from obspy import Stream
from obspy import read
from obspy.geodetics import gps2dist_azimuth
from sklearn import svm

import numpy as np 
import scipy.signal as signal
import matplotlib.pyplot as plt 

def get_dist_in_km(input_stream):
    tr = input_stream[0]

    # 測站的緯度和經度
    station_latitude = tr.stats.sac.stla
    station_longitude = tr.stats.sac.stlo


    # 地震源的緯度和經度
    event_latitude = tr.stats.sac.evla
    event_longitude = evlo = tr.stats.sac.evlo


    # 使用 ObsPy 的 gps2dist_azimuth 函數來計算距離
    distance_m, azimuth1, azimuth2 = gps2dist_azimuth(station_latitude, station_longitude, event_latitude, event_longitude)
    distance_in_km = distance_m / 1000

    return distance_in_km

def get_magnitude(input_stream):
    tr = input_stream[0]
    magnitude = tr.stats.sac.mag
    return magnitude

def get_depth(input_stream):
    tr = input_stream[0]
    depth = tr.stats.sac.evdp
    return depth


RCEC = ['5AF48', '5AE11', '5ADF9', '5AE28', '5AF0F', '5AE83', '5AEBA', '5AFE5', '5AE1C']
IES = ['5AE21', '5AE99', '5AE73', '5AF8A', '5AFA8', '5AEE6']
station_list = RCEC + IES

stream_list = []
pag_array = np.array([])
distance_array = np.array([])
magnitude_array = np.array([])
depth_array = np.array([])

minimum_pga = 2.0
minimum_pga = 0.0

file_name_part = 'HLX'

all = 0
useable = 0

for datadir in station_list:

    datadir = os.path.join('./data_QSIS_Event', datadir)

    sac_files_X = glob.glob(f'{datadir}/*{file_name_part}*.sac')
    sac_files_Y = [s.replace('X', 'Y') for s in sac_files_X]
    sac_files_Z = [s.replace('X', 'Z') for s in sac_files_X]


    for i in range(len(sac_files_X)):
        stream = Stream()

        stream_x = read(sac_files_X[i])
        stream_x[0].data = stream_x[0].data * 0.01
        stream += stream_x

        stream_y = read(sac_files_Y[i])
        stream_y[0].data = stream_y[0].data * 0.01
        stream += stream_y

        stream_z = read(sac_files_Z[i])
        stream_z[0].data = stream_z[0].data * 0.01
        stream += stream_z
        
        trace_x = stream_x[0]
        trace_y = stream_y[0]
        trace_z = stream_z[0]
        

        data_len_X = len(stream[0].data)
        data_len_Y = len(stream[1].data)
        data_len_Z = len(stream[2].data)
        include_stream = False
        include_stream = data_len_X == data_len_Y and data_len_Y == data_len_Z and data_len_Z == data_len_X
        include_stream = include_stream and data_len_X >= 40000


        if include_stream:
            all = all + 1
            pga_xyz = np.sqrt(trace_x.data**2 + trace_y.data**2 + trace_z.data**2)
            pga_total = max(pga_xyz)
            pga_total = pga_total * 100

            print("Total PGA (gal): ", pga_total)
            if pga_total >= minimum_pga:
                useable = useable + 1

                stream_list.append(stream)

                distance_in_km = get_dist_in_km(stream)
                magnitude = get_magnitude(stream)
                depth = get_depth(stream)

                pag_array = np.append(pag_array, pga_total, axis = None)
                distance_array = np.append(distance_array, distance_in_km, axis = None)
                magnitude_array = np.append(magnitude_array, magnitude, axis = None)
                depth_array = np.append(depth_array, depth, axis = None)

print(useable)
print(all)

# 合并distance_array和magnitude_array成为特征向量
X = np.column_stack((distance_array, magnitude_array))

# 创建SVM分类器
clf = svm.SVC(kernel='linear')

# 将pag_array大于2.0的数据点标记为1，小于等于2.0的标记为0
y = np.where(pag_array > 2.0, 1, 0)

# 拟合SVM模型
clf.fit(X, y)

# 创建散点图
plt.figure(figsize=(8, 6))

# 用depth_array的值作为颜色深度，将颜色映射到灰度
apm_coeff = 7
c_map = 'viridis'
# scatter = plt.scatter(distance_array, magnitude_array, c=depth_array, cmap=c_map, s=pag_array * apm_coeff, marker='o')

plt.scatter(distance_array[pag_array > 2.0], magnitude_array[pag_array > 2.0], c = depth_array[pag_array > 2.0], cmap = c_map, s = pag_array[pag_array > 2.0] * apm_coeff * 1.25, marker = '+', label = 'pag > 2.0')
plt.scatter(distance_array[pag_array <= 2.0], magnitude_array[pag_array <= 2.0], c = depth_array[pag_array <= 2.0], cmap = c_map, s = pag_array[pag_array <= 2.0] * apm_coeff, marker = 'o', label = 'pag <= 2.0')


# cbar = plt.colorbar(scatter)
# cbar.set_label('Depth')

cbar = plt.colorbar()
cbar.set_label('Depth')

# 添加轴标签
plt.xlabel('Distance (km)')
plt.ylabel('Magnitude (ML)')

# 绘制分隔线
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

# 创建网格点来绘制分隔线
xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 50), np.linspace(ylim[0], ylim[1], 50))
Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# 绘制分隔线
plt.contour(xx, yy, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])

# 显示图形
plt.title('Scatter Plot with Depth and SVM Separation')
plt.grid(True)
plt.savefig('./Dist_ML_Plot/plot_with_SVM_separation.pdf' , dpi = 900, bbox_inches = 'tight')
plt.show()