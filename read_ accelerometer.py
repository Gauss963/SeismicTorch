import time
import numpy as np
from Phidget22.Phidget import *
from Phidget22.Devices.Accelerometer import *

all_data = np.empty((0, 3))  # 假設acceleration有3個維度

def onAccelerationChange(self, acceleration, timestamp):
    global all_data  # 宣告使用全局變量

    acceleration_arr = np.array([acceleration])
    acceleration_arr = acceleration_arr * 9.81 * 100
    
    all_data = np.vstack((all_data, acceleration_arr))

ch = Accelerometer()
ch.setOnAccelerationChangeHandler(onAccelerationChange)

ch.openWaitForAttachment(1000)
ch.setDataRate(500)

min_rate = ch.getMinDataRate()
max_rate = ch.getMaxDataRate()
print(f"Min Data Rate: {min_rate}, Max Data Rate: {max_rate}")

dataRate = ch.getDataRate()
print("DataRate: " + str(dataRate))

start_time = time.perf_counter()
while time.perf_counter() - start_time < 10:
    pass

ch.close()

print("All data collected:", all_data)
print(all_data.shape)
