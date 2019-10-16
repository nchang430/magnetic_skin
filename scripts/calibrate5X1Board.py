import serial
import time
import csv
import array
from datetime import datetime
import random
import math
import os
import matplotlib.pyplot as plt
import numpy as np

###### FUNCTION DEFINITIONS ######
def plotCalibrationData(calibrationData):
    calibrationData = np.asarray(calibrationData)
    
    plt.subplot(131)

    plt.scatter(calibrationData[:,0], calibrationData[:,1], c="red")
    plt.title('XY Plane')
    plt.axis('equal')

    plt.subplot(132)
    plt.scatter(calibrationData[:,1], calibrationData[:,2], c="green")
    plt.title('YZ Plane')
    plt.axis('equal')    

    plt.subplot(133)
    plt.scatter(calibrationData[:,2], calibrationData[:,0], c="blue")
    plt.title('ZX Plane')
    plt.axis('equal')
    
    plt.show()

    return

def collectCalibrationData(calibrationData):
    calibrationCMD = "calibrate\n";

    serZero.write(calibrationCMD.encode('utf-8'));
    print("Start Figure 8 Pattern with the Boards")

    # Collect Calibration Data
    while True:
        if(serZero.in_waiting):
            zero_bytes = serZero.readline()
            decoded_zero_bytes = zero_bytes.decode('utf-8')
            decoded_zero_bytes = decoded_zero_bytes.strip()
            if decoded_zero_bytes == 'End Calibration':
                print(decoded_zero_bytes)
                break
            else:
                print(decoded_zero_bytes)
                data = [float(x) for x in decoded_zero_bytes.split()]
                calibrationData.append(data)
    return calibrationData

###### FUNCTION DEFINITIONS ######
date_time = datetime.now().strftime("%Y-%m-%d %H-%M")
filename = date_time + " E# 5X1 Board Calibration.npz"
serZero = serial.Serial('COM4', 115200, timeout=1)

# Start up arduino and make sure sensors are working properly
print("Press reset on the arduino")
while True:
    if(serZero.in_waiting):
        zero_bytes = serZero.readline()
        decoded_zero_bytes = zero_bytes.decode('utf-8')
        decoded_zero_bytes = decoded_zero_bytes.strip()
        if decoded_zero_bytes == 'Ready!':
            print(decoded_zero_bytes)
            break
        else:
            print(decoded_zero_bytes) 

# Ask user to collect and confirm calibration data quality
calibrationData = [];
calibrationData = collectCalibrationData(calibrationData)
plotCalibrationData(calibrationData)

while(1):
    
    output = input("Does the plot show circles? [Y/N]: ")
    if(output=="Y" or output=="y"):
        print("Continuing to data collection...")
        calibrationData = np.asarray(calibrationData)
        break
    elif(output=="N" or output=="n"):
        print("Returning to calibration...")
        calibrationData = collectCalibrationData(calibrationData)
        plotCalibrationData(calibrationData)
    else:
        print("Unknown input. Asking again")

# Calculate scale and offset for each magnetometer separately
offsets = np.zeros(18)
scales = np.zeros(18)

for i in range(0, 6):
  offsets[3*i:3*i+3] = (np.amin(calibrationData[:,4*i:4*i+3], axis=0)+np.amax(calibrationData[:,4*i:4*i+3], axis=0))/2
  scales[3*i:3*i+3] = (np.amax(calibrationData[:,4*i:4*i+3], axis=0)-np.amin(calibrationData[:,4*i:4*i+3], axis=0))/2
  scales[3*i:3*i+3] = scales[3*i:3*i+3]/(np.sum(scales[3*i:3*i+3])/3)

print(offsets)
print(scales)

# Calculate scaling transform from reference mag to signal mag
calibrationData_notemp = calibrationData[:,[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18,20,21,22]]
calibrated = np.multiply(scales,(calibrationData_notemp-offsets))
plotCalibrationData(calibrated)

# make homogeneous coordinates
calibrated = np.insert(calibrated, 3, 1, axis=1)
calibrated = np.insert(calibrated, 7, 1, axis=1)
calibrated = np.insert(calibrated, 11, 1, axis=1)
calibrated = np.insert(calibrated, 15, 1, axis=1)
calibrated = np.insert(calibrated, 19, 1, axis=1)
calibrated = np.insert(calibrated, 23, 1, axis=1)
print(calibrated[0:5,:])

# find affine transform through least square minimization
A = np.zeros(shape=(5,4,4))
transform_ref = np.zeros(shape=(np.size(calibrated,0),20))
for i in range(1, 6):
    Ai, residuals, rank, s = np.linalg.lstsq(calibrated[:,0:4], calibrated[:,4*i:4*i+4])
    A[(i-1),:,:] = Ai
    transform_ref[:,4*(i-1):4*i] = calibrated[:,0:4].dot(Ai)
print(A)

# plot err between signal and transform_reference
plt.plot(calibrated[:,4]-transform_ref[:,0], c="red")
plt.plot(calibrated[:,5]-transform_ref[:,1], c="green")
plt.plot(calibrated[:,6]-transform_ref[:,2], c="blue")
plt.title('Error Over Sample Number')
plt.show()

plt.plot(calibrated[:,8]-transform_ref[:,4], c="red")
plt.plot(calibrated[:,9]-transform_ref[:,5], c="green")
plt.plot(calibrated[:,10]-transform_ref[:,6], c="blue")
plt.title('Error Over Sample Number')
plt.show()

# save offsets, scale, and A to file
np.savez(filename, svd_data=transform_ref, data=calibrationData, affine=A, offsets=offsets, scales=scales)
