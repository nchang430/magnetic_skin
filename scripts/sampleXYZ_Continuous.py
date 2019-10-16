import serial
import time
import csv
import array
import datetime
import random
import math
import os
import numpy as np


filename = "2019-10-10 5X1 Delay5 XYZTF T460 C E7"
fulldata  = []

startString = "Start Stream\n"
stopString = "Stop Stream\n"

locations = []
# first number is the encoder value from calibration point on acrylic board
# second number is top left corner of 40x40 skin
topleft_X = 205+18.5
topleft_Y = -29.5+1.5
# third number is offset for skin thickness
Z = 132+8+1.5 
speed = 0.35

# set up 3000 locations for sampling
for idx in range(0,3000):
    cx = round(random.uniform(topleft_X, topleft_X+40), 1)
    cy = round(random.uniform(topleft_Y, topleft_Y+40), 1)
    cz = round(random.uniform(Z-1.5, Z),1)
    s1 =  "#1 G1 X"+str(cx)+" Y"+str(cy)+"Z145 F1\n"
    s2 =  "#2 G1 X"+str(cx)+" Y"+str(cy)+" Z"+str(cz)+" F"+str(speed)+"\n"
    locations.append(s1)
    locations.append(s2)

time.sleep(3)

serZero = serial.Serial('COM4', 115200, timeout=1)

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

serArm = serial.Serial('COM9', 115200, timeout=1)
serArm.flushInput()
# uarm swift pro takes a long time to setup
time.sleep(15)
serZero.flushInput()
serArm.flushInput()

#move to start position to collect resting signal
startPos = "G1 X224 Y-28 Z160 F1\n"
serArm.write(startPos.encode('utf-8'))
time.sleep(10)

calibrationCMD = "calibrate\n";
calibrationData = []
serZero.write(calibrationCMD.encode('utf-8'));

# Collect Resting Data
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

with open(filename + " RestingPoints.txt","a") as f:
    for item in calibrationData:
        f.write("%s\n" % item)
calibrationData.clear()

coordCMD = "#4 P2220\n" # get cartesian coordinates
moveUp = "G2204 X0 Y0 Z8 F1\n"
time.sleep(1)


for x in range(0,len(locations)): 
    pos1 = locations[x]
    print(pos1)
    serArm.write(pos1.encode('utf-8'))
    print("start position number: %d" % (x))
    while True:
        if(serArm.in_waiting):
            arm_bytes = serArm.readline()
            decoded_arm_bytes = arm_bytes.decode('utf-8')
            decoded_arm_bytes = decoded_arm_bytes.strip()
            if decoded_arm_bytes == "$1 ok":
                break
            elif decoded_arm_bytes == "$2 ok":
                serArm.write(coordCMD.encode('utf-8'))
            elif "$4 ok" in decoded_arm_bytes:
                robotString = ''.join( c for c in decoded_arm_bytes if  c not in '$XYZ')
                robotXYZ = robotString.split(" ")[2:5]
                print(robotXYZ)
                serZero.write(startString.encode('utf-8'))
            else:
                print("something else came in arm serial")
                print(decoded_arm_bytes)


        if(serZero.in_waiting):
            zero_bytes = serZero.readline()
            decoded_zero_bytes = zero_bytes.decode('utf-8')
            decoded_zero_bytes = decoded_zero_bytes.strip()
            if decoded_zero_bytes == 'End':
                serArm.write(moveUp.encode('utf-8'))
                time.sleep(2)
                serZero.flushInput()
                break
            else:
                print(decoded_zero_bytes) 
                data = [float(x) for x in decoded_zero_bytes.split()]
                fulldata.append(robotXYZ + data)


    print("end iteration number: %d" % (x))

    with open(filename + ".txt","a") as f:
        for item in fulldata:
            f.write("%s\n" % item)
    fulldata.clear()
    time.sleep(3)

# done with all the sampling, rest the robot arm in safe place

moveToRest = "G1 X177 Y2 Z145 F1\n" 
relaxMotors = "M2019\n"
attachMotors = "M17\n"

serArm.write(moveUp.encode('utf-8'))
time.sleep(5)
serArm.write(moveToRest.encode('utf-8'))
time.sleep(40)
serArm.write(relaxMotors.encode('utf-8'))
time.sleep(60)

# shutdown computer to remove power from arduino
os.system('shutdown -s')
exit()