
import serial #pip install pyserial
from datetime import datetime

date_time = datetime.now().strftime("%Y-%m-%d %H-%M")
filename = date_time + " 5X Data.pkl"
serZero = serial.Serial('/dev/tty.usbmodem146202', 115200, timeout=1)

startTime = datetime.now().strftime("%Y-%m-%d %H-%M")
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

fulldata = []
while True:
    if(serZero.in_waiting):
        zero_bytes = serZero.readline()
        decoded_zero_bytes = zero_bytes.decode('utf-8')
        decoded_zero_bytes = decoded_zero_bytes.strip()
        data = [float(x) for x in decoded_zero_bytes.split()]
        fulldata.append(data)

endTime = datetime.now().strftime("%Y-%m-%d %H-%M")

#pickle fulldata

t = endTime - startTime 
infodict = {'start_t': startTime, 'end_t': endTime, 't': t, 'data': fulldata}
                
                