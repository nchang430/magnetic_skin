import serial  # pip install pyserial
from datetime import datetime

date_time = datetime.now().strftime("%Y-%m-%d %H-%M")
filename = date_time + " 5X Data.pkl"
serZero = serial.Serial("/dev/tty.usbmodem144202", 115200, timeout=1)

# baker tty.usbmodem144202
startTime = datetime.now().strftime("%Y-%m-%d %H-%M")
# Start up arduino and make sure sensors are working properly
fulldata = []
print("Press reset on the arduino")
# start
while True:
    if serZero.in_waiting:
        zero_bytes = serZero.readline()
        decoded_zero_bytes = zero_bytes.decode("utf-8")
        decoded_zero_bytes = decoded_zero_bytes.strip()
        print("Waiting for reset")
        if decoded_zero_bytes == "Ready!":
            break
fulldata = []
while True:
    try:
        if serZero.in_waiting:
            zero_bytes = serZero.readline()
            decoded_zero_bytes = zero_bytes.decode("utf-8")
            decoded_zero_bytes = decoded_zero_bytes.strip()
            data = [float(x) for x in decoded_zero_bytes.split()]
            fulldata.append(data)
            print(decoded_zero_bytes)
            #     pickle.dump(infodict, "savefilenamekjdshaflkjsdh")
    except:
        break

endTime = datetime.now().strftime("%Y-%m-%d %H-%M")
#t = endTime - startTime
breakpoint()
infodict = {"start_t": startTime, "end_t": endTime, "data": fulldata}
pickle.dump(infodict, "savefilenamekjdshaflkjsdh")
print("DONE COLLECTING")
