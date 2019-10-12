import serial  # pip install pyserial
from datetime import datetime
import pickle
import argparse
import numpy as np
import pathlib as path


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description="Collect mag skin data")
    parser.add_argument(
        "--dataroot",
        dest="dataroot",
        help="dataroot folder",
        default="../data/",
        type=str,
    )
    parser.add_argument(
        "--type",
        dest="type",
        help="run type: base, collect",
        default="base",
        type=str,
    )
    parser.add_argument(
        "--batchnum",
        dest="batchnum",
        help="current data batch number",
        default=1,
        type=int,
    )
    parser.add_argument(
        "--tol",
        dest="tol",
        help="difference tolerance",
        default=0.22,
        type=float,
    )

    args = parser.parse_args()
    return args


args = parse_args()


def processData(data):
    assert len(data) == 20
    for j in range(20, 0, -4):
        data = np.delete(data, j - 1)
    assert len(data) == 15
    return data


def isSignal(data, avg_base):
    # checking for signal away from baseline
    diff = data - avg_base
    for j in range(len(diff)):
        if abs(diff[j]) > abs(args.tol * avg_base[j]):
            return True
    return False


def predictDigit(bucket, model):
    print("prediting")
    bucket = np.array(bucket).flatten()
    digit = model.predict(bucket)
    return digit


def collectData(avg_base, model):
    serZero = serial.Serial("/dev/tty.usbmodem146302", 115200, timeout=1)
    startTime = datetime.now()
    # Start up arduino and make sure sensors are working properly
    fulldata = []
    print("Press reset on the arduino")
    # start
    while True:
        if serZero.in_waiting:
            zero_bytes = serZero.readline()
            decoded_zero_bytes = zero_bytes.decode("utf-8")
            decoded_zero_bytes = decoded_zero_bytes.strip()
            # print("Waiting for reset")
            if decoded_zero_bytes == "Ready!":
                break
    fulldata = []
    cur_bucket = []
    digit_running = False
    while True:
        try:
            if serZero.in_waiting:
                zero_bytes = serZero.readline()
                decoded_zero_bytes = zero_bytes.decode("utf-8")
                decoded_zero_bytes = decoded_zero_bytes.strip()
                data = np.array([float(x) for x in decoded_zero_bytes.split()])
                data = processData(data)
                if (
                    isSignal(data, avg_base)
                    and not digit_running
                    and not abs(data[0]) == 0.15
                ):
                    print("Digit Detected")
                    cur_bucket = []
                    cur_bucket.append(data)
                    digit_running = True
                elif digit_running:
                    cur_bucket.append(data)

                if len(cur_bucket) == 19:
                    cur_bucket = np.array(cur_bucket).flatten()
                    cur_bucket = cur_bucket.reshape(1, -1)
                    try:
                        digit = model.predict(cur_bucket)[0]
                        if digit > 0:
                            print(f"Digit: {digit}")
                        # print(f"Digit: {model.predict(cur_bucket)[0]}")
                    except:
                        breakpoint()
                    cur_bucket = []
                    digit_running = False
                fulldata.append(data)
                # print(decoded_zero_bytes)
        except:
            break

    endTime = datetime.now()
    return endTime, startTime, fulldata


def main():
    avg_base = np.zeros(15)
    dataroot = path.Path(args.dataroot)
    base_pkls = list(dataroot.glob("base*"))
    for pkl in base_pkls:
        cur_dict = pickle.load(pkl.open("rb"))
        avg_base = avg_base + np.array(cur_dict["avg_base"])
    avg_base = avg_base / len(base_pkls)

    model = pickle.load(open("best_est.joblib", "rb"))
    endTime, startTime, fulldata = collectData(avg_base, model)

    breakpoint()


if __name__ == "__main__":
    main()

