"""
collectContinuousData.py: collect digit data
author: Nadine Chang
date: Sept 28, 2019
project: Magnetic Skin
"""

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


def collectData():
    serZero = serial.Serial("/dev/tty.usbmodem14302", 115200, timeout=1)
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
        except:
            break

    endTime = datetime.now()
    return endTime, startTime, fulldata


def save(startTime, endTime, fulldata, avg_base):
    filename = f"{args.dataroot}{args.type}_data_batch{args.batchnum}.pkl"
    infodict = {
        "start_t": startTime.strftime("%Y-%m-%d %H-%M"),
        "end_t": endTime.strftime("%Y-%m-%d %H-%M"),
        "total_t": endTime - startTime,
        "data": fulldata,
        "avg_base": avg_base,
        "type": args.type,
    }
    pickle.dump(infodict, open(filename, "wb"))


def main():
    endTime, startTime, fulldata = collectData()

    # remove init data
    if abs(fulldata[0][0]) == 0.15:
        del fulldata[0]
    for i in range(len(fulldata) - 1, -1, -1):
        x = fulldata[i]
        assert len(x) == 20
        x = np.array(x)
        for j in range(20, 0, -4):
            x = np.delete(x, j - 1)
        assert len(x) == 15
        fulldata[i] = x

    if args.type == "base":
        # remove temp, avg data, check size
        avg_base = np.zeros(15)
        for x in fulldata:
            avg_base = avg_base + x
        avg_base = [x / len(fulldata) for x in avg_base]
        print(f"avg base num: {avg_base}")
    else:
        avg_base = np.zeros(15)
        dataroot = path.Path(args.dataroot)
        base_pkls = list(dataroot.glob("base*"))
        for pkl in base_pkls:
            cur_dict = pickle.load(pkl.open("rb"))
            avg_base = avg_base + np.array(cur_dict["avg_base"])
        avg_base = avg_base / len(base_pkls)

    # checking for signal away from baseline
    for i in range(len(fulldata)):
        x = fulldata[i]
        diff = x - avg_base
        for j in range(len(diff)):
            if abs(diff[j]) > abs(args.tol * avg_base[j]):
                print(f"{i}, {abs(diff[j])}, {abs(args.tol * avg_base[j])}")
                continue

    # save(startTime, endTime, fulldata, avg_base)
    print("DONE COLLECTING")
    breakpoint()


if __name__ == "__main__":
    main()

    """
    avg_base = np.array( [ -35.2282729805014,
            -101.85376044568241,
            -530.0596378830087,
            -308.9774373259055,
            284.7739554317551,
            -1091.5754596100278,
            -21.344707520891355,
            -137.83245125348193,
            -864.9162674094721,
            -202.3215877437327,
            -163.25055710306415,
            -666.7459052924787,
            -110.93732590529252,
            -77.79108635097492,
            -364.8731754874653,
        ]
    )

    # checking for signal away from baseline
    for i in range(len(fulldata)):
        x = fulldata[i]
        diff = x - avg_base
        for j in range(len(diff)):
            if abs(diff[j]) > abs(args.tol * avg_base[j]):
                print(f"{i}, {abs(diff[j])}, {abs(args.tol * avg_base[j])}")
                continue
    """

    """
    #get average abs sum signal for base, remove temp, check size of signal
    #replaced with avg vector...better 
    avg_base = 0
    for x in fulldata:
        assert len(x) == 20
        for i in range(20, 0, -4):
            del x[i - 1]
        assert len(x) == 15
        abs_x = [abs(y) for y in x]
        avg_base += sum(abs_x)
    avg_base = avg_base / len(fulldata)
    print(f"avg base num: {avg_base}")
    """

