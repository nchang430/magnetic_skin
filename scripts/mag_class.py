"""
sklearn_sanity.py: multiclass (1 vs all) SVMs sanity checks with sklearn
author: Nadine Chang
date: Sept 28, 2019
project: Magnetic Skin
"""

import sys
import code
import argparse
import numpy as np
import traceback as tb
import pickle
import math
import seaborn as sns
import pandas as pd
from statistics import median
from shinyutils import LazyHelpFormatter
import pathlib as path
from functools import partial
from sklearn import svm
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.svm import LinearSVC
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def parse_args():
    """
    Parse input arguments
    """
    formatter_class = lambda prog: LazyHelpFormatter(
        prog, max_help_position=float("inf"), width=float("inf")
    )
    parser = argparse.ArgumentParser(
        description="Train individual networks for each ROI"
    )
    sub_parsers = parser.add_subparsers(dest="cmd")
    sub_parsers.required = True
    run_parser = sub_parsers.add_parser(
        "run", usage=argparse.SUPPRESS, formatter_class=formatter_class
    )

    data_parser = run_parser.add_argument_group("data")
    data_parser.add_argument(
        "--dataroot",
        dest="dataroot",
        help="dataroot folder",
        default="../data/",
        type=str,
    )
    data_parser.add_argument(
        "--datapoints",
        dest="datapoints",
        help="num of datapoints per sample",
        default=2,
        type=int,
    )

    data_parser.add_argument(
        "--spc", dest="spc", help="samples per class", default=37, type=int
    )
    data_parser.add_argument(
        "--tol",
        dest="tol",
        help="difference tolerance",
        default=0.22,
        type=float,
    )
    # parser.print_help()
    args = parser.parse_args()
    return args


args = parse_args()


def processData(raw_dict):
    """process data"""
    avg_base = np.zeros(15)
    dataroot = path.Path(args.dataroot)
    base_pkls = list(dataroot.glob("base*"))
    for pkl in base_pkls:
        cur_dict = pickle.load(pkl.open("rb"))
        avg_base = avg_base + np.array(cur_dict["avg_base"])
    avg_base = avg_base / len(base_pkls)

    # remove non writing signals
    all_sig_dict = {}
    for digit in range(0, 10):
        raw_data = raw_dict[digit]["data"]
        cur_signals = []
        # checking for signal away from baseline
        for i in range(len(raw_data)):
            x = raw_data[i]
            diff = x - avg_base
            for j in range(len(diff)):
                if abs(diff[j]) > abs(args.tol * avg_base[j]):
                    cur_signals.append(i)
                    continue
        all_sig_dict[digit] = cur_signals

    # remove noise
    tol = 3
    for digit, signals in all_sig_dict.items():
        signals = list(set(signals))
        for i in range(len(signals) - 1, -1, -1):
            num = signals[i]
            if i == len(signals) - 1 and abs(num - signals[i - 1]) > tol:
                del signals[i]
            elif i == 0 and abs(num - signals[i + 1]) > tol:
                del signals[i]
            elif (
                0 < i < len(signals) - 1
                and abs(num - signals[i + 1]) > tol
                and abs(num - signals[i - 1]) > tol
            ):
                del signals[i]
        signals.sort()
        all_sig_dict[digit] = signals
        # print(f"{digit}, {len(signals)}")e
    # breakpoint()

    all_buckets = {}
    for digit, signals in all_sig_dict.items():
        buckets = []
        cur_bucket_num = 0
        cur_bucket = []
        old_sig = signals[0]
        # print(f"{digit}, {len(signals)}")
        # breakpoint()
        for i in range(1, len(signals)):
            sig = signals[i]
            if (3 >= digit or digit >= 7) and abs(sig - old_sig) < 7:
                cur_bucket.append(sig)
            elif 3 < digit < 7 and abs(sig - old_sig) < 13:
                cur_bucket.append(sig)
            else:
                buckets.append(cur_bucket)
                cur_bucket_num += 1
                cur_bucket = [sig]
            old_sig = sig
        # breakpoint()
        # for i in range(len(buckets) - 1, -1, -1):
        #     if len(buckets[i]) < 3:
        #         del buckets[i]
        all_buckets[digit] = buckets

    # find appropriate time lengths per buckets(digit)
    time_median = []
    time_percentile = []
    mid_time_dict = {}
    for d, s in all_buckets.items():
        lens = np.array([(max(x) - min(x) + 1) for x in s])
        mid_time = np.array(
            [math.ceil((max(x) - min(x)) / 2) + min(x) for x in s]
        )
        mid_time_dict[d] = mid_time
        median_len = np.median(lens)
        time_median.append(median_len)
        percentile_len = np.percentile(lens, 80)
        time_percentile.append(percentile_len)
        print(f"{d}, {median_len}, {percentile_len}")

    # set time length 20 for all digits
    half_time = 10
    final_buckets = {}
    final_signals = {}
    for d, t in mid_time_dict.items():
        signals_idx = [
            np.concatenate(
                (
                    np.sort(
                        np.arange(x - 1, x - half_time, -1)
                    ),  # first half before anchor
                    np.arange(x, x + half_time),  # second half after anchor
                )
            )
            for x in t
        ]
        final_buckets[d] = signals_idx

        raw_data = np.array(raw_dict[d]["data"])
        sigs = [raw_data[idx].flatten() for idx in signals_idx]
        final_signals[d] = sigs

    pickle.dump(
        final_buckets, open(f"{args.dataroot}digits_idx_final.pkl", "wb")
    )
    pickle.dump(
        final_signals, open(f"{args.dataroot}digits_signals_final.pkl", "wb")
    )


def loadRawData():
    """load data"""
    raw_dict = {}
    for i in range(0, 10):
        raw_dict[i] = pickle.load(
            open(f"{args.dataroot}collect_data_batch{i+1}.pkl", "rb")
        )
    return raw_dict


def loadData():
    final_signals = pickle.load(
        open(f"{args.dataroot}digits_signals_final.pkl", "rb")
    )
    # train_X = []
    # train_y = []
    # test_X = []
    # test_y = []
    X = []
    y = []
    # 80 20, 30 train, 7 test
    for digit, signal in final_signals.items():
        np.random.shuffle(signal)
        X = X + signal[: args.spc]
        y = y + ([digit] * args.spc)
        assert len(X) == len(y)
        # train_X = train_X + signal[: args.spc]
        # train_y = train_y + ([digit] * args.spc)
        # test_X = test_X + signal[args.spc : args.spc + 7]
        # test_y = test_y + ([digit] * 7)
    # breakpoint()

    return X, y
    # return train_X, train_y, test_X, test_y


def avg_precision(average):
    return average_precision_score(average=self.average)


def runsvm(X, y):
    svc = LinearSVC(dual=False, tol=1e-5)
    pipe = make_pipeline(StandardScaler(), svc)
    breakpoint()

    parameters = {
        "linearsvc__penalty": ["l1", "l2"],
        "linearsvc__C": np.logspace(-10, 10, 20),
    }
    model = GridSearchCV(
        pipe,
        parameters,
        n_jobs=-1,
        iid=True,
        cv=KFold(5, shuffle=True),
        refit=True,
        verbose=1,
    )
    scores = cross_validate(
        model,
        X,
        y,
        scoring="accuracy",
        cv=KFold(5, shuffle=True),
        n_jobs=-1,
        verbose=1,
        return_estimator=True,
    )
    breakpoint()
    # scores['estimator'][0].best_estimator_
    # scores = cross_validate(
    #     model,
    #     X,
    #     y,
    #     scoring=avg_precision("macro"),
    #     cv=KFold(5, shuffle=True),
    #     n_jobs=-1,
    #     verbose=1,
    # )


def main():
    # raw_dict = loadRawData()
    # processed_dict = processData(raw_dict)

    X, y = loadData()
    runsvm(X, y)


if __name__ == "__main__":
    main()
