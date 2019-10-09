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
        "--spc", dest="spc", help="samples per class", default=50, type=int
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
    breakpoint()
    for i in range(0, 10):
        raw_data = raw_dict[i]["data"]
        breakpoint()
        # checking for signal away from baseline
        for i in range(len(raw_data)):
            x = raw_data[i]
            diff = x - avg_base
            for j in range(len(diff)):
                if abs(diff[j]) > abs(args.tol * avg_base[j]):
                    print(f"{i}, {abs(diff[j])}, {abs(args.tol * avg_base[j])}")
                    continue

    #     ys[i * 10 : (i * 10) + 10] = np.arange(0, 10)
    # for i in range(10):
    #     assert(sum(ys == i))

    # f = pickle.load(open("digits_data_batch1.pkl", "rb"))
    # Xs = f["data"]
    # times = f["start_t"] - f["end_t"]

    # assert len(ys) == len(Xs)
    # final_dict = {'Xs'=Xs, 'ys'=ys}
    # pickle.dump(final_dict, open("final_digits_data.pkl", "w"))


def loadData():
    """load data"""
    raw_dict = {}
    for i in range(0, 10):
        raw_dict[i] = pickle.load(
            open(f"{args.dataroot}collect_data_batch{i+1}.pkl", "rb")
        )
    return raw_dict


def avg_precision(average):
    return average_precision_score(average=self.average)


def runsvm(X, y):
    svc = LinearSVC(dual=False, tol=1e-5)
    pipe = make_pipeline(StandardScaler(), svc)

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
    )
    breakpoint()
    scores = cross_validate(
        model,
        X,
        y,
        scoring=avg_precision("macro"),
        cv=KFold(5, shuffle=True),
        n_jobs=-1,
        verbose=1,
    )


def main():
    raw_dict = loadData()
    breakpoint()
    processed_dict = processData(raw_dict)
    # runsvm(X, y)


if __name__ == "__main__":
    main()
