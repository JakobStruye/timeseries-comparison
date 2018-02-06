import pandas as pd
import numpy as np
from sys import argv


def get_mae(predicted, actual):
    return np.nanmean(np.abs(predicted - actual))


def get_mse(predicted, actual):
    squared = np.square(predicted - actual)
    return np.nanmean(squared)

def get_nrmse(predicted, actual):
    return np.sqrt(get_mse(predicted, actual)) / np.nanstd(actual)

def get_mape(predicted, actual, ignore=None):
    if not ignore:
        ignore = 0
    return np.nansum(np.abs(predicted[ignore:] - actual[ignore:])) / np.nansum(np.abs(actual[ignore:]))
    #return np.nansum(np.abs(predicted - actual)) / np.nansum(np.abs(actual))

def get_mase(predicted, actual, actual_rolled, ignore=None):
    print predicted.shape
    if not ignore:
        for i in range(len(predicted)):
            if not np.isnan(predicted[i]):
                ignore = i
                break
    return np.nanmean(np.abs(predicted[ignore:] - actual[ignore:]) / get_mae(actual_rolled[ignore:], actual[ignore:]))


if __name__ == '__main__':
    names = ["HTM", "GRU"]
    expName = "reddit" if len(argv) < 2 else argv[1]
    if expName != "rssi":
        expResult = pd.read_csv("prediction/" + expName + "_TM_pred.csv", header=0, skiprows=[1, 2],
                                names=['step', 'value', 'prediction5'])
    expResult2 = pd.read_csv("prediction/" + expName + "_gru_pred.csv", header=0, skiprows=[1, 2],
                                 names=['step', 'value', 'prediction5'])

    ignore_first_n = 4320

    if expName == "rssi":
        ress = [expResult2]
    else:
        ress = [expResult, expResult2]

    if expName == "reddit":
        season = 24
    elif expName == "nyc_taxi":
        season = 48
    elif expName == "rssi":
        season = 1440
    doRoll = [False, False]

    for i in range(len(ress)):
        res = ress[i]
        actual = np.array(res['value'])
        predicted = np.array(res['prediction5'])
        if doRoll[i]:
            actual = np.roll(actual, -5)

        actual_rolled = np.roll(actual, 5)
        actual_rolled_seasonal = np.roll(actual, season)


        actual[0:ignore_first_n] = np.nan
        actual_rolled[0:ignore_first_n] = np.nan
        actual_rolled_seasonal[0:ignore_first_n] = np.nan
        predicted[0:ignore_first_n] = np.nan

        print names[i], "MSE:", get_mse(predicted, actual)

        print names[i], "NRMSE:", get_nrmse(predicted, actual)

        print names[i], "MAE:", get_mae(predicted, actual)

        print names[i], "MAPE:", get_mape(predicted, actual)

        print names[i], "MASE:", get_mase(predicted, actual, actual_rolled)

        print names[i], "MASE (seasonal):", get_mase(predicted, actual, actual_rolled_seasonal)
