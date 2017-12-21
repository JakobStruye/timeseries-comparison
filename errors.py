import pandas as pd
import numpy as np


def get_mae(predicted, actual):
    return np.nanmean(np.abs(predicted - actual))


def get_mse(predicted, actual):
    squared = np.square(predicted - actual)
    return np.nanmean(squared)

def get_nrmse(predicted, actual):
    return np.sqrt(np.nanmean(get_mse(predicted, actual))) / np.nanmean(actual)

def get_mape(predicted, actual):
    return np.nansum(np.abs(predicted - actual)) / np.nansum(np.abs(actual))


names = ["HTM", "GRU"]

expResult = pd.read_csv("prediction/nyc_taxi_TM_pred.csv", header=0, skiprows=[1, 2],
                            names=['step', 'value', 'prediction5'])          
expResult2 = pd.read_csv("prediction/nyc_taxi_gru_pred.csv", header=0, skiprows=[1, 2],
                             names=['step', 'value', 'prediction5'])

ignore_first_n = 3000
ress = [expResult, expResult2]
doRoll = [True, False]
for i in range(len(ress)):
    res = ress[i]
    actual = np.array(res['value'])
    predicted = np.array(res['prediction5'])
    if doRoll[i]:
        actual = np.roll(actual, -5)

    actual[0:ignore_first_n] = np.nan
    predicted[0:ignore_first_n] = np.nan

    print names[i], "MSE:", get_mse(predicted, actual)

    print names[i], "NRMSE:", get_nrmse(predicted, actual)

    print names[i], "MAE:", get_mae(predicted, actual)

    print names[i], "MAPE:", get_mape(predicted, actual)
