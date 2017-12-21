# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2016, Numenta, Inc.  Unless you have an agreement
# with Numenta, Inc., for a separate license for this software code, the
# following terms and conditions apply:
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero Public License version 3 as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU Affero Public License for more details.
#
# You should have received a copy of the GNU Affero Public License
# along with this program.  If not, see http://www.gnu.org/licenses.
#
# http://numenta.org/licenses/
# ----------------------------------------------------------------------

import csv
from optparse import OptionParser

from scipy import random

import pandas as pd
from htmresearch.support.sequence_learning_utils import *

import numpy as np
from dataset_settings import *

from keras.models import Sequential
from keras.layers import Dense, GRU
from keras.optimizers import adam

from datetime import datetime
from tqdm import tqdm

plt.ion()

x_cols = { #"nyc_taxi": ["dayofweek", "timeofday"],
           #"test": ["x"],
           #"sunspot": ["incr"],
           #"reddit": ["dayofweek", "timeofday"]
         }

detailedSets = ["reddit"] #with more than 1 csv


limit_to = None  # None for no limit
lookback = 50


def readDataSet(dataSet, dataSetDetailed):

    if dataSet in detailedSets:
        dataSource = "%s/%s" % (dataSet,dataSetDetailed)
    else:
        dataSource = dataSet

    filePath = 'data/'+dataSource+'.csv'

    if dataSet=='nyc_taxi':
        df = pd.read_csv(filePath, header=0, skiprows=[1,2],
                         names=['time', 'data', 'timeofday', 'dayofweek'])
        sequence = df['data']
        dayofweek = df['dayofweek']
        timeofday = df['timeofday']

        seq = pd.DataFrame(np.array(pd.concat([sequence, timeofday, dayofweek], axis=1)),
                           columns=['data', 'timeofday', 'dayofweek'])

    elif "sunspot" in dataSet:
        df = pd.read_csv(filePath, header=0, skiprows=[],
                         names=['year','month','day','val','spots','stdev','number_of_observations','indicator','incr'])
        sequence = df['spots']
        incr = df['incr']
        seq = pd.DataFrame(np.array(pd.concat([sequence, incr], axis=1)),
                           columns=['data', 'incr'])
    elif dataSet == 'reddit':
        df = pd.read_csv(filePath, header=0, skiprows=[1, ],
                         names=['time', 'count'])
        sequence = df['count']

        timestamps = df["time"]
        daysofweek = []
        times = []

        for timestamp in timestamps:
            timestamp = timestamp.split(" ")
            (dayofweek, timeofday) = getDayAndTime(timestamp[0], timestamp[1])
            daysofweek.append(dayofweek)
            times.append(timeofday)

        daysofweek = pd.Series(daysofweek, index=df.index)
        times = pd.Series(times, index=df.index)
        seq = pd.DataFrame(np.array(pd.concat([sequence, times, daysofweek], axis=1)),
                           columns=['data', 'timeofday', 'dayofweek'])

    elif "test" in dataSet:
        df = pd.read_csv(filePath, header=0, skiprows=[],
                         names=['x', 'y'])
        sequence = df['y']
        incr = df['x']
        seq = pd.DataFrame(np.array(pd.concat([sequence, incr], axis=1)),
                           columns=['data', 'x'])
    else:
        raise(' unrecognized dataset type ')

    return seq

def getX(sequence):
    if dataSet in x_cols:
        cols = [sequence[key] for key in x_cols[dataSet]]

    else:
        col = np.array(sequence['data'])
        col = col[:len(col) - predictionStep]
        result_col = []
        for i in range(len(col) - lookback):
            result_col.append(col[i:i+lookback])
        cols = [result_col, ]
    return np.column_stack(tuple(cols))


def getDayAndTime(date, time):
    type = hour_types[dataSet]
    timeofday = None
    dayofweek = datetime.strptime(date, date_formats[dataSet]).weekday() if date else None
    time = time.split(":") if time else None
    if type == HourType.TO_MINUTE:
        timeofday = float(int(time[0]) * 24 + (int(time[1]) if len(time) > 1 else 0))
    elif type is not None:
        raise Exception("TODO")
    return (dayofweek, timeofday)


def _getArgs():
    parser = OptionParser(usage="%prog PARAMS_DIR OUTPUT_DIR [options]"
                                "\n\nCompare TM performance with trivial predictor using "
                                "model outputs in prediction directory "
                                "and outputting results to result directory.")
    parser.add_option("-d",
                      "--dataSet",
                      type=str,
                      default='nyc_taxi',
                      dest="dataSet",
                      help="DataSet Name, choose from sine, SantaFe_A, MackeyGlass")

    parser.add_option("-e",
                      "--dataSetDetailed",
                      type=str,
                      default='2007-10_hour',
                      dest="dataSetDetailed",
                      help="DataSet Detailed Name, currently only for the reddit set")


    (options, remainder) = parser.parse_args()
    #print options

    return options, remainder



def saveResultToFile(dataSet, predictedInput, targetInput, algorithmName):
    #inputFileName = 'data/' + dataSet + '.csv'
    #inputFile = open(inputFileName, "rb")

    #csvReader = csv.reader(inputFile)

    # skip header rows
    #csvReader.next()

    outputFileName = './prediction/' + dataSet + '_' + algorithmName + '_pred.csv'
    print "Saving to " + './prediction/' + dataSet + '_' + algorithmName + '_pred.csv'
    outputFile = open(outputFileName, "w")
    csvWriter = csv.writer(outputFile)
    csvWriter.writerow(
        ['timestamp', 'data', 'prediction-' + str(predictionStep) + 'step'])
    csvWriter.writerow(['datetime', 'float', 'float'])
    csvWriter.writerow(['', '', ''])

    for i in xrange(len(sequence)):
        #row = csvReader.next()
        csvWriter.writerow([i, targetInput[i], predictedInput[i]])

    #inputFile.close()
    outputFile.close()

def normalize(column, sequence):
    mean = np.mean(sequence[column])
    std = np.std(sequence[column])
    sequence[column] = (sequence[column] - mean)/std


if __name__ == "__main__":

    (_options, _args) = _getArgs()
    dataSet = _options.dataSet
    dataSetDetailed = _options.dataSetDetailed

    x_dims = len(x_cols[dataSet]) if dataSet in x_cols else lookback
    random.seed(6)
    np.random.seed(6)
    nodes = 160
    rnn = Sequential()
    rnn.add(GRU(nodes, input_shape=(None,x_dims), kernel_initializer='he_uniform'))
    #rnn.add(Dropout(0.2))
    rnn.add(Dense(1, kernel_initializer='he_uniform'))
    adam = adam(lr=0.001, decay=0.0)#1e-3)
    rnn.compile(loss='mse', optimizer=adam)

    epochs = 120


    useTimeOfDay = True
    useDayOfWeek = True

    retrain_interval = 1500
    nTrain = retrain_interval*2
    numLags = 100
    predictionStep = 20
    batch_size = 1
    online=True


    # prepare dataset as pyBrain sequential dataset
    sequence = readDataSet(dataSet, dataSetDetailed)
    if limit_to:
        sequence = sequence[:limit_to]



    # standardize data by subtracting mean and dividing by std
    meanSeq = np.mean(sequence['data'])
    stdSeq = np.std(sequence['data'])
    sequence['data'] = (sequence['data'] - meanSeq)/stdSeq

    for key in sequence.keys():
        if key != "data":
            normalize(key, sequence)

    predictedInput = np.zeros((len(sequence),))
    targetInput = np.zeros((len(sequence),))
    trueData = np.zeros((len(sequence),))


    allX = getX(sequence)
    allY = np.array(sequence['data'])
    if dataSet not in x_cols:
        allY = allY[lookback:]
    trainX = allX[0:nTrain]
    trainY = allY[predictionStep:nTrain+predictionStep]
    #print trainX[0] * stdSeq + meanSeq
    #print trainY[0] * stdSeq + meanSeq
    trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    rnn.fit(trainX, trainY, epochs=epochs, batch_size=batch_size, verbose=2)

    for i in xrange(0,nTrain):
        targetInput[i] = allY[i+predictionStep]


    # nTrain = 1
    # online = True
    # numLags = 0
    # retrain_interval = 1
    for i in tqdm(xrange(nTrain, len(allX)-predictionStep)):

        if i % retrain_interval == 0 and i > numLags+nTrain and online:
            #rnn = Sequential()
            #rnn.add(GRU(nodes, input_shape=(None,2), kernel_initializer='he_uniform'))
            #rnn.add(Dense(1, kernel_initializer='he_uniform'))
            #rnn.compile(loss='mse', optimizer='adam')
            trainX = allX[i-nTrain:i]
            trainY = allY[i-nTrain:i]
            trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
            rnn.fit(trainX, trainY, epochs=epochs, batch_size=batch_size, verbose=0)

        targetInput[i] = allY[i+predictionStep]
        predictedInput[i] = rnn.predict(np.reshape(allX[i+predictionStep], (1,1,x_dims)))
        trueData[i] = sequence['data'][i]

    predictedInput = (predictedInput * stdSeq) + meanSeq
    targetInput = (targetInput * stdSeq) + meanSeq
    trueData = (trueData * stdSeq) + meanSeq

    saveResultToFile(dataSet, predictedInput, targetInput, 'gru')

    skipTrain = error_ignore_first[dataSet]
    from plot import computeSquareDeviation
    squareDeviation = computeSquareDeviation(predictedInput, targetInput)
    squareDeviation[:skipTrain] = None
    nrmse = np.sqrt(np.nanmean(squareDeviation)) / np.nanstd(targetInput)
    print "", nodes, "NRMSE {}".format(nrmse)
    mae = np.nanmean(np.abs(targetInput-predictedInput))
    print "MAE {}".format(mae)