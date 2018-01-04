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

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2' #Suppress build warnings

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
import sys

plt.ion()

x_cols = { #"nyc_taxi": ["dayofweek", "timeofday"],
           #"test": ["x"],
           #"sunspot": ["incr"],
           #"reddit": ["dayofweek", "timeofday"]
         }

detailedSets = ["reddit"] #with more than 1 csv

differenceSets = ["reddit"]

index = None

def readDataSet(dataSet, dataSetDetailed, s):

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
            (dayofweek, timeofday) = getDayAndTime(timestamp[0], timestamp[1], s)
            daysofweek.append(dayofweek)
            times.append(timeofday)

        daysofweek = pd.Series(daysofweek, index=df.index)
        times = pd.Series(times, index=df.index)
        index = df.index
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

def getX(sequence, s):
    if s.dataSet in x_cols:
        cols = [sequence[key] for key in x_cols[s.dataSet]]

    else:
        col = np.array(sequence['data'])
        col = col[:len(col) - s.predictionStep]
        result_col = []
        for i in range(len(col) - s.lookback):
            result_col.append(col[i:i+s.lookback])
        cols = [result_col, ]
    return np.column_stack(tuple(cols))


def getDayAndTime(date, time, s):
    type = hour_types[s.dataSet]
    timeofday = None
    dayofweek = datetime.strptime(date, date_formats[s.dataSet]).weekday() if date else None
    time = time.split(":") if time else None
    if type == HourType.TO_MINUTE:
        timeofday = float(int(time[0]) * 24 + (int(time[1]) if len(time) > 1 else 0))
    elif type is not None:
        raise Exception("TODO")
    return (dayofweek, timeofday)


def _getArgs():
    if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
        print "USE FLAGS"
        exit(1)
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



def saveResultToFile(dataSet, predictedInput, targetInput, algorithmName, s):
    #inputFileName = 'data/' + dataSet + '.csv'
    #inputFile = open(inputFileName, "rb")

    #csvReader = csv.reader(inputFile)

    # skip header rows
    #csvReader.next()

    outputFileName = './prediction/' + dataSet + '_' + algorithmName + '_pred.csv'
    if s.max_verbosity > 0:
        print "Saving to " + './prediction/' + dataSet + '_' + algorithmName + '_pred.csv'
    outputFile = open(outputFileName, "w")
    csvWriter = csv.writer(outputFile)
    csvWriter.writerow(
        ['timestamp', 'data', 'prediction-' + str(s.predictionStep) + 'step'])
    csvWriter.writerow(['datetime', 'float', 'float'])
    csvWriter.writerow(['', '', ''])

    for i in xrange(len(predictedInput)):
        #row = csvReader.next()
        csvWriter.writerow([i, targetInput[i], predictedInput[i]])

    #inputFile.close()
    outputFile.close()

def normalize(column, sequence):
    mean = np.mean(sequence[column])
    std = np.std(sequence[column])
    sequence[column] = (sequence[column] - mean)/std

def difference(dataset, interval=1):
    newseries = dataset.copy()
    for i in range(interval, len(dataset)):
        newseries['data'][i] = dataset['data'][i] - dataset['data'][i - interval]
    return newseries

# invert differenced value
def inverse_difference(history, yhat, interval=1):
    return yhat + history[interval]



class GruSettings:
    epochs = 3

    useTimeOfDay = True
    useDayOfWeek = True

    retrain_interval = 1500

    numLags = 100
    predictionStep = 5
    batch_size = 128
    online = True
    nodes = 12


    limit_to = None  # None for no limit
    lookback = 50

    max_verbosity = 0

    def __init__(self):
        (_options, _args) = _getArgs()
        self.dataSet = _options.dataSet
        if self.dataSet in detailedSets:
            self.dataSetDetailed = _options.dataSetDetailed

    def finalize(self) :
        self.nTrain = self.retrain_interval * 2


def run_gru(s):




    x_dims = len(x_cols[s.dataSet]) if s.dataSet in x_cols else s.lookback
    random.seed(6)
    np.random.seed(6)
    rnn = Sequential()
    rnn.add(GRU(s.nodes, input_shape=(None,x_dims), kernel_initializer='he_uniform', stateful=False))
    #rnn.add(Dropout(0.2))
    rnn.add(Dense(1, kernel_initializer='he_uniform'))
    opt = adam(lr=0.001, decay=0.0)#1e-3)
    rnn.compile(loss='mse', optimizer=opt)




    # prepare dataset as pyBrain sequential dataset
    sequence = readDataSet(s.dataSet, s.dataSetDetailed, s)
    if s.limit_to:
        sequence = sequence[:s.limit_to]



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

    if s.dataSet in differenceSets:
        backup_sequence = sequence
        sequence = difference(sequence, s.lookback)

    allX = getX(sequence, s)
    allY = np.array(sequence['data'])
    if s.dataSet not in x_cols:
        allY = allY[s.lookback:]
    trainX = allX[0:s.nTrain]
    trainY = allY[s.predictionStep:s.nTrain+s.predictionStep]
    #print trainX[0] * stdSeq + meanSeq
    #print trainY[0] * stdSeq + meanSeq
    trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    #for i in range(epochs):
    rnn.fit(trainX, trainY, epochs=s.epochs, batch_size=s.batch_size, verbose=min(s.max_verbosity, 2))
    #    if i < epochs - 1:
    #        rnn.reset_states()
    for i in xrange(0,s.nTrain):
        targetInput[i] = allY[i+s.predictionStep]


    # nTrain = 1
    # online = True
    # numLags = 0
    # retrain_interval = 1

    #PREP
    #for i in tqdm(xrange(0, nTrain + predictionStep)):
    #    predictedInput[i] = rnn.predict(np.reshape(allX[i], (1,1,x_dims)))

    for i in tqdm(xrange(s.nTrain, len(allX)-s.predictionStep), disable=s.max_verbosity==0):

        if i % s.retrain_interval == 0 and i > s.numLags+s.nTrain and s.online:
            #rnn = Sequential()
            #rnn.add(GRU(nodes, input_shape=(None,2), kernel_initializer='he_uniform'))
            #rnn.add(Dense(1, kernel_initializer='he_uniform'))
            #rnn.compile(loss='mse', optimizer='adam')
            trainX = allX[i-s.nTrain:i]
            trainY = allY[i-s.nTrain:i]
            trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
            rnn.fit(trainX, trainY, epochs=s.epochs, batch_size=s.batch_size, verbose=0)

        targetInput[i] = allY[i+s.predictionStep]
        predictedInput[i] = rnn.predict(np.reshape(allX[i+s.predictionStep], (1,1,x_dims)))
        if s.dataSet in differenceSets:
            predictedInput[i] = inverse_difference(backup_sequence['data'], predictedInput[i], i-1)
            targetInput[i] = inverse_difference(backup_sequence['data'], targetInput[i], i-1)
        trueData[i] = sequence['data'][i]

    predictedInput = (predictedInput * stdSeq) + meanSeq
    targetInput = (targetInput * stdSeq) + meanSeq
    trueData = (trueData * stdSeq) + meanSeq

    saveResultToFile(s.dataSet, predictedInput, targetInput, 'gru', s)

    skipTrain = error_ignore_first[s.dataSet]
    from plot import computeSquareDeviation
    squareDeviation = computeSquareDeviation(predictedInput, targetInput)
    squareDeviation[:skipTrain] = None
    nrmse = np.sqrt(np.nanmean(squareDeviation)) / np.nanstd(targetInput)
    if s.max_verbosity > 0:
        print "", s.nodes, "NRMSE {}".format(nrmse)
    mae = np.nanmean(np.abs(targetInput-predictedInput))
    if s.max_verbosity > 0:
        print "MAE {}".format(mae)

    return nrmse

if __name__ == "__main__":
    settings = GruSettings()
    settings.finalize()
    run_gru(settings)
