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

from optparse import OptionParser

from scipy import random

import pandas as pd

import numpy as np
from dataset_settings import *

from keras.models import Sequential
from keras.layers import Dense, GRU, LSTM, Dropout
from keras.optimizers import adam

from datetime import datetime
from tqdm import tqdm
import sys
from data_processing import DataProcessor
import errors
import tensorflow as tf
import keras.backend as K
from keras.callbacks import Callback

x_cols = { #"nyc_taxi": ["dayofweek", "timeofday"],
           #"test": ["x"],
           #"sunspot": ["incr"],
           #"reddit": ["dayofweek", "timeofday"]
         }

detailedSets = ["reddit"] #with more than 1 csv

differenceSets = []#["reddit"]

index = None

global_step = tf.Variable(0, name='global_step', trainable=False, dtype=tf.int32)
increment_global_step_op = tf.assign(global_step, global_step+1)
reset_global_step_op = tf.assign(global_step, 0)

batches = tf.get_variable("batches", [215, 7, 1, 1], dtype=tf.float32,
  initializer=tf.zeros_initializer)
images_placeholder = tf.placeholder(tf.float32, shape=(215,7,1,1))
batches_op = tf.assign(batches, images_placeholder)


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

    return options, remainder

class LossCallback(Callback):

    def on_batch_end(self, batch, logs={}):
        K.get_session().run(increment_global_step_op)

    def on_epoch_begin(self, epoch, logs={}):
        K.get_session().run(reset_global_step_op)

def mase_loss(y_true, y_pred):
    return K.mean(K.abs(y_true - y_pred) / K.mean(K.abs(y_true - tf.gather(batches, global_step))))

def configure_batches(season_length, batch_size, target_input):
    new_batches = []
    start = season_length
    counter = 0
    while start < len(target_input):
        counter += 1
        preds = target_input[start - season_length+5:min(start-season_length+batch_size+5, len(target_input))]

        new_batches.append(preds)
        start = start + batch_size
    batches2 = np.reshape(np.array(new_batches), (215,7,1,1))
    fd = { images_placeholder: batches2}
    result = K.get_session().run(batches_op, feed_dict=fd)




class GruSettings:
    epochs = 5

    useTimeOfDay = True
    useDayOfWeek = True

    retrain_interval = 1500

    numLags = 100
    predictionStep = 5
    batch_size = 7
    online = False
    nodes = 12


    limit_to = None  # None for no limit
    lookback = 20

    max_verbosity = 2

    dataSet = None
    dataSetDetailed = None

    lr = 0.001

    def __init__(self):
        (_options, _args) = _getArgs()
        self.dataSet = _options.dataSet
        if self.dataSet in detailedSets:
            self.dataSetDetailed = _options.dataSetDetailed

    def finalize(self) :
        self.nTrain = self.retrain_interval * 1

def run_gru(s):

    x_dims = len(x_cols[s.dataSet]) if s.dataSet in x_cols else s.lookback
    random.seed(6)
    np.random.seed(6)
    rnn = Sequential()
    rnn.add(GRU(s.nodes, input_shape=(None,x_dims), kernel_initializer='he_uniform', stateful=False))

    #rnn.add(Dropout(0.15))
    rnn.add(Dense(1, kernel_initializer='he_uniform'))
    opt = adam(lr=s.lr, decay=0.0)#1e-3)
    rnn.compile(loss='mae', optimizer=opt)

    # prepare dataset as pyBrain sequential dataset
    sequence = readDataSet(s.dataSet, s.dataSetDetailed, s)
    if s.limit_to:
        sequence = sequence[:s.limit_to]


    dp = DataProcessor()
    # standardize data by subtracting mean and dividing by std
    #(meanSeq, stdSeq) = dp.normalize('data', sequence)

    dp.windowed_normalize(sequence)


    for key in sequence.keys():
        if key != "data":
            dp.normalize(key, sequence)

    predictedInput = np.zeros((len(sequence),))
    targetInput = np.zeros((len(sequence),))
    trueData = np.zeros((len(sequence),))

    if s.dataSet in differenceSets:
        predictedInputNodiff = np.zeros((len(sequence),))
        targetInputNodiff = np.zeros((len(sequence),))

    if s.dataSet in differenceSets:
        backup_sequence = sequence
        sequence = dp.difference(sequence, s.lookback)

    allX = getX(sequence, s)
    allY = np.array(sequence['data'])

    allX = allX[48:]
    allY = allY[48:]
    #if s.dataSet not in x_cols:
    #    allY = allY[s.lookback:]
    trainX = allX[0:s.nTrain]
    trainY = allY[s.predictionStep:s.nTrain+s.predictionStep]
    trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    curBatch = 1.0
    callback = LossCallback()
    temp_set = np.array(sequence['data'])[:48+s.nTrain+5]
    configure_batches(48, s.batch_size, np.reshape(temp_set, (temp_set.shape[0], 1, 1)))
    rnn.fit(trainX, trainY, epochs=s.epochs, batch_size=s.batch_size, verbose=min(s.max_verbosity, 2), callbacks=[callback])
    for i in xrange(0,s.nTrain):
        targetInput[i] = allY[i+s.predictionStep]

    for i in tqdm(xrange(s.nTrain+s.predictionStep, len(allX)), disable=s.max_verbosity==0):
        if i % s.retrain_interval == 0 and i > s.numLags+s.nTrain and s.online:
            trainX = allX[i-s.nTrain-s.predictionStep:i-s.predictionStep]
            trainY = allY[i-s.nTrain:i]
            trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
            temp_set = np.array(sequence['data'])[i-s.nTrain-s.predictionStep - 48 :i]
            configure_batches(48, s.batch_size, np.reshape(temp_set, (temp_set.shape[0], 1, 1)))
            rnn.fit(trainX, trainY, epochs=s.epochs, batch_size=s.batch_size, verbose=2, callbacks=[callback])

        targetInput[i] = allY[i+s.predictionStep]
        predictedInput[i] = rnn.predict(np.reshape(allX[i], (1,1,x_dims)))
        if s.dataSet in differenceSets:
            predictedInputNodiff[i] = predictedInput[i]
            targetInputNodiff[i] = targetInput[i]
            predictedInput[i] = dp.inverse_difference(backup_sequence['data'], predictedInput[i], i-1)
            targetInput[i] = dp.inverse_difference(backup_sequence['data'], targetInput[i], i-1)
        predictedInput[0] = 0
        trueData[i] = sequence['data'][i]

    #predictedInput = dp.denormalize(predictedInput, meanSeq, stdSeq)
    #targetInput = dp.denormalize(targetInput, meanSeq, stdSeq)
    dp.windowed_denormalize(predictedInput, targetInput)
    if s.dataSet in differenceSets:

        # predictedInputNodiff = dp.denormalize(predictedInputNodiff)
        # targetInputNodiff = dp.denormalize(targetInputNodiff)
        pass
    #trueData = (trueData * stdSeq) + meanSeq

    dp.saveResultToFile(s.dataSet, predictedInput, targetInput, 'gru', s.predictionStep, s.max_verbosity)
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

    if s.dataSet in differenceSets:
        dp.saveResultToFile(s.dataSet, predictedInputNodiff, targetInputNodiff, 'gru_nodiff', s.predictionStep, s.max_verbosity)
        squareDeviation = computeSquareDeviation(predictedInputNodiff, targetInputNodiff)
        squareDeviation[:skipTrain] = None
        nrmse = np.sqrt(np.nanmean(squareDeviation)) / np.nanstd(targetInputNodiff)
        if s.max_verbosity > 0:
            print "", s.nodes, "NRMSE {}".format(nrmse)
        mae = np.nanmean(np.abs(targetInputNodiff-predictedInputNodiff))
        if s.max_verbosity > 0:
            print "MAE {}".format(mae)
        mase = errors.get_mase(predictedInput, targetInput, np.roll(targetInput, 24))
        if s.max_verbosity > 0:
            print "MAE {}".format(mae)
    return nrmse

if __name__ == "__main__":
    settings = GruSettings()
    settings.finalize()
    run_gru(settings)
