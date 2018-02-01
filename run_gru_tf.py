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


from datetime import datetime
from tqdm import tqdm
import sys
from data_processing import DataProcessor
import errors
import tensorflow as tf
import rnn_tf
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, GRU
from keras.optimizers import adam
from adaptive_normalization import AdaptiveNormalizer


detailedSets = ["reddit"] #with more than 1 csv

index = None

global_step = None
increment_global_step_op = None
reset_global_step_op = None

batches = None
images_placeholder = None
batches_op = None



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
    elif "rssi" in dataSet:
        df = pd.read_csv(filePath, header=0, skiprows=[],
                         names=['data'])
        sequence = df['data']
        seq = pd.DataFrame(np.array(pd.concat([sequence,], axis=1)),
                           columns=['data'])
    elif "sin" in dataSet:
        df = pd.read_csv(filePath, header=0, skiprows=[],
                         names=['data'])
        sequence = df['data']
        seq = pd.DataFrame(np.array(pd.concat([sequence, ], axis=1)),
                           columns=['data'])
    else:
        raise(' unrecognized dataset type ')

    return seq

def getX(seq_full, s):
    """
    Converts the full sequence of data to a lookback format:
    x[t] contains the _lookback_ latest values known at time t

    :param seq_full: Full set of values
    :param s: Settings
    :return: Lookback format of x
    """

    #Ignore the final _predictionStep_ here; they have no known y
    col = seq_full[:len(seq_full) - s.predictionStep + 1]
    result_col = []
    counter = 0
    for i in range(s.front_buffer, len(col)):
        counter += 1
        result_col.append(col[i-s.lookback: i])
    return np.array(result_col)


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

class GruSettings:
    epochs = 75

    useTimeOfDay = True
    useDayOfWeek = True

    retrain_interval = 1500

    predictionStep = 5
    batch_size = 17
    online = True
    nodes = 58


    limit_to = None  # None for no limit
    lookback = 50
    season = 48
    max_verbosity = 2

    normalization_type = 'default' #'default', 'windowed' or 'AN' (adaptive normalization
    implementation = "tf" #"tf" or "keras"

    dataSet = None
    dataSetDetailed = None

    lr = 0.001

    def __init__(self):
        (_options, _args) = _getArgs()
        self.dataSet = _options.dataSet
        if self.dataSet in detailedSets:
            #For types of data with multiple datasets (e.g. per hour and per minute for Reddit)
            self.dataSetDetailed = _options.dataSetDetailed
        else:
            self.dataSetDetailed = None

    def finalize(self) :
        """ To be called after setting everything"""
        self.nTrain = max(self.retrain_interval * 2, self.season * 3)
        if self.batch_size is None:
            #No batch learning
            self.batch_size = self.nTrain
        if self.nTrain % self.batch_size != 0:
            if self.max_verbosity > 0:
                print "Adding", self.batch_size - (self.nTrain % self.batch_size), "to nTrain", self.nTrain
            self.nTrain += self.batch_size - (self.nTrain % self.batch_size)
        self.numLags = 0.25 * self.nTrain #Don't immediately retrain

        #The first time at which we can actually predict: need enough headroom for both MASE calculation
        #and filling the lookback window
        self.front_buffer = max(self.season - self.predictionStep, self.lookback)


def run_gru(s):
    global global_step
    global increment_global_step_op
    global reset_global_step_op
    global batches
    global images_placeholder
    global batches_op
    global_step = tf.Variable(0, name='global_step', trainable=False, dtype=tf.int32)
    increment_global_step_op = tf.assign(global_step, global_step + 1)
    reset_global_step_op = tf.assign(global_step, 0)

    batches = tf.get_variable("batches", [s.nTrain / int(s.batch_size), s.batch_size, 1, 1], dtype=tf.float32,
                              initializer=tf.zeros_initializer)
    images_placeholder = tf.placeholder(tf.float32, shape=(s.nTrain / int(s.batch_size), s.batch_size, 1, 1))
    batches_op = tf.assign(batches, images_placeholder)


    x_dims = s.lookback
    random.seed(6)
    np.random.seed(6)
    tf.set_random_seed(6)
    if s.implementation == "keras":
        rnn = Sequential()
        rnn.add(GRU(s.nodes, input_shape=(None,x_dims), kernel_initializer='he_uniform'))

        #rnn.add(Dropout(0.15))
        rnn.add(Dense(1, kernel_initializer='he_uniform'))
        opt = adam(lr=s.lr, decay=0.0)#1e-3)
        rnn.compile(loss='mae', optimizer=opt)
    elif s.implementation == "tf":
        data = tf.placeholder(tf.float32, [None, 1,  s.lookback])  # Number of examples, number of input, dimension of each input
        target = tf.placeholder(tf.float32, [None, 1])
        cell = rnn_tf.GRUCell(s.nodes)
        #cell = rnn_tf.LSTMCell(s.nodes)
        #cell = tf.nn.rnn_cell.GRUCell(s.nodes)
        val, _ = tf.nn.dynamic_rnn(cell, data, dtype=tf.float32)
        dense = tf.layers.dense(val, 1)
        pred = tf.reshape(dense, (tf.shape(dense)[0], 1))
        optimizer = tf.train.AdamOptimizer(learning_rate=s.lr)
        #cost = tf.losses.mean_squared_error(target, pred)
        cost = tf.reduce_mean(tf.abs(target - pred))
        minimize = optimizer.minimize(cost)

    else:
        raise Exception("Unknown implementation " + s.implementation)


    sequence = readDataSet(s.dataSet, s.dataSetDetailed, s)
    if s.limit_to:
        sequence = sequence[:s.limit_to]

    seq_full = sequence['data'].values #use .values to copy

    targetInput = seq_full[s.front_buffer + s.predictionStep - 1:].copy() #grab this now to avoid having to denormalize

    dp = DataProcessor()
    if s.normalization_type == 'default':
        (meanSeq, stdSeq) = dp.normalize('data', sequence, s.nTrain)
    elif s.normalization_type == 'windowed':
        dp.windowed_normalize(sequence)
    elif s.normalization_type == 'AN':
        an = AdaptiveNormalizer(s.lookback, s.lookback + s.predictionStep)
        an.set_pruning(False)
        an.set_source_data(seq_full, s.nTrain)
        an.do_ma('s')
        an.do_stationary()
        an.remove_outliers()
        seq_norm = an.do_adaptive_normalize()
    else:
        raise Exception("Unsupported normalization type: " + s.normalization_type)

    seq_actual = seq_full[s.front_buffer:] #Leave enough headroom for MASE calculation and lookback

    if s.normalization_type != "AN":
        #Default and windowed change the seq itself but still require creating lookback frames
        allX = getX(seq_full, s)
        allY = seq_actual[s.predictionStep-1:]
    else:
        #AN creates a new array but takes care of lookback internally
        allX= seq_norm[:,0:-s.predictionStep]
        allY = np.reshape(seq_norm[:,-1], (-1,))


    predictedInput = np.full((len(allY),), np.nan) #Initialize all predictions to NaN

    trainX = allX[:s.nTrain]
    trainY = allY[:s.nTrain]
    trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    trainY = np.reshape(trainY, (trainY.shape[0], 1))
    if s.implementation == "keras":
        rnn.fit(trainX, trainY, epochs=s.epochs, batch_size=s.batch_size, verbose=min(s.max_verbosity, 2))
    elif s.implementation == "tf":
        sess = tf.Session()
        init = tf.global_variables_initializer()
        sess.run(init)

        for epoch in tqdm(range(s.epochs)):
            sess.run(minimize, feed_dict={data: trainX, target: trainY})
            #print(psutil.Process(os.getpid()).memory_percent())
            # var = [v for v in tf.trainable_variables() if v.name == "rnn/gru_cell/gates/kernel:0"][0]
            # print sess.run(tf.reduce_min(var))
            # print sess.run(tf.reduce_max(var))
            # var = [v for v in tf.trainable_variables() if v.name == "rnn/gru_cell/gates/bias:0"][0]
            # print sess.run(tf.reduce_min(var))
            # print sess.run(tf.reduce_max(var))
            # var = [v for v in tf.trainable_variables() if v.name == "rnn/gru_cell/candidate/kernel:0"][0]
            # print sess.run(tf.reduce_min(var))
            # print sess.run(tf.reduce_max(var))
            # var = [v for v in tf.trainable_variables() if v.name == "rnn/gru_cell/candidate/bias:0"][0]
            # print sess.run(tf.reduce_min(var))
            # print sess.run(tf.reduce_max(var))
            # print "loop"

    latestStart = None
    for i in tqdm(xrange(s.nTrain + s.predictionStep, len(allX)), disable=s.max_verbosity == 0):
        if i % s.retrain_interval == 0 and i > s.numLags+s.nTrain and s.online:
            if s.normalization_type == 'AN':
                predictedInput = np.array(an.do_adaptive_denormalize(predictedInput, therange=(i-s.retrain_interval, i)))
                latestStart = i
                an.set_ignore_first_n(i-s.nTrain-s.predictionStep)
                an.do_ma('s')
                an.do_stationary()
                an.remove_outliers()
                seq_norm = an.do_adaptive_normalize()

                allX = seq_norm[:, 0:-s.predictionStep]
                allY = np.reshape(seq_norm[:, -1], (-1,))

            trainX = allX[i-s.nTrain-s.predictionStep:i-s.predictionStep]
            trainY = allY[i-s.nTrain-s.predictionStep:i-s.predictionStep]
            trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
            trainY = np.reshape(trainY, (trainY.shape[0], 1))
            if s.implementation == "keras":
                rnn.fit(trainX, trainY, epochs=s.epochs, batch_size=s.batch_size, verbose=0)
            elif s.implementation == "tf":
                for epoch in range(s.epochs):
                    sess.run(minimize, feed_dict={data: trainX, target: trainY})
        if s.implementation == "keras":
            predictedInput[i] = rnn.predict(np.reshape(allX[i], (1,1,x_dims)))

        elif s.implementation == "tf":
            predictedInput[i] = sess.run(dense, feed_dict={data: np.reshape(allX[i], (1, 1, x_dims))})

    for i in range(s.nTrain + s.predictionStep):
        predictedInput[i] = np.nan


    if s.normalization_type == 'default':
        predictedInput = dp.denormalize(predictedInput, meanSeq, stdSeq)
    elif s.normalization_type == 'windowed':
        dp.windowed_denormalize(predictedInput, targetInput)
    elif s.normalization_type == 'AN':
        if latestStart:
            predictedInput = np.array(an.do_adaptive_denormalize(predictedInput, therange=(latestStart, len(predictedInput))))
        else:
            predictedInput = np.array(an.do_adaptive_denormalize(predictedInput))

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
    mase = errors.get_mase(predictedInput, targetInput, np.roll(targetInput, s.season))
    if s.max_verbosity > 0:
        print "MASE {}".format(mase)

    if s.implementation == "tf":
        sess.close()
    return mase

if __name__ == "__main__":
    settings = GruSettings()
    settings.finalize()

    run_gru(settings)
