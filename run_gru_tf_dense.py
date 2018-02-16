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
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Dropout, GRU, Input, Multiply, Activation, Reshape
from keras.optimizers import adam, rmsprop
from adaptive_normalization import AdaptiveNormalizer
import core_binary


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

        seq = pd.DataFrame(np.array(pd.concat([sequence], axis=1)),
                           columns=['data'], dtype=np.float64)

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
        seq = pd.DataFrame(np.array(pd.concat([sequence], axis=1)),
                           columns=['data'])

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
    print s.front_buffer, "FBUF"
    result_col = []
    counter = 0
    for i in range(s.front_buffer, len(col)):
        counter += 1
        if s.lookback > 1:
            result_col.append(col[i-s.lookback: i])
        else:
            result_col.append(np.reshape(np.array(col[i - 1]), (1,1))) #...welp
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

    nTrain = None #chosen automatically if not specified

    ignore_for_error = None

    normalization_type = 'default' #'default', 'windowed' or 'AN' (adaptive normalization
    implementation = "tf" #"tf" or "keras"
    rnn_type = "lstm" #"lstm" or "gru"
    use_binary = False

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
        if not self.nTrain:
            self.nTrain = max(self.retrain_interval * 2, self.season * 3)
            if self.max_verbosity > 0:
                print "Automatically set nTrain to", self.nTrain
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
        if self.ignore_for_error: #offset for values not predicted
            self.ignore_for_error-= (self.front_buffer + self.predictionStep - 1)


def run_gru(s):
    prob = tf.placeholder_with_default(1.0, shape=())

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
        if s.use_binary:
            raise Exception("Binary Keras not implemented")
        rnn = Sequential()
        if s.rnn_type == "lstm":
            rnn.add(LSTM(s.nodes, input_shape=(None,x_dims), kernel_initializer='he_uniform'))
        elif s.rnn_type == "gru":
            rnn.add(GRU(s.nodes, input_shape=(None, x_dims), kernel_initializer='he_uniform'))

        rnn.add(Dropout(0.5))
        rnn.add(Dense(1, kernel_initializer='he_uniform'))
        opt = rmsprop(lr=s.lr)#1e-3)
        rnn.compile(loss='mae', optimizer=opt)

        input = Input(shape=(1, x_dims))
        dense1 = Dense(s.nodes, activation='sigmoid')(input)
        dense2 = Dense(s.nodes, activation='sigmoid')(input)
        dense3 = Dense(s.nodes, activation='tanh')(input)
        mult1 = Multiply()([dense2, dense3])
        act1 = Activation('tanh')(mult1)
        mult2 = Multiply()([dense1, act1])
        reshape = Reshape((s.nodes,))(mult2)
        dropout = Dropout(0.5)(reshape)
        dense_out = Dense(1)(dropout)
        rnn = Model(inputs=[input], outputs=[dense_out])
        opt = adam(lr=s.lr)  # 1e-3)
        rnn.compile(loss='mae', optimizer=opt)
        print rnn.summary()


    elif s.implementation == "tf":
        data = tf.placeholder(tf.float32, [None, s.lookback,  1])  # Number of examples, number of input, dimension of each input
        target = tf.placeholder(tf.float32, [None, 1])
        if s.rnn_type == "lstm" and s.use_binary:
            cell = rnn_tf.LSTMCell(s.nodes)

        elif s.rnn_type == "lstm" and not s.use_binary:
            cell = tf.nn.rnn_cell.LSTMCell(s.nodes)
        elif s.rnn_type == "gru" and s.use_binary:
            cell = rnn_tf.GRUCell(s.nodes)
        elif s.rnn_type == "gru" and not s.use_binary:
            cell = tf.nn.rnn_cell.GRUCell(s.nodes)


        val, _ = tf.nn.dynamic_rnn(cell, data, dtype=tf.float32)
        with tf.name_scope('rnn_summaries'):
            var = val
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('histogram', var)
        val = tf.nn.dropout(val, prob)
        if not s.use_binary:
            dense = tf.layers.dense(val, 1)
        else:
            dense = core_binary.dense(val, 1)
        with tf.name_scope('dense_summaries'):
            var = dense
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('histogram', var)
        pred = tf.reshape(dense, (tf.shape(dense)[0], 1))
        summary = tf.summary.merge_all()
        optimizer = tf.train.AdamOptimizer(learning_rate=s.lr)
        #cost = tf.losses.mean_squared_error(target, pred)
        cost = tf.reduce_mean(tf.abs(target - pred))
        minimize = optimizer.minimize(cost)

    else:
        raise Exception("Unknown implementation " + s.implementation)


    sequence = readDataSet(s.dataSet, s.dataSetDetailed, s)
    if s.limit_to:
        sequence = sequence[:s.limit_to]

    #TEMP SANITY CHECK
    # sequence['data'][7001] = 0
    # sequence['data'][7002] = 0
    # sequence['data'][7003] = 0
    # sequence['data'][7004] = 0
    # sequence['data'][7005] = 0
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

    seq_full_norm = sequence['data'].values
    seq_actual_norm = seq_full_norm[s.front_buffer:]

    if s.normalization_type != "AN":
        #Default and windowed change the seq itself but still require creating lookback frames
        allX = getX(seq_full_norm, s)
        allY = seq_actual_norm[s.predictionStep-1:]
    else:
        #AN creates a new array but takes care of lookback internally
        allX= seq_norm[:,0:-s.predictionStep]
        allY = np.reshape(seq_norm[:,-1], (-1,))
        # TODO FIX PROPERLY (now rolled too far)
        too_long = len(allX) - (len(seq_full) - s.front_buffer - s.predictionStep + 1)
        if too_long > 0:
            allX = allX[too_long:]
            allY = allY[too_long:]

    print len(allX), len(allY), s.front_buffer
    predictedInput = np.full((len(allY),), np.nan) #Initialize all predictions to NaN

    trainX = allX[:s.nTrain]
    trainY = allY[:s.nTrain]
    trainX = np.reshape(trainX, (trainX.shape[0],1,  trainX.shape[1]))
    trainY = np.reshape(trainY, ( trainY.shape[0],))
    if s.implementation == "keras":
        rnn.fit(trainX, trainY, epochs=s.epochs, batch_size=s.batch_size, verbose=min(s.max_verbosity, 2))
    elif s.implementation == "tf":
        sess = tf.Session()
        writer = tf.summary.FileWriter("results/", graph=sess.graph)
        init = tf.global_variables_initializer()
        sess.run(init)

        for v in tf.trainable_variables():
            print v.name
        for epoch in tqdm(range(s.epochs)):
            the_cost, _, summ = sess.run([cost, minimize, summary], feed_dict={data: trainX, target: trainY, prob: 0.5})
            writer.add_summary(summ, epoch)
            if epoch % 10 == 0:
                print the_cost
            #print(psutil.Process(os.getpid()).memory_percent())
            var = [v for v in tf.trainable_variables() if v.name == "rnn/gru_cell/gates/kernel:0"][0]
            print sess.run(tf.reduce_min(var))
            print sess.run(tf.reduce_max(var))
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
        var = [v for v in tf.trainable_variables() if v.name == "dense/bias:0"]
        print sess.run(var)

    minval = 10
    latestStart = None
    for i in tqdm(xrange(s.nTrain + s.predictionStep, len(allX)), disable=s.max_verbosity == 0):
    #for i in tqdm(xrange(0, len(allX)), disable=s.max_verbosity == 0):
    #for i in tqdm(xrange(10475, len(allX)), disable=s.max_verbosity == 0):
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
                    sess.run(minimize, feed_dict={data: trainX, target: trainY, prob: 0.5})


        if s.implementation == "keras":
            predictedInput[i] = rnn.predict(np.reshape(allX[i], (1,1,x_dims)))

        elif s.implementation == "tf":
            predictedInput[i] = sess.run(dense, feed_dict={data: np.reshape(allX[i], (1, x_dims, 1))})
            #if len(allX) > i+5:
            #    predictedInput[i] = allY[i-3000]

        # if i == 10000:
        #     print allX[i]
        #     print "should be ", (targetInput[i] - meanSeq) / stdSeq
        #     print "predicted as ", predictedInput[i]

    # for i in range(s.nTrain + s.predictionStep):
    #     predictedInput[i] = np.nan
    print "SMALLEST", minval
    # np.set_printoptions(threshold=np.nan, suppress=True)
    # print "ALLY START"
    # for val in allY:
    #     print val
    # print "ALLY STOP"

    if s.normalization_type == 'default':
        predictedInput = dp.denormalize(predictedInput, meanSeq, stdSeq)
        #targetInput = dp.denormalize(targetInput, meanSeq, stdSeq)
    elif s.normalization_type == 'windowed':
        dp.windowed_denormalize(predictedInput, targetInput,  pred_step=s.predictionStep)
    elif s.normalization_type == 'AN':
        if latestStart:
            predictedInput = np.array(an.do_adaptive_denormalize(predictedInput, therange=(latestStart, len(predictedInput))))
        else:
            predictedInput = np.array(an.do_adaptive_denormalize(predictedInput))
        if an.pruning:
            targetInput = np.delete(targetInput, an.deletes)
    print len(predictedInput), len(targetInput), "LENS"
    #TEMP SANITY CHECK
    #print predictedInput[7005 - s.front_buffer - s.predictionStep +1]
    #print predictedInput[7006 - s.front_buffer - s.predictionStep + 1]
    dp.saveResultToFile(s.dataSet, predictedInput, targetInput, 'gru', s.predictionStep, s.max_verbosity)
    skipTrain = s.ignore_for_error
    from plot import computeSquareDeviation
    squareDeviation = computeSquareDeviation(predictedInput, targetInput)
    squareDeviation[:skipTrain] = None
    nrmse = np.sqrt(np.nanmean(squareDeviation)) / np.nanstd(targetInput)
    if s.max_verbosity > 0:
        print "", s.nodes, "NRMSE {}".format(nrmse)
    mae = np.nanmean(np.abs(targetInput-predictedInput))
    if s.max_verbosity > 0:
        print "MAE {}".format(mae)
    mape = errors.get_mape(predictedInput,targetInput, s.ignore_for_error)
    if s.max_verbosity > 0:
            print "MAPE {}".format(mape)
    mase = errors.get_mase(predictedInput, targetInput, np.roll(targetInput, s.season), s.ignore_for_error)
    if s.max_verbosity > 0:
        print "MASE {}".format(mase)

    if s.implementation == "tf":
        sess.close()
    return mase

if __name__ == "__main__":
    settings = GruSettings()
    settings.finalize()

    run_gru(settings)
