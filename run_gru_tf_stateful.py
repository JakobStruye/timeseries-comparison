import numpy as np
import random as rn

import os
os.environ['PYTHONHASHSEED'] = '0' #not sure if needed

rn.seed(1)
np.random.seed(2)

from scipy import random
random.seed(3)

import tensorflow as tf
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
from keras import backend as K

tf.set_random_seed(4)

sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

os.environ['TF_CPP_MIN_LOG_LEVEL']='2' #Suppress build warnings

from optparse import OptionParser
import pandas as pd
from dataset_settings import *


from datetime import datetime
from tqdm import tqdm
import sys
from data_processing import DataProcessor
import errors
import rnn_tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, GRU, LSTM
#from recurrent import LSTM, GRU
from keras.optimizers import adam
from keras.callbacks import TensorBoard
from adaptive_normalization import AdaptiveNormalizer
import core_binary
import time

np.set_printoptions(suppress=True) #no scientific notation

detailedSets = ["reddit"] #with more than 1 csv


def readDataSet(dataSet, dataSetDetailed, s):
    """
    Reads data set from file.
    Always predicts the FIRST column returned, uses first n columns as X inputs depending on configuration

    :param dataSet: Data set name
    :param dataSetDetailed: Additional info on which set to use
    :param s: The settings
    :return: The dataset in Pandas format
    """
    if dataSet in detailedSets:
        dataSource = "%s/%s" % (dataSet,dataSetDetailed)
    else:
        dataSource = dataSet

    filePath = 'data/'+dataSource+'.csv'

    if dataSet=='nyc_taxi':
        df = pd.read_csv(filePath, header=0, skiprows=[1,2],
                         names=['time', 'data', 'timeofday', 'dayofweek'])
        sequence = df['data']

        timestamps = df['time']

        daysofweek = []
        timesofday = []

        for timestamp in timestamps:
            timestamp = timestamp.split(" ")
            (dayofweek, timeofday) = getDayAndTime(timestamp[0], timestamp[1], s)
            daysofweek.append(dayofweek)
            timesofday.append(timeofday)

        daysofweek = pd.Series(daysofweek, index=df.index)
        timesofday = pd.Series(timesofday, index=df.index)

        seq = pd.DataFrame(np.array(pd.concat([sequence, daysofweek, timesofday], axis=1)),
                           columns=['data', 'dayofweek', 'timeofday'], dtype=np.float64)

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
    col = seq_full[:- s.predictionStep + 1]
    result_col = []
    for i in range(s.front_buffer, len(col)):
        if s.lookback > 1:
            result_col.append(col[i-s.lookback: i])
        else:
            result_col.append(np.reshape(np.array(col[i - 1]), (1,s.x_dims))) #...welp
    return np.array(result_col)


def getDayAndTime(date, time, s):
    type = hour_types[s.dataSet]
    timeofday = None
    dayofweek = datetime.strptime(date, date_formats[s.dataSet]).weekday() if date else None
    time = time.split(":") if time else None
    if type == HourType.TO_MINUTE:
        timeofday = float(int(time[0]) * 60 + (int(time[1]) if len(time) > 1 else 0))
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

def get_gradients(model):
    """Return the gradient of every trainable weight in model

    Parameters
    -----------
    model : a keras model instance

    First, find all tensors which are trainable in the model. Surprisingly,
    `model.trainable_weights` will return tensors for which
    trainable=False has been set on their layer (last time I checked), hence the extra check.
    Next, get the gradients of the loss with respect to the weights.

    """
    weights = [tensor for tensor in model.trainable_weights if model.get_layer(tensor.name[:-2]).trainable]
    optimizer = model.optimizer

    return optimizer.get_gradients(model.total_loss, weights)

class GruSettings:
    epochs = 75

    useTimeOfDay = True
    useDayOfWeek = True

    retrain_interval = 1500

    predictionStep = 5
    batch_size = 17
    online = True
    nodes = 58
    loss = 'mae'


    limit_to = None  # None for no limit
    lookback = 50
    season = 48
    max_verbosity = 2

    lookback_as_features = False  # Considers each lookback value a separate feature if True, ignored if lookback is None
    feature_count = 1  # Uses the first `feature_count` selected columns as features, ignored if `lookback_as_features`

    nTrain = None #chosen automatically if not specified

    stateful=True

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
        if not self.lookback:
            self.batch_size = 1 #One large batch size 1 of many time_steps
        if self.nTrain % self.batch_size != 0:
            if self.max_verbosity > 0:
                print "Adding", self.batch_size - (self.nTrain % self.batch_size), "to nTrain", self.nTrain
            self.nTrain += self.batch_size - (self.nTrain % self.batch_size)
        #self.numLags = 0.25 * self.nTrain #Don't immediately retrain

        if not self.lookback:
            self.lookback_as_features = False
        #The first time at which we can actually predict: need enough headroom for both MASE calculation
        #and filling the lookback window
        self.front_buffer = max(self.season - self.predictionStep, self.lookback)
        if self.ignore_for_error: #offset for values not predicted
            self.ignore_for_error-= (self.front_buffer + self.predictionStep - 1)
        self.x_dims = self.lookback if self.lookback_as_features else self.feature_count
        if self.lookback_as_features:
            self.feature_count = 1
        self.time_steps_train = self.nTrain if not self.lookback else self.lookback if not self.lookback_as_features else 1
        self.return_sequences = not self.lookback
        self.train_samples_count = 1 if not self.lookback else self.nTrain
        self.actual_input_shape_train = (self.train_samples_count, -1, self.x_dims)
        self.actual_output_shape_train = (1, -1,1) if self.return_sequences else (-1,1)
        self.rnn_input_shape = (None, self.x_dims)
        self.batch_output_shape = (self.batch_size, self.x_dims)
        self.predict_input_shape = (1, self.lookback if not self.lookback_as_features else 1, self.x_dims if not self.lookback_as_features else self.lookback)
        self.rnn_batch_size = 1 if not self.lookback else None #Must be specified for stateful

    def print_settings(self):
        print "RNN type:", self.rnn_type
        print "nTrain:", self.nTrain
        print "Prediction step:", self.predictionStep
        print "Epochs:", self.epochs
        print "Batch size:", self.batch_size
        print "Is online:", self.online
        print "Retrain:", self.retrain_interval
        print "Nodes:", self.nodes
        print "Lookback:", self.lookback
        print "Lookback as features?:", self.lookback_as_features
        print "Features:", self.feature_count
        print "Stateful:", self.stateful
        print "Loss:", self.loss
        print "Normalization:", self.normalization_type
        print "Limit to:", self.limit_to
        print "Season:", self.season
        print "Ignore for error:", self.ignore_for_error
        print "Implmentation:", self.implementation
        print "Binary?:", self.use_binary
        print "Dataset:", self.dataSet
        print "Dataset detailed:", self.dataSetDetailed

def run_gru(s):
    s.print_settings()
    prob = tf.placeholder_with_default(1.0, shape=()) #Retain probability for TF dropout




    if s.implementation == "keras":
        if s.use_binary:
            raise Exception("Binary Keras not implemented")
        rnn = Sequential()
        if s.rnn_type == "lstm":
            rnn_layer = LSTM(s.nodes, input_shape=s.rnn_input_shape, batch_size=s.rnn_batch_size, kernel_initializer='he_uniform', stateful=s.stateful, return_sequences=s.return_sequences)
            rnn.add(rnn_layer)
        elif s.rnn_type == "gru":
            rnn_layer = GRU(s.nodes, input_shape=s.rnn_input_shape, batch_size=s.rnn_batch_size, kernel_initializer='he_uniform', stateful=s.stateful, return_sequences=s.return_sequences)
            rnn.add(rnn_layer)

        rnn.add(Dropout(0.5))
        rnn.add(Dense(1, kernel_initializer='he_uniform'))
        opt = adam(lr=s.lr, decay=0.0, epsilon=0.001)#, clipvalue=1.)#1e-3)
        rnn.compile(loss=s.loss, optimizer=opt)
        print(rnn.summary())
    elif s.implementation == "tf":
        data = tf.placeholder(tf.float32, (s.batch_size,) + s.rnn_input_shape)  # Number of examples, number of input, dimension of each input
        target = tf.placeholder(tf.float32, (1, 3000, 1)) #TODO s.batch_output_shape)
        print data, target
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

        if s.loss=='mse':
            cost = tf.losses.mean_squared_error(target, pred)
        elif s.loss=='mae':
            cost = tf.reduce_mean(tf.abs(target - pred))
        else:
            raise Exception("Unrecognized loss type " + s.loss)
        minimize = optimizer.minimize(cost)

    else:
        raise Exception("Unknown implementation " + s.implementation)


    sequence = readDataSet(s.dataSet, s.dataSetDetailed, s).values

    if s.limit_to:
        sequence = sequence[:s.limit_to]

    #Get rid of unneeded columns
    sequence = sequence[:,0:s.feature_count]
    #sequence[-1000,0] = 666
    #print "Changed -1000 to 666"


    """
    We need to leave some values unpredicted in front so that
      - We can fill the lookback window for each prediction
      - We can get the value from 1 season earlier for MASE
    --> Don't use the first `front_buffer` values as prediction
    --> Independent from `prediction_step`, so the first actual value predicted is `front_buffer`\
        plus however many steps the `prediction_step` is higher than 1
        In other words, the most recent X-value for the first prediction will be the final value in the `front_buffer`
    """

    first_prediction_index = s.front_buffer + s.predictionStep - 1


    targetInput = sequence[first_prediction_index:, 0].copy() #grab this now to avoid having to denormalize


    dp = DataProcessor()
    if s.normalization_type == 'default':
        (meanSeq, stdSeq) = dp.normalize(sequence, s.nTrain)

    elif s.normalization_type == 'windowed':
        dp.windowed_normalize(sequence, columns=[0])
        if s.feature_count > 1:
            dp.normalize(sequence, s.nTrain, columns=range(1,s.feature_count))
    elif s.normalization_type == 'AN':
        an = AdaptiveNormalizer(s.lookback, s.lookback + s.predictionStep)
        an.set_pruning(False)
        an.set_source_data(sequence, s.nTrain)
        an.do_ma('s')
        an.do_stationary()
        an.remove_outliers()
        seq_norm = an.do_adaptive_normalize()
        if s.feature_count > 1:
            dp.normalize(sequence, s.nTrain, columns=range(1,s.feature_count))
            start = sequence.shape[0] - seq_norm.shape[0] - s.lookback - s.predictionStep +  1
            for i in range(seq_norm.shape[0]):
                seq_norm[i,:,1:s.feature_count] = sequence[start+i:start+i+seq_norm.shape[1], 1:s.feature_count]

    else:
        raise Exception("Unsupported normalization type: " + s.normalization_type)

    #seq_actual = sequence[s.front_buffer:] #Leave enough headroom for MASE calculation and lookback
    #seq_full_norm = np.reshape(sequence[:,0], (sequence.shape[0],))
    #seq_actual_norm = seq_full_norm[s.front_buffer:]
    if s.normalization_type != "AN":
        #Default and windowed change the seq itself but still require creating lookback frames
        allX = getX(sequence, s)
        allY = sequence[first_prediction_index:,0]
    else:
        #AN creates a new array but takes care of lookback internally
        allX= seq_norm[:,0:-s.predictionStep]
        allY = np.reshape(seq_norm[:,-1,0], (-1,))
    predictedInput = np.full((len(allY),), np.nan) #Initialize all predictions to NaN
    print tf.Session().run(tf.reduce_sum(tf.cast(tf.is_finite(sequence[:]), tf.float32)))


    trainX = allX[:s.nTrain]
    trainY = allY[:s.nTrain]
    trainX = np.reshape(trainX, s.actual_input_shape_train)
    trainY = np.reshape(trainY, s.actual_output_shape_train)

    if s.implementation == "keras":
        #for _ in tqdm(range(s.epochs)):
        for _ in tqdm(range(1)):
            rnn.fit(trainX, trainY, epochs=s.epochs, batch_size=s.batch_size, verbose=min(s.max_verbosity, 2), shuffle=not s.stateful)#, validation_data=(trainX, trainY), callbacks=[TensorBoard(log_dir='./logs', histogram_freq=1, write_grads=True)])
            if s.stateful:
                rnn_layer.reset_states()

    elif s.implementation == "tf":
        sess = tf.Session()
        writer = tf.summary.FileWriter("results/", graph=sess.graph)
        init = tf.global_variables_initializer()
        sess.run(init)

        for epoch in tqdm(range(s.epochs)):
            the_cost, _, summ = sess.run([cost, minimize, summary], feed_dict={data: trainX, target: trainY, prob: 0.5})
            writer.add_summary(summ, epoch)
            if epoch % 100 == 0:
                print the_cost

        var = [v for v in tf.trainable_variables() if v.name == "dense/bias:0"]
        print sess.run(var)

    # weights = rnn.trainable_weights  # weight tensors
    # for layer in rnn.layers:
    #     print layer.name
    # weights = [weight for weight in weights] # if rnn.get_layer(
    # #weight.name[:-2].split('/')).trainable]  # filter down weights tensors to only ones which are trainable
    # gradients = rnn.optimizer.get_gradients(rnn.model.total_loss, weights)
    # print gradients
    #
    # input_tensors = [rnn.inputs[0],  # input data
    #                  rnn.sample_weights[0],  # how much to weight each sample by
    #                  rnn.targets[0],  # labels
    #                  K.learning_phase(),  # train or test mode
    #                  ]
    #
    # get_gradients = K.function(inputs=input_tensors, outputs=gradients)
    #
    # inputs = [trainX,  # X
    #           np.ones(len(trainX)),  # sample weights
    #           trainY,  # y
    #           0  # learning phase in TEST mode
    #           ]
    # grads = get_gradients(inputs)
    # print grads
    # for grad in grads:
    #     print "NAN?", np.any(np.isnan(grad))
    # #exit(1)
    # rnn.save_weights("weights.out")


    
    # for layer in rnn.layers:
    #     print layer.get_weights()
    #for i in xrange(0, s.nTrain + s.predictionStep):
    #   rnn.predict(np.reshape(allX[i], (1, 1, x_dims)))
    #predictedInput[s.nTrain + s.predictionStep : len(allX)] =  rnn.predict(np.reshape(allX[s.nTrain + s.predictionStep : len(allX)], (1, 12510, x_dims)))
    latestStart = None
    do_non_lookback = True
    print "FROM", s.nTrain + s.predictionStep, len(allX)
    for i in tqdm(xrange(s.nTrain + s.predictionStep, len(allX)), disable=s.max_verbosity == 0):
        if i % s.retrain_interval == 0 and s.online:
            do_non_lookback = True
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

            if s.lookback:
                trainX = allX[i-s.nTrain-s.predictionStep:i-s.predictionStep]
                trainY = allY[i-s.nTrain-s.predictionStep:i-s.predictionStep]
            else:
                trainX = allX[:i-s.predictionStep]
                trainY = allY[:i-s.predictionStep]

            trainX = np.reshape(trainX, s.actual_input_shape_train)
            trainY = np.reshape(trainY, s.actual_output_shape_train)
            if s.implementation == "keras":
                for _ in tqdm(range(s.epochs)):
                    rnn.fit(trainX, trainY, epochs=1, batch_size=s.batch_size, verbose=0,
                            shuffle=not s.stateful)
                    if s.stateful:
                        rnn_layer.reset_states()
            elif s.implementation == "tf":
                for epoch in range(s.epochs):
                    sess.run(minimize, feed_dict={data: trainX, target: trainY, prob: 0.5})

        if s.lookback:
            if s.implementation == "keras":
                predictedInput[i] = rnn.predict(np.reshape(allX[i], s.predict_input_shape))

            elif s.implementation == "tf":
                predictedInput[i] = sess.run(dense, feed_dict={data: np.reshape(allX[i], s.predict_input_shape)})
        elif do_non_lookback:
            do_non_lookback = False
            up_to = min(allX.shape[0], i - (i % s.retrain_interval) + s.retrain_interval) if s.online else allX.shape[0]
            start_time = time.time()
            #print allX[0]
            new_predicts = rnn.predict(np.reshape(allX[:up_to], (1, up_to, s.x_dims)))
            new_predicts = np.reshape(new_predicts, (new_predicts.shape[1],))
            #print new_predicts[0]
            predictedInput[i:up_to] = new_predicts[i:up_to]




    for i in range(s.nTrain + s.predictionStep):
        predictedInput[i] = np.nan


    if s.normalization_type == 'default':
        predictedInput = dp.denormalize(predictedInput, meanSeq[0], stdSeq[0])
    elif s.normalization_type == 'windowed':
        dp.windowed_denormalize(predictedInput,  pred_step=s.predictionStep)
    elif s.normalization_type == 'AN':
        if latestStart:
            predictedInput = np.array(an.do_adaptive_denormalize(predictedInput, therange=(latestStart, len(predictedInput))))
        else:
            predictedInput = np.array(an.do_adaptive_denormalize(predictedInput))
        if an.pruning:
            targetInput = np.delete(targetInput, an.deletes)



    #print "Last not to change:", predictedInput[-996], targetInput[-996]
    #print "First to change:", predictedInput[-995], targetInput[-995]
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
