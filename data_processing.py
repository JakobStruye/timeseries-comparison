
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from tqdm import tqdm
import time
from pandas import Series
import csv


class DataProcessor():

    window_size = 4

    def windowed_normalize(self, sequence, columns = None):
        self.scaler_dicts = dict()
        if columns is None:
            columns = range(sequence.shape[1])
        for col in columns:

            length = len(sequence[:,col])
            scalers = dict()
            self.scaler_dicts[col] = scalers

            col_orig = sequence[:,col].copy()

            mean = np.mean(col_orig[:self.window_size]) #new
            std = np.std(col_orig[:self.window_size]) #new
            sequence[0:self.window_size, col] = (sequence[0:self.window_size, col] - mean) / std #new
            for i in range(0, self.window_size):  #new
                scalers[i] = (mean, std) #new
            for i in tqdm(range(self.window_size,length), disable=True):
                #scaler = MinMaxScaler(feature_range=(-1, 1))
                #scaler.fit(seq_2d[i-99:i+1])
                mean = np.mean(col_orig[i-self.window_size+1:i+1]) #new
                std = np.std(col_orig[i-self.window_size+1:i+1]) #new

                #newval = np.reshape(scaler.transform(seq_2d[i:i+1]), 1)
                if i == 9504:
                    print mean, std, sequence[i]
                newval = np.reshape((sequence[i, col] - mean) / std, 1) #new

                sequence[i, col] = newval

                #scalers[i] = scaler
                scalers[i] = (mean, std) #new

    def windowed_denormalize(self, predictedInput, pred_step=5):
        assert(self.scaler_dicts)
        the_range = range(len(predictedInput))
        for i in the_range:
            index = len(self.scaler_dicts[0]) - len(predictedInput) + i - pred_step
            scaler = self.scaler_dicts[0][index]
            #predictedInput[i] = np.nan if np.isnan(predictedInput[i]) else scaler.inverse_transform(np.reshape(predictedInput[i:i + 1], (1, 1)))
            predictedInput[i] = np.nan if np.isnan(predictedInput[i]) else predictedInput[i] * scaler[1] + scaler[0]
            #targetInput[i] = np.nan if np.isnan(targetInput[i]) else  scaler.inverse_transform(np.reshape(targetInput[i:i + 1], (1, 1)))



    def normalize(self, sequence, nTrain, columns=None):
        if columns is None:
            columns = range(sequence.shape[1])
        #Mean and std per column!
        mean = np.mean(sequence[:nTrain,columns], axis=0)
        std = np.std(sequence[:nTrain,columns], axis=0)
        sequence[:,columns] = (sequence[:,columns] - mean)/std

        return (mean,std)

    def normalizeNoTrain(self, sequence, columns=None):
        if columns is None:
            columns = range(sequence.shape[1])
        #Mean and std per column!
        mean = np.mean(sequence[:,columns], axis=0)
        std = np.std(sequence[:,columns], axis=0)
        sequence[:,columns] = (sequence[:,columns] - mean)/std

        return (mean,std)


    def denormalize(self, column, mean, std):
        return (column * std) + mean

    def difference(self, dataset, interval=1):
        newseries = dataset.copy()
        for i in range(interval, len(dataset)):
            #newseries['data'][i] = dataset['data'][i] - dataset['data'][i - interval]
            newseries['data'][i] = dataset['data'][i] - dataset['data'][i - interval] - 0.995 * newseries['data'][i-1]
        return newseries

    # invert differenced value
    def inverse_difference(self, history, yhat, interval=1):
        return yhat + history[interval]


    def saveResultToFile(self, dataSet, predictedInput, targetInput, algorithmName, prediction_step=5, max_verbosity=2):

        outputFileName = './prediction/' + dataSet + '_' + algorithmName + '_pred.csv'
        if max_verbosity > 0:
            print "Saving to " + './prediction/' + dataSet + '_' + algorithmName + '_pred.csv'
        outputFile = open(outputFileName, "w")
        csvWriter = csv.writer(outputFile)
        csvWriter.writerow(
            ['timestamp', 'data', 'prediction-' + str(prediction_step) + 'step'])
        csvWriter.writerow(['datetime', 'float', 'float'])
        csvWriter.writerow(['', '', ''])

        for i in xrange(len(predictedInput)):
            csvWriter.writerow([i, targetInput[i], '%.13f' % predictedInput[i]])

        outputFile.close()