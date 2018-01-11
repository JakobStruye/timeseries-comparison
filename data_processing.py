from sklearn.preprocessing import MinMaxScaler
import numpy as np
from tqdm import tqdm
import time
from pandas import Series
import csv

class DataProcessor():



    def windowed_normalize(self, sequence, field_name='data', is_data_field=False, length=None, print_progress=False):
        if length is None:
            length = len(sequence[field_name])
        self.scalers = dict()
        scaler = MinMaxScaler(feature_range=(-1, 1))
        seq_2d = np.copy(sequence[field_name].values.reshape(len(sequence[field_name]), 1))
        scaler.fit(seq_2d[0:100])
        for i in range(0,100):
            self.scalers[i] = scaler
        newfield = []

        if is_data_field:
            for i in range(0,100):
                newfield.append(np.reshape(scaler.transform(seq_2d[i:i+1]), 1))
        else:
            sequence[field_name][0:100] = np.reshape(scaler.transform(seq_2d[0:100]), 100)
        for i in tqdm(range(100,length), disable=True):
            scaler = MinMaxScaler(feature_range=(-1, 1))
            scaler.fit(seq_2d[i-99:i+1])
            newval = np.reshape(scaler.transform(seq_2d[i:i+1]), 1)
            if is_data_field:
                newfield.append(newval)
                pass
            else:
                sequence[field_name][i] = newval

            self.scalers[i] = scaler
        if is_data_field:
            new_column = Series(newfield, name=field_name, index=sequence.index)
            sequence.update(new_column)

    def windowed_denormalize(self, predictedInput, targetInput, only_i=None):
        assert(self.scalers)
        if only_i is None:
            the_range = range(len(predictedInput))
        else:
            the_range = [only_i]
        for i in the_range:
            scaler = self.scalers[i]
            predictedInput[i] = scaler.inverse_transform(np.reshape(predictedInput[i:i + 1], (1, 1)))
            targetInput[i] = scaler.inverse_transform(np.reshape(targetInput[i:i + 1], (1, 1)))


    def normalize(self, column, sequence):
        mean = np.mean(sequence[column])
        std = np.std(sequence[column])
        sequence[column] = (sequence[column] - mean)/std
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