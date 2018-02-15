
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
            #sequence[field_name][0:100] = np.reshape(scaler.transform(seq_2d[0:100]), 100)
            mean = np.mean(sequence[field_name][:100]) #new
            std = np.std(sequence[field_name][:100]) #new
            sequence[field_name][0:100] = (sequence[field_name][0:100] - mean) / std #new
            for i in range(0, 100):  #new
                self.scalers[i] = (mean, std) #new
        for i in tqdm(range(100,length), disable=True):
            scaler = MinMaxScaler(feature_range=(-1, 1))
            scaler.fit(seq_2d[i-99:i+1])
            mean = np.mean(sequence[field_name][i-99:i+1]) #new
            std = np.std(sequence[field_name][i-99:i+1]) #new
            newval = np.reshape(scaler.transform(seq_2d[i:i+1]), 1)
            newval = np.reshape((sequence[field_name][i] - mean) / std, 1) #new

            if is_data_field:
                newfield.append(newval)
                pass
            else:
                sequence[field_name][i] = newval

            self.scalers[i] = scaler
            self.scalers[i] = (mean, std) #new
        if is_data_field:
            new_column = Series(newfield, name=field_name, index=sequence.index)
            sequence.update(new_column)
        print "NORMD", sequence[field_name][10000], len(self.scalers)

    def windowed_denormalize(self, predictedInput, targetInput, only_i=None, pred_step=5):
        assert(self.scalers)
        print "DENORMD", predictedInput[10000], targetInput[10000]

        if only_i is None:
            the_range = range(len(predictedInput))
        else:
            the_range = [only_i]
        for i in the_range:
            index = len(self.scalers) - len(predictedInput) + i - pred_step
            scaler = self.scalers[index]
            #predictedInput[i] = np.nan if np.isnan(predictedInput[i]) else scaler.inverse_transform(np.reshape(predictedInput[i:i + 1], (1, 1)))
            predictedInput[i] = np.nan if np.isnan(predictedInput[i]) else predictedInput[i] * scaler[1] + scaler[0]
            #targetInput[i] = np.nan if np.isnan(targetInput[i]) else  scaler.inverse_transform(np.reshape(targetInput[i:i + 1], (1, 1)))



    def frame_normalize(self, column, sequence, start, mid, end):
        print "Normalize from", start, "to", end
        mean = np.mean(sequence[column][start:mid])
        std = np.std(sequence[column][start:mid])
        sequence[column][start:end] = (sequence[column][start:end] - mean)/std
        return (mean,std)

    def frame_denormalize(self, column, sequence, mean, std, start, end):
        start = max(0, start)
        print "Denormalize from", start, "to", end
        sequence[column][start:end] = (sequence[column][start:end] * std) + mean

    def frame_denormalize_col(self, column, mean, std, start, end):
        print "Col denormalize from", start, "to", end
        column[start:end] = (column[start:end] * std) + mean

    def normalize(self, column, sequence):
        mean = np.mean(sequence[column])
        std = np.std(sequence[column])
        sequence[column] = (sequence[column] - mean)/std
        return (mean,std)


    def normalize(self, column, sequence, nTrain):
        mean = np.mean(sequence[column][:nTrain])
        std = np.std(sequence[column][:nTrain])
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



    #ADAPTIVE NORMALIZATION
