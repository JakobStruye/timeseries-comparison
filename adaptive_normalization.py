from __future__ import division
import math
import numpy as np

class AdaptiveNormalizer:
    """
    Normalizes a source data set of length n into an array of Disjoint Sliding Windows (DSWs)
    First calculates the Moving Average (MA; simple or exponential),
    where value i of the MA depends on values [i:i+k] of the source data.
    Then creates (n*w+1) DSWs of each size w where the j'th value of the i'th DSW is data[i+j]/MA[i]
    The case where k > w - 1 is implemented but untested as it is not needed with LSTM/GRU.

    Finally the DSWs are normalized to [-1,1] where the min and max values are only chosen from a subset of
    the DSWs (likely the "train set").

    Note that when setting k=lookback and w=lookback+predictionStep,
    a DSW contains the lookback array, some unneeded values (if predictionStep > 1) and then the prediction.
    In this case the lookback array values are completely independent from the prediction (and the "unneeded" values)
    meaning they could be computed in a live situation when the prediction is still in the future.
    """

    def __init__(self, k, w):
        """

        :param k: Window for the moving average.
        :param w: Width of one DSW. DSW i's values depend on
        """
        self.k = k
        self.w = w
        print "AN", self.k, self.w
        self.multiplier = 1.5

        self.pruning = True

        self.ignore_first_n = 0 #excludes first n DSWs from the training set


    def set_source_data(self, data, test_size):
        self.data = data
        self.test_size = test_size
        self.data_size = len(data)

        self.configure()

    def set_k(self, k):
        self.k = k
        self.alpha = 2.0 / (k + 1.0) #can be overridden
        self.configure()

    def configure(self):
        self.ma_size = len(self.data) - self.k + 1
        if self.k > self.w - 1:
            self.data_limited = self.data[self.k-(self.w-1):]
            self.data_size_limited = len(self.data_limited)
        else:
            self.data_limited = self.data
            self.data_size_limited = self.data_size

        self.dsw_count = self.data_size_limited - self.w + 1
        self.dsw_train_count = self.dsw_count - self.test_size

        self.ma = None
        self.r = None
        self.r_pruned = None

        assert (len(self.data) >= self.k >= 1)

    def set_alpha(self, alpha):
        self.alpha = alpha

    def set_multiplier(self, multiplier):
        self.multiplier = multiplier

    def set_pruning(self, on=True):
        self.pruning = on

    def set_ignore_first_n(self, n):
        self.ignore_first_n = n
        self.configure()

    def get_simple_moving_avg(self):
        #MA[i] = avg(data[i:i+k])
        new_data = [np.mean(self.data[i:i+self.k], axis=0) for i in range(self.ma_size)]
        return new_data

    def get_exponential_moving_avg(self):
        #MA[0] = avg[data[0:k], MA[i] = 1-alpha * MA[i-1] + alpha * data[i+k-1]
        new_data = [np.mean(self.data[0:self.k], axis=0)]
        for i in range(1, self.ma_size):
            new_data.append((1 - self.alpha) * new_data[-1] + self.alpha * self.data[i+self.k-1])
        return new_data

    def get_stationary(self):
        assert(self.ma) #must be set by now
        print "MA DIM", np.array(self.ma[0]).shape

        r = [self.data_limited[int(math.ceil(i / self.w) + (i - 1) % self.w) - 1] / self.ma[int(math.ceil(i / self.w)) - 1] for i in
             range(1 , self.dsw_count * self.w + 1)]

        r = np.array(r)
        r = np.reshape(r, (self.dsw_count, self.w, r.shape[1]))
        return r

    def get_level_of_adjustment_row(self, ma, row_index):
        return sum([(self.data[j] - ma[row_index]) ** 2 for j in range(row_index,row_index+self.w)]) / self.w

    def get_level_of_adjustment(self, ma):
        return np.mean([self.get_level_of_adjustment_row(ma, i) for i in range(self.dsw_train_count)])

    def get_best_stationary(self):
        """
        Finds best values for k and whether to use simple or exponantial MA. Untested as not needed with GRU/LSTM (fixed k)
        :return:
        """
        best_loa = float('+inf')
        best_k = None
        best_alg = None
        for k in range(1, self.w): #Todo k > self.w - 1
            self.set_k(k)
            sma = self.get_simple_moving_avg()
            ema = self.get_exponential_moving_avg()
            loa_simple = self.get_level_of_adjustment(sma)
            loa_exp = self.get_level_of_adjustment(ema)
            print "Simple LOA:", loa_simple, "Exp LOA:", loa_exp
            if loa_simple < best_loa or loa_exp < best_loa:
                best_loa = min(loa_simple, loa_exp)
                best_k = k
                best_alg = 's' if loa_simple <= loa_exp else 'e'

        print "Best alg:", best_alg, "for k", best_k
        self.set_k(best_k)

        self.ma = self.get_simple_moving_avg() if loa_simple <= loa_exp else self.get_exponential_moving_avg()

        self.do_stationary()

    def do_ma(self, type):
        """
        Calculates and remembers the MA for the given type
        :param type: The type
        :return: None
        """
        assert type == 's' or type == 'e'
        self.ma = self.get_simple_moving_avg() if type == 's' else self.get_exponential_moving_avg()

    def do_stationary(self):
        """
        Calculates and remembers the DSWs.
        :return: None
        """
        self.r = self.get_stationary()
        self.r_ignore = self.r[:self.ignore_first_n]
        self.r_train = self.r[self.ignore_first_n:self.dsw_train_count]
        self.r_test = self.r[self.dsw_train_count:]


    def get_thresholds(self, mult):
        """
        Gets the pruning/normalizing thresholds based on the training set DSWs
        :return: The thresholds
        """
        r_flat = np.reshape(self.r_train, (np.prod(self.r_train.shape),))
        q1 = np.percentile(r_flat, 25)
        q3 = np.percentile(r_flat, 75)
        iqr = q3-q1
        thresh_bottom = q1 - mult*iqr
        thresh_top = q3 + mult * iqr
        return (thresh_bottom, thresh_top)


    def remove_outliers(self):
        """
        Removes any outliers. Currently untested as unused
        :return:
        """
        if self.pruning:
            (thresh_bottom, thresh_top) = self.get_thresholds(self.multiplier * 2.0)
            #todo ignore n first
            self.r_pruned = np.array([self.r_train[i] if np.min(self.r_train[i]) >= thresh_bottom and np.max(self.r_train[i]) <= thresh_top else np.full([self.w], np.nan) for i in range(self.r_train.shape[0]) ])
            self.deletes = []
            for i in range(self.r_pruned.shape[0]) :
                if np.isnan(self.r_pruned[i][0]):
                    self.deletes.append(i)
            print self.deletes
            self.r_pruned = np.delete(self.r_pruned, self.deletes, 0)
            self.ma = np.delete(self.ma, self.deletes, 0)
            self.dsw_count -= len(self.deletes)


        else:
            self.r_pruned = np.vstack((self.r_ignore, self.r_train))


    def do_adaptive_normalize(self):
        """
        Normalizes all DSWs, with only the training set guaranteed to be in [-1,1]
        :return:
        """
        (thresh_bottom, thresh_top) = self.get_thresholds(self.multiplier)
        self.min_r = max(thresh_bottom, np.min(self.r_train))
        self.max_r = min(thresh_top, np.max(self.r_train))
        print "DOMEAN", self.r_train.shape
        self.mean = np.mean(self.r_train, axis=(0,1))
        self.std = np.std(self.r_train, axis=(0,1))
        print self.mean.shape, self.std.shape, "means"

        def do_norm(val):
            #return 2 * ((val - self.min_r) / (self.max_r - self.min_r)) - 1
            return (val - self.mean) / self.std
        normalized = do_norm(np.vstack((self.r_pruned, self.r_test)))
        print normalized

        return normalized

    def do_adaptive_denormalize(self, r_norm, offset=0, therange=None):
        """
        Denormalizes an array based on the latest normalization used.

        :param r_norm: The array to denormalize
        :param offset: Optional (default 0). Shifts the MA used to denormalize right.
        :param therange: Optional (default None). Only denormalizes the range (start,stop) if set
        :return: Copy of the provided array with all requested values denormalized
        """
        denorm = r_norm.copy()
        print len(self.ma), therange, r_norm.shape
        print self.mean.shape, self.std.shape, self.ma[10].shape, denorm.shape, "DENORMMM"
        for i in range(0 if not therange else therange[0], r_norm.shape[0] if not therange else therange[1]):

            #val = ((r_norm[i] + 1.0) / 2.) * (self.max_r - self.min_r) + self.min_r
            val = (r_norm[i] * self.std[0]) + self.mean[0]
            denorm[i] = val * self.ma[i+offset][0]


        return denorm


if __name__ == '__main__':
    a = np.array([1,5,4,4,7,6,2,3,1,0,5,9])
    #a = np.array([1.734, 1.720, 1.707, 1.708, 1.735, 1.746, 1.744, 1.759, 1.751, 1.749, 1.763, 1.753, 1.774])
    # print dp.get_simple_moving_avg(a, 5)
    # ma = dp.get_exponential_moving_avg(a, 5)
    # r = dp.get_stationary(a, 5, 6, 'e')
    # print dp.get_level_of_adjustment_row(a, ma, 0, 6)
    # print dp.get_level_of_adjustment(a, ma, (13 - 6 + 1), 6)
    an = AdaptiveNormalizer(4,5)
    an.set_source_data(a, 1)
    #an.set_denominator_offset(2)

    an.get_best_stationary()
    print an.r
    #an.remove_outliers()
    #print an.r_pruned
    #normalized = an.do_adaptive_normalize()
    #print normalized
    #an.do_adaptive_denormalize(np.reshape(normalized[:,-1], (-1,)))