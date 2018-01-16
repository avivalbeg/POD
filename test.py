import datetime as dt
from datetime import timedelta

import itertools
import time

import pandas_datareader.data as web
import pandas as pd
import numpy as np
np.set_printoptions(linewidth=200)
import random

from keras.layers.core import Dense
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.regularizers import l2, l1
from pandas_datareader import data

from models.DataLoader import DataLoader
from models.Model import ANN, TimeSeriesLstm
from models.eval_tools import AnnConfig
from models.utils import divXy


def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


START_DATE = dt.datetime(2000, 1, 1)
END_DATE = dt.datetime(2016, 12, 31)
LOOKBACK = 11
TICKERS = ['AAPL', 'AMZN', 'BAC']#, 'T', 'GOOG', 'MO', 'DAL', 'AA', 'AXP', 'BABA', 'ABT', 'UA', 'AMAT', 'AMGN', 'AAL', 'AIG', 'ALL', 'ADBE', 'GOOGL', 'ACN', 'ABBV', 'MT', 'LLY', 'AGN', 'APA', 'ADP', 'APC', 'AKAM', 'NLY']
PREDICT_FEATURE = 1  # We predict close value at next day

# Dataframe features:
" Adj Close       Close        High         Low        Open     Volume"


class StockDataLoader(DataLoader):
    def __init__(self, tickers, start_date, end_date, lookback):
        # Read data about given tickers from start to end date
        df = data.DataReader(tickers, 'yahoo', start_date, end_date)
        # print(df[:,:,tickers[0]]) # For example, see how one stock looks like
        # Collect data in given interval (determined by lookback)

        array = df.as_matrix()
        n_days = array.shape[1]

        X, y = [], []
        for i in range(0, n_days, lookback):
            # For every lookback days, we take the data
            # from those days and add it to the overall pool of data
            this_array = array[:, i:i + lookback, :]
            this_array = np.swapaxes(this_array, 0, 2)

            # Find where there are no nans:
            nums = np.squeeze(np.argwhere((np.max(np.max(np.isnan(this_array),
                                                        axis=1),axis=1)-1)*-1))
            # Take only those stocks (gets rid of stocks with NaN values)
            this_array = np.take(this_array,nums, axis=0)

            this_X, this_y = np.split(this_array, (-1,), 1)
            this_y = np.squeeze(this_y, 1) # Get rid of unnecessary axis

            X.append(this_X)
            y.append(this_y)
        # Now we turn this into a matrix
        # For simplicity, remove last datapoint to
        # X,y = shuffle_together(X,y)
        X = np.vstack(X[:-1])
        y = np.vstack(y[:-1])[:, PREDICT_FEATURE]

        X, y = unison_shuffled_copies(X, y)

        self.trainX = X[:int(len(X) / 2)]
        self.train_y = y[:int(len(X) / 2)]

        self.testX = X[int(len(X) / 2):]
        self.test_y = y[int(len(X) / 2):]


def main():
    data = StockDataLoader(TICKERS, START_DATE, END_DATE, LOOKBACK)

    nEpochs = 1000

    regs = sorted(np.logspace(-4, 2, num=4, base=10))
    batchSizes = [32, 16]
    hidSizes = [500, 1000, 500]
    layerCounts = [2, 3, 2, 4]
    dropouts = [.5, .2, .7]
    activations = ['sigmoid']

    configs = [AnnConfig(data, reg, batchSize, nEpochs, 0,
                         hidSize, nLayers, dropout, activation)
               for reg, batchSize, hidSize, nLayers, dropout, activation in
               itertools.product(regs, batchSizes, hidSizes, layerCounts,
                                 dropouts, activations)]

    configs = configs[:1]
    accs = {}
    for config in configs:
        model = TimeSeriesLstm(config)

        print("Trying config:")
        print([config])
        print()

        step = 5
        maxAcc, tnEpochs = 0, 0
        for i in range(1, int(config.nEpochs / step) + 1):
            print("Epoch " + str(i * step))

            model.train(config.data.trainX, config.data.train_y,
                        epochsOverride=step,
                        verbose=1)
            mse=model.eval(config.data.testX, config.data.test_y)
            print(int(mse))


if __name__ == '__main__':
    main()
