
import sys
import time
import numpy as np

import tensorflow as tf
from pandas.core.frame import DataFrame


def idEncode(it):
    """Encodes all elements of iterable it with unique ids 
    starting from 0. Returns a numpy array."""
    idCounter = 0
    ids = {}
    new = np.zeros(len(it))
    for i in range(len(it)):
        x = it[i]
        if not x in ids:
            ids[x] = idCounter
            idCounter += 1
        new[i] = ids[x]
    return np.int32(new)

def doublesToClasses(y, nClasses):
    """
    Partition a vector y into nClasses distinct classes
    """
    y=y.reshape((1,len(y)))[0]
    mn, mx = np.min(y), np.max(y)
    margin = (mx - mn) / float(nClasses)  # The size of each class
    chunkified = list(map(lambda x:x - (x % margin),
                              y))
    return idEncode(chunkified)

def feature_normalize(dataset):
    mu = np.mean(dataset, axis=0)
    sigma = np.std(dataset, axis=0)
    return (dataset - mu) / sigma

def append_bias_reshape(features, labels):
    n_training_samples = features.shape[0]
    n_dim = features.shape[1]
    f = np.reshape(np.c_[np.ones(n_training_samples), features], [n_training_samples, n_dim + 1])
    l = np.reshape(labels, [n_training_samples, 1])
    return f, l

def printdb(x=""):
    print("% " + str(x))

def accuracy(y, yhat):
    """ Precision for classifier """
    assert(y.shape == yhat.shape)
    return np.sum(y == yhat) * 100.0 / y.size


def dataSplit(n, ration=.8):
    """
    Split a number into two integers that are
    as close as possible to the specified ratio.
    """
    split1 = n * ration
    split2 = n - split1 + split1 % 1
    
    return int(split1), int(split2)

def toOneHot(y):
    """
    Transform an n-by-1 vector into an n-by-m
    matrix of one hot encoded row vectors.
    """
    n = np.max(y) + 1
    return np.eye(n)[y]

def fromOneHot(y):
    """
    Transform an n-by-m matrix into an n-by-1
    vector.
    """
    return np.apply_along_axis(lambda x:np.argmax(x), 0, y)

def crossEntropyLoss(y, yhat):
    """
    Compute the cross entropy loss in tensorflow.
    The loss should be summed over the current minibatch.

    y is a one-hot tensor of shape (n_samples, n_classes) and yhat is a tensor
    of shape (n_samples, n_classes). y should be of dtype tf.int32, and yhat should
    be of dtype tf.float32.

    The functions tf.to_float, tf.reduce_sum, and tf.log might prove useful. (Many
    solutions are possible, so you may not need to use all of these functions).

    Args:
        y:    tf.Tensor with shape (n_samples, n_classes). One-hot encoded.
        yhat: tf.Tensor with shape (n_sample, n_classes). Each row encodes a
                    probability distribution and should sum to 1.
    Returns:
        out:  tf.Tensor with shape (1,) (Scalar output). You need to construct this
                    tensor in the problem.
    """

    y = tf.cast(y, tf.float32)
    out = tf.negative(tf.reduce_sum(tf.log(tf.reduce_sum(tf.multiply(y, yhat), axis=1)), axis=0))

    return out



def getMinibatches(data, minibatchSize, shuffle=True):
    """
    Iterates through the provided data one minibatch at at time. You can use this function to
    iterate through data in minibatches as follows:

        for inputs_minibatch in getMinibatches(inputs, minibatchSize):
            ...

    Or with multiple data sources:

        for inputs_minibatch, labels_minibatch in getMinibatches([inputs, labels], minibatchSize):
            ...

    Args:
        data: there are two possible values:
            - a list or numpy array
            - a list where each element is either a list or numpy array
        minibatchSize: the maximum number of items in a minibatch
        shuffle: whether to randomize the order of returned data
    Returns:
        minibatches: the return value depends on data:
            - If data is a list/array it yields the next minibatch of data.
            - If data a list of lists/arrays it returns the next minibatch of each element in the
              list. This can be used to iterate through multiple data sources
              (e.g., features and labels) at the same time.

    """
    list_data = type(data) is list and (type(data[0]) is list or type(data[0]) is np.ndarray)
    data_size = len(data[0]) if list_data else len(data)
    indices = np.arange(data_size)
    if shuffle:
        np.random.shuffle(indices)
    for minibatchStart in np.arange(0, data_size, minibatchSize):
        minibatchIndices = indices[minibatchStart:minibatchStart + minibatchSize]
        yield [minibatch(d, minibatchIndices) for d in data] if list_data \
            else minibatch(data, minibatchIndices)


def minibatch(data, minibatch_idx):
    return data[minibatch_idx] if type(data) is np.ndarray else [data[i] for i in minibatch_idx]


def test_all_close(name, actual, expected):
    if actual.shape != expected.shape:
        raise ValueError("{:} failed, expected output to have shape {:} but has shape {:}"
                         .format(name, expected.shape, actual.shape))
    if np.amax(np.fabs(actual - expected)) > 1e-6:
        raise ValueError("{:} failed, expected {:} but value is {:}".format(name, expected, actual))
    else:
        print (name, "passed!")


def logged_loop(iterable, n=None):
    if n is None:
        n = len(iterable)
    step = max(1, n / 1000)
    prog = Progbar(n)
    for i, elem in enumerate(iterable):
        if i % step == 0 or i == n - 1:
            prog.update(i + 1)
        yield elem


class Progbar(object):
    """
    Progbar class copied from keras (https://github.com/fchollet/keras/)
    Displays a progress bar.
    # Arguments
        target: Total number of steps expected.
        interval: Minimum visual progress update interval (in seconds).
    """

    def __init__(self, target, width=30, verbose=1):
        self.width = width
        self.target = target
        self.sum_values = {}
        self.unique_values = []
        self.start = time.time()
        self.total_width = 0
        self.seen_so_far = 0
        self.verbose = verbose

    def update(self, current, values=[], exact=[]):
        """
        Updates the progress bar.
        # Arguments
            current: Index of current step.
            values: List of tuples (name, value_for_last_step).
                The progress bar will display averages for these values.
            exact: List of tuples (name, value_for_last_step).
                The progress bar will display these values directly.
        """

        for k, v in values:
            if k not in self.sum_values:
                self.sum_values[k] = [v * (current - self.seen_so_far), current - self.seen_so_far]
                self.unique_values.append(k)
            else:
                self.sum_values[k][0] += v * (current - self.seen_so_far)
                self.sum_values[k][1] += (current - self.seen_so_far)
        for k, v in exact:
            if k not in self.sum_values:
                self.unique_values.append(k)
            self.sum_values[k] = [v, 1]
        self.seen_so_far = current

        now = time.time()
        if self.verbose == 1:
            prev_total_width = self.total_width
            sys.stdout.write("\b" * prev_total_width)
            sys.stdout.write("\r")

            numdigits = int(np.floor(np.log10(self.target))) + 1
            barstr = '%%%dd/%%%dd [' % (numdigits, numdigits)
            bar = barstr % (current, self.target)
            prog = float(current) / self.target
            prog_width = int(self.width * prog)
            if prog_width > 0:
                bar += ('=' * (prog_width - 1))
                if current < self.target:
                    bar += '>'
                else:
                    bar += '='
            bar += ('.' * (self.width - prog_width))
            bar += ']'
            sys.stdout.write(bar)
            self.total_width = len(bar)

            if current:
                time_per_unit = (now - self.start) / current
            else:
                time_per_unit = 0
            eta = time_per_unit * (self.target - current)
            info = ''
            if current < self.target:
                info += ' - ETA: %ds' % eta
            else:
                info += ' - %ds' % (now - self.start)
            for k in self.unique_values:
                if type(self.sum_values[k]) is list:
                    info += ' - %s: %.4f' % (k, self.sum_values[k][0] / max(1, self.sum_values[k][1]))
                else:
                    info += ' - %s: %s' % (k, self.sum_values[k])

            self.total_width += len(info)
            if prev_total_width > self.total_width:
                info += ((prev_total_width - self.total_width) * " ")

            sys.stdout.write(info)
            sys.stdout.flush()

            if current >= self.target:
                sys.stdout.write("\n")

        if self.verbose == 2:
            if current >= self.target:
                info = '%ds' % (now - self.start)
                for k in self.unique_values:
                    info += ' - %s: %.4f' % (k, self.sum_values[k][0] / max(1, self.sum_values[k][1]))
                sys.stdout.write(info + "\n")

    def add(self, n, values=[]):
        self.update(self.seen_so_far + n, values)




def getSentenceFeatures(termDict, wordVectors, words):
    """
    Obtain the words feature for sentiment analysis by averaging its
    word vectors.

    Inputs:
    termDict -- a dictionary that maps words to their indices in
              the word vector list
    wordVectors -- word vectors (each row) for all termDict
    words -- a list of words in the words of interest

    Output:
    - sentVector: feature vector for the words
    """
    sentVector = np.zeros((wordVectors.shape[1],))
    
    for s in words:
        sentVector += wordVectors[termDict[s], :]

    sentVector *= 1.0 / len(words)

    return sentVector

    
