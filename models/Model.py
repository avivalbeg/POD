import time

import numpy as np
import tensorflow as tf
from sklearn.pipeline import Pipeline

from .utils import crossEntropyLoss, getMinibatches, accuracy,\
    feature_normalize, append_bias_reshape


import time
from sklearn.preprocessing.data import OneHotEncoder, PolynomialFeatures

from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.linear_model.base import LinearRegression
from sklearn.neighbors.classification import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import mean_squared_error as mse, r2_score


# Abstract classes
  
class Model:
    """
    Interface for statistical models.
    Can be implemented by tensorflow-based models,
    in which case the session argument is relevant.
    
    inputPlaceholder is an n by m matrix where n is the number of training
    examples and m is the number of features.
    
    outputPlaceholder is a nX1 labels vector which assigns each training
    example a class.
    
    Api:
        train
        predict
        eval
    
    """
    
    def __init__(self, config):

        """Initializes the model.

        Args:
            _config: A model configuration object of type Config
        """
        self._config = config
    def train(self, X, y):
        """Train model on provided data in the given session.
        For tensorflow models: to use trained model, call Model.predict with the same session.

        Args:
            inputs: np.ndarray of shape (n_samples, nFeatures)
            labels: np.ndarray of shape (n_samples, nClasses)
            sess: tf.Session() (necessary only for models implemented with tensorflow)
        """

        raise NotImplementedError
    def predict(self, X):
        raise NotImplementedError
    def eval(self, X, y):
        """Evaluates the model on input matrix.
        Returns:
            double that measures accuracy.
        
        """
        return accuracy(y, self.predict(X))
    def __str__(self):
        return self.__class__.__name__
    def __repr__(self):
        return str(self)


class SKLearnModel(Model):
    pass

class TFModel(Model):
    

    def predict(self, X, session):  
        return session.run(self.prediction, feed_dict=
                            {self.inputsPlaceholder: X})  
              
    def train(self, XyIter, session):
        for _ in range(self._config.nEpochs):
            batch_xs, batch_ys = XyIter()
            session.run(self.train_step,
                         feed_dict={self.inputsPlaceholder: batch_xs,
                                    self.labelsPlaceholder: batch_ys})
    def eval(self, X, y, session):
        raise NotImplementedError

class ANN(Model):
    """Abstracts a Tensorflow graph for a learning task.

    We use various ANN classes as usual abstractions to encapsulate tensorflow
    computational graphs. Each algorithm you will construct in this homework will
    inherit from a ANN object.
    """
    
    # API
    
    def __init__(self, config):
        self._config = config
        # self._build()

    # Private
    
    def _build(self):
        self._addPlaceholders()
        self.prediction = self._addPredictionOp()
        self.cost = self._addLossOp(self.prediction)
        self.train_op = self._addTrainingOp(self.cost)
        
    def _trainOnBatch(self, sess, inputs_batch, labels_batch):
        """Perform one step of gradient descent on the provided batch of data.

        Args:
            sess: tf.Session()
            input_batch: np.ndarray of shape (n_samples, nFeatures)
            labels_batch: np.ndarray of shape (n_samples, nClasses)
        Returns:
            cost: cost over the batch (a scalar)
        """
        raise NotImplementedError

    def _predictOnBatch(self, inputsBatch, sess):
        """Make predictions for the provided batch of data

        Args:
            sess: tf.Session()
            input_batch: np.ndarray of shape (n_samples, nFeatures)
        Returns:
            predictions: np.ndarray of shape (n_samples, nClasses)
        """
        
        feedDict = {self.inputPlaceholder:inputsBatch}
        
        predictions = sess.run(self.prediction, feed_dict=feedDict)
        # Choose best class for each row in input batch
        return np.argmax(predictions, axis=1) 

    def _createFeedDict(self, inputsBatch, labelsBatch=None):
        """Creates the feed_dict for one step of training.

        A feed_dict takes the form of:
        feed_dict = {
                <placeholder>: <tensor of values to be passed for placeholder>,
                ....
        }

        If labelsBatch is None, then no labels are added to feed_dict.

        Args:
            inputsBatch: A batch of input data.
            labelsBatch: A batch of label data.
        Returns:
            feed_dict: The feed dictionary mapping from placeholders to values.
        """
        raise NotImplementedError("Each Model must re-implement this method.")
    
    def _addPredictionOp(self):
        """Implements the core of the model that transforms a batch of input data into predictions.

        Returns:
            prediction: A tensor of shape (batchSize, nClasses)
        """
        raise NotImplementedError("Each Model must re-implement this method.")

    def _addLossOp(self, pred):
        """Adds Ops for the cost function to the computational graph.

        Args:
            prediction: A tensor of shape (batchSize, nClasses)
        Returns:
            cost: A 0-d tensor (scalar) output
        """
        raise NotImplementedError("Each Model must re-implement this method.")

    def _addTrainingOp(self, loss):
        """Sets up the training Ops.

        Creates an optimizer and applies the gradients to all trainable variables.
        The Op returned by this function is what must be passed to the
        sess.run() to train the model. See

        https://www.tensorflow.org/versions/r0.7/api_docs/python/train.html#Optimizer

        for more information.

        Args:
            cost: Loss tensor (a scalar).
        Returns:
            train_op: The Op for training.
        """

        raise NotImplementedError("Each Model must re-implement this method.")

    def _addPlaceholders(self):
        """Adds placeholder variables to tensorflow computational graph.

        Tensorflow uses placeholder variables to represent locations in a
        computational graph where data is inserted.  These placeholders are used as
        inputs by the rest of the model building and will be fed data during
        training.

        See for more information:
        https://www.tensorflow.org/versions/r0.7/api_docs/python/io_ops.html#placeholders
        """
        raise NotImplementedError("Each Model must re-implement this method.")
  
 
# Concrete classes 
    
class LogisticRegressionModel(SKLearnModel):
    def __init__(self, config):
        Model.__init__(self, config)
        self._model = LogisticRegression(C=1.0 / (config.reg + 1e-12),)
    def train(self, X, y):
        self._model.fit(X, y)
    def predict(self, X):
        return self._model.predict(X)

class KMeansModel(SKLearnModel):
    def __init__(self, config):
        Model.__init__(self, config)
        self._model = KMeans(n_clusters=config.nClasses)
    def train(self, X, y):
        self._model.fit(X)
    def predict(self, X):
        return self._model.predict(X)

class KNeighborsModel(SKLearnModel):
    def __init__(self, config):
        Model.__init__(self, config)
        self._model = KNeighborsClassifier()
    def train(self, X, y):
        self._model.fit(X, y)
    def predict(self, X):
        return self._model.predict(X)


class SVMModel(SKLearnModel):
    def __init__(self, config):
        Model.__init__(self, config)
        self._model = SVC()
    def train(self, X, y):
        self._model.fit(X, y)
    def predict(self, X):
        return self._model.predict(X)

class RegressionModel(SKLearnModel):
    def __init__(self, config):
        SKLearnModel.__init__(self, config)
    def train(self, X, y):
        self._model.fit(X, y)
    def predict(self, X):
        return self._model.predict(X)
    def eval(self, X, y):
        """Returns the MSE of the model."""
        return mse(y,self.predict(X))

class LinearRegressionModel(RegressionModel):
    def __init__(self, config):
        SKLearnModel.__init__(self, config)
        self._model = LinearRegression()

class PolynomialRegressionModel(RegressionModel):
    def __init__(self, config,degree=2):
        SKLearnModel.__init__(self, config)
        self._model = Pipeline([('poly', PolynomialFeatures(degree=degree)),
                   ('linear', LinearRegression(fit_intercept=False))])

class ClassifierANN(ANN):
    """Implements a Softmax classifier with cross-entropy cost."""
            
    def eval(self, X, y, session):    
        correct_prediction = tf.equal(tf.argmax(self.prediction, 1), tf.argmax(self.labelsPlaceholder, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        return (session.run(accuracy, feed_dict={self.inputsPlaceholder: X,
                                              self.labelsPlaceholder: y}))



class SoftmaxANN(ClassifierANN,TFModel):
    
    def __init__(self, config):
        ClassifierANN.__init__(self, config)

        # placeholders
        self.inputsPlaceholder = tf.placeholder(tf.float32, [None, self._config.nFeatures])
        self.labelsPlaceholder = tf.placeholder(tf.float32, [None, self._config.nClasses])

        # hidden layer
        W = tf.Variable(tf.zeros((self._config.nFeatures, self._config.nClasses)))
        b = tf.Variable(tf.zeros(self._config.nClasses))
        y = tf.matmul(self.inputsPlaceholder, W) + b

        # output
        self.prediction = tf.nn.softmax(y)

        self.cost = tf.negative(tf.reduce_sum(tf.log(tf.reduce_sum(tf.multiply(tf.cast(self.labelsPlaceholder, tf.float32),
                                                                         self.prediction),axis=1)),axis=0))
        self.train_step = tf.train.GradientDescentOptimizer(self._config.lr).minimize(self.cost)

    def predict(self, X, session):
        return TFModel.predict(self,X,session)
    def train(self, XyIter, session):
        TFModel.train(self,XyIter,session)

class DoubleLayerSoftmaxANN(ClassifierANN,TFModel):
    """
    Based on:
    https://stackoverflow.com/questions/38136961/how-to-create-2-layers-neural-network-using-tensorflow-and-python-on-mnist-data
    """
    
    def __init__(self, config, hiddenLayerSize=1000):
        ClassifierANN.__init__(self, config)
        
        # placeholders 
        self.inputsPlaceholder = tf.placeholder(tf.float32, [None, self._config.nFeatures])
        self.labelsPlaceholder = tf.placeholder(tf.float32, [None, self._config.nClasses])
        
        # layer 1
        W1 = tf.Variable(tf.random_normal((self._config.nFeatures, hiddenLayerSize)))
        b1 = tf.Variable(tf.random_normal((1,)))
        y1 = tf.nn.sigmoid(tf.matmul(self.inputsPlaceholder, W1) + b1) 
        
        # layer 2
        W2 = tf.Variable(tf.random_normal((hiddenLayerSize, self._config.nClasses)))
        b2 = tf.Variable(tf.random_normal((1,)))
        y2 = tf.nn.softmax(tf.matmul(y1, W2) + b2)
        
        # output
        self.prediction = y2
        
        self.cost = tf.reduce_mean(-tf.reduce_sum(self.labelsPlaceholder * tf.log(self.prediction),
        reduction_indices=[1]))
        self.train_step = tf.train.GradientDescentOptimizer(self._config.lr).minimize(self.cost)

    def predict(self, X, session):
        return TFModel.predict(self,X,session)
    def train(self, XyIter, session):
        TFModel.train(self,XyIter,session)


class RegressionModel(TFModel):
            
    def eval(self, X, y, session):    
        return (session.run(self.cost, feed_dict={self.labelsPlaceholder: y,
                                                  self.prediction: self.predict(X,session)}))
