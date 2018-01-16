import time

import numpy as np
import time
import os

import scipy
import tensorflow as tf
from keras.layers.core import Dropout
from keras.regularizers import l2, l1
from tensorflow.python.ops.rnn import static_rnn
from tensorflow.contrib.rnn import LSTMCell

from sklearn.preprocessing.data import OneHotEncoder, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.linear_model.base import LinearRegression
from sklearn.neighbors.classification import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import mean_squared_error as mse, r2_score

import keras
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence

from .utils import accuracy


import gym
import numpy as np
import random
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam

from collections import deque



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

        feedDict = {self.inputPlaceholder: inputsBatch}

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
        self._model = LogisticRegression(C=1.0 / (config.reg + 1e-12), )

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
        return mse(y, self.predict(X))


class LinearRegressionModel(RegressionModel):
    def __init__(self, config):
        SKLearnModel.__init__(self, config)
        self._model = LinearRegression()


class PolynomialRegressionModel(RegressionModel):
    def __init__(self, config, degree=2):
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

class FeedforwardNeuralClassifier(ClassifierANN):
        def __init__(self, config):
            Model.__init__(self, config)
            from keras.regularizers import l2, l1
            from keras import Sequential
            from keras.layers import Dense
            assert config.nLayers >= 1

            self._config = config

            model = Sequential()
            model.add(Dense(config.hidSize,
                            input_dim=config.nFeatures,
                            kernel_regularizer=l2(config.reg),
                            bias_regularizer=l1(config.reg),
                            activity_regularizer=l1(config.reg),
                            activation=config.activation))

            for _ in range(config.nLayers):
                model.add(Dense(config.hidSize,
                                kernel_regularizer=l2(config.reg),
                                bias_regularizer=l1(config.reg),
                                activity_regularizer=l1(config.reg),
                                activation=config.activation,

                                ))

            model.add(Dense(config.nClasses, activation='softmax'))
            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

            self._model = model

        def train(self, X, y, epochsOverride=0, verbose=0):
            self._model.fit(X, y,
                            batch_size=self._config.batchSize,
                            epochs=epochsOverride or self._config.nEpochs,
                            verbose=verbose)

        def eval(self, X, y, verbose=0):
            scores = self._model.evaluate(X, y, verbose=verbose)
            return scores[1]

        def predict(self, X):
            return self._model.predict(X)

        def save(self, path, overwrite=True):
            self._model.save(path, overwrite=overwrite)

class SoftmaxANN(ClassifierANN, TFModel):
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

        self.cost = tf.negative(
            tf.reduce_sum(tf.log(tf.reduce_sum(tf.multiply(tf.cast(self.labelsPlaceholder, tf.float32),
                                                           self.prediction), axis=1)), axis=0))
        self.train_step = tf.train.GradientDescentOptimizer(self._config.lr).minimize(self.cost)

    def predict(self, X, session):
        return TFModel.predict(self, X, session)

    def train(self, XyIter, session):
        TFModel.train(self, XyIter, session)


class DoubleLayerSoftmaxANN(SoftmaxANN):
    """
    Based on:
    https://stackoverflow.com/questions/38136961/how-to-create-2-layers-neural-network-using-tensorflow-and-python-on-mnist-data
    """

    def __init__(self, config, hidSize=1000):
        ClassifierANN.__init__(self, config)

        # placeholders 
        self.inputsPlaceholder = tf.placeholder(tf.float32, [None, self._config.nFeatures])
        self.labelsPlaceholder = tf.placeholder(tf.float32, [None, self._config.nClasses])

        # layer 1
        W1 = tf.Variable(tf.random_normal((self._config.nFeatures, hidSize)))
        b1 = tf.Variable(tf.random_normal((1,)))
        y1 = tf.nn.sigmoid(tf.matmul(self.inputsPlaceholder, W1) + b1)

        # layer 2
        W2 = tf.Variable(tf.random_normal((hidSize, self._config.nClasses)))
        b2 = tf.Variable(tf.random_normal((1,)))
        y2 = tf.nn.softmax(tf.matmul(y1, W2) + b2)

        # output
        self.prediction = y2

        self.cost = tf.reduce_mean(-tf.reduce_sum(self.labelsPlaceholder * tf.log(self.prediction),
                                                  reduction_indices=[1]))
        self.train_step = tf.train.GradientDescentOptimizer(self._config.lr).minimize(self.cost)


class TripleLayerSoftmaxANN(SoftmaxANN):
    """
    Based on:
    https://stackoverflow.com/questions/38136961/how-to-create-2-layers-neural-network-using-tensorflow-and-python-on-mnist-data
    """

    def __init__(self, config, hidSize1=1000, hidSize2=1000):
        ClassifierANN.__init__(self, config)

        # placeholders 
        self.inputsPlaceholder = tf.placeholder(tf.float32, [None, self._config.nFeatures])
        self.labelsPlaceholder = tf.placeholder(tf.float32, [None, self._config.nClasses])

        # layer 1
        W1 = tf.Variable(tf.random_normal((self._config.nFeatures, hidSize1)))
        b1 = tf.Variable(tf.random_normal((1,)))
        y1 = tf.nn.sigmoid(tf.matmul(self.inputsPlaceholder, W1) + b1)

        # layer 2
        W2 = tf.Variable(tf.random_normal((hidSize1, hidSize2)))
        b2 = tf.Variable(tf.random_normal((1,)))
        y2 = tf.nn.softmax(tf.matmul(y1, W2) + b2)

        # layer 3
        W3 = tf.Variable(tf.random_normal((hidSize2, self._config.nClasses)))
        b3 = tf.Variable(tf.random_normal((1,)))
        y3 = tf.nn.softmax(tf.matmul(y2, W3) + b3)

        # output
        self.prediction = y3

        self.cost = tf.reduce_mean(-tf.reduce_sum(self.labelsPlaceholder * tf.log(self.prediction),
                                                  reduction_indices=[1]))
        self.train_step = tf.train.GradientDescentOptimizer(self._config.lr).minimize(self.cost)


class TfLstmClassifier(TFModel, ANN):
    def __init__(self, config):
        # TODO: Make sure configs have these fields
        num_layers = config.nLayers
        hidden_size = config.hiddenSize
        max_grad_norm = config.maxGradNorm
        nDims = config.nFeatures

        self.batch_size = config.batchSize

        learning_rate = config.lr
        num_classes = config.nClasses
        self.input = tf.placeholder(tf.float32, [None, nDims], name='input')
        self.labels = tf.placeholder(tf.int64, [None, nDims], name='labels')
        self.keep_prob = tf.placeholder("float", name='Drop_out_keep_prob')
        with tf.name_scope("LSTM_setup") as scope:
            def single_cell():
                return tf.contrib.rnn.DropoutWrapper(LSTMCell(hidden_size), output_keep_prob=self.keep_prob)

            cell = tf.contrib.rnn.MultiRNNCell([single_cell() for _ in range(num_layers)])
            initial_state = cell.zero_state(self.batch_size, tf.float32)
        print(self.input)
        input_list = tf.unstack(self.input, axis=1)
        print((input_list))
        print(len(input_list))
        print((input_list[0]))
        outputs, _ = static_rnn(cell, input_list, dtype=tf.float32)
        output = outputs[-1]

        # Generate a classification from the last cell_output
        # Note, this is where timeseries classification differs from sequence to sequence
        # modelling. We only output to Softmax at last time step
        with tf.name_scope("Softmax") as scope:
            with tf.variable_scope("Softmax_params"):
                softmax_w = tf.get_variable("softmax_w", [hidden_size, num_classes])
                softmax_b = tf.get_variable("softmax_b", [num_classes])
            logits = tf.nn.xw_plus_b(output, softmax_w, softmax_b)
            # Use sparse Softmax because we have mutually exclusive classes
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=self.labels, name='softmax')
            self.cost = tf.reduce_sum(loss) / self.batch_size
        with tf.name_scope("Evaluating_accuracy") as scope:
            correct_prediction = tf.equal(tf.argmax(logits, 1), self.labels)
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
            h1 = tf.summary.scalar('accuracy', self.accuracy)
            h2 = tf.summary.scalar('cost', self.cost)

        with tf.name_scope("Optimizer") as scope:
            tvars = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars),
                                              max_grad_norm)  # We clip the gradients to prevent explosion
            optimizer = tf.train.AdamOptimizer(learning_rate)
            gradients = zip(grads, tvars)
            self.train_op = optimizer.apply_gradients(gradients)

        self.merged = tf.summary.merge_all()
        self.init_op = tf.global_variables_initializer()
        print('Finished computation graph')

class LstmClassifier(ANN):
    def __init__(self, config):
        ANN.__init__(self, config)
        print(config.nLayers)
        assert config.nLayers >= 1

        self._config = config

        self._model = Sequential()


        # Add input layer

        self._model.add(Dense(config.hidSize, input_shape=config.inputShape,
                              kernel_regularizer=l2(config.reg),
                              activity_regularizer=l1(config.reg)
                              ))


        # Add LSTM layers

        lstmLayer = lambda retSeqs: LSTM(
                                config.hidSize,
                                 kernel_regularizer=l2(config.reg),
                                 recurrent_regularizer=l2(config.reg),
                                 bias_regularizer=l1(config.reg),
                                 activity_regularizer=l1(config.reg),
                                 dropout=config.dropout,
                                 recurrent_dropout=0.,  # vary?
                                 return_sequences=retSeqs)

        for _ in range(config.nLayers-1):
            self._model.add(lstmLayer(True))
        self._model.add(lstmLayer(False))

        # Add softmax layer to output classes
        self._model.add(Dense(config.data.nClasses(), activation='softmax'))

        self._model.compile(loss='categorical_crossentropy',
                            optimizer=Adam(lr=config.lr),
                            metrics=['accuracy'])

    def train(self, X, y, epochsOverride=0, verbose=1):
        self._model.fit(X, y,
                        batch_size=self._config.batchSize,
                        epochs=epochsOverride or self._config.nEpochs,
                        verbose=verbose)

    def eval(self, X, y, verbose=0):
        scores = self._model.evaluate(X, y, verbose=verbose)
        return scores[1]

    def predict(self, X):
        return self._model.predict(X)

    def save(self, path, overwrite=True):
        self._model.save(path, overwrite=overwrite)

    def trainGraded(self, step=1, verbose=1):
        """Finds the ideal #epochs for this model."""

        trainX, train_y, testX, test_y = self._config.data.trainX, self._config.data.train_y, self._config.data.testX, self._config.data.test_y
        maxAcc, nEpochs = 0, 0
        for i in range(1, int(self._config.nEpochs / step) + 1):
            self._model.fit(trainX, train_y, epochs=step,
                            batch_size=self._config.batchSize,
                            verbose=verbose)
            _, acc = self._model.evaluate(testX, test_y, verbose=0)
            if acc > maxAcc:
                print("New accuracy:", acc)
                maxAcc, nEpochs = acc, i * step
        return maxAcc, nEpochs


class TimeSeriesLstm(ANN):
    def __init__(self, config):
        ANN.__init__(self, config)
        assert config.nLayers >= 1

        self._config = config

        self._model = Sequential()

        # Add input layer

        self._model.add(LSTM(config.hidSize,
                             input_shape=config.inputShape,
                             kernel_regularizer=l2(config.reg),
                             recurrent_regularizer=l2(config.reg),
                             bias_regularizer=l1(config.reg),
                             activity_regularizer=l1(config.reg),
                             dropout=config.dropout,
                             recurrent_dropout=0.,  # vary?
                             return_sequences=True
                             ))

        # Add LSTM layers

        lstmLayer = lambda retSeqs: LSTM(
            config.hidSize,
            kernel_regularizer=l2(config.reg),
            recurrent_regularizer=l2(config.reg),
            bias_regularizer=l1(config.reg),
            activity_regularizer=l1(config.reg),
            dropout=config.dropout,
            recurrent_dropout=0.,  # vary?
            return_sequences=retSeqs)

        for _ in range(config.nLayers - 1):
            self._model.add(lstmLayer(True))
        self._model.add(lstmLayer(False))

        # Add softmax layer to output classes
        self._model.add(Dense(1))

        self._model.compile(loss='mean_squared_error', optimizer='adam')

    def train(self, X, y, epochsOverride=0, verbose=1):
        self._model.fit(X, y,
                        batch_size=self._config.batchSize,
                        epochs=epochsOverride or self._config.nEpochs,
                        verbose=verbose)

    def eval(self, X, y, verbose=1):
        return self._model.evaluate(X, y, verbose=verbose)

    def predict(self, X):
        return self._model.predict(X)

    def save(self, path, overwrite=True):
        self._model.save(path, overwrite=overwrite)


class DQN:
    def __init__(self, env):
        self.env = env
        self.memory = deque(maxlen=2000)

        self.gamma = 0.85
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.005
        self.tau = .125

        self.model = self.create_model()
        self.target_model = self.create_model()

    def create_model(self):
        model = Sequential()
        state_shape = self.env.observation_space.shape
        model.add(Dense(24, input_dim=state_shape[0], activation="relu"))
        model.add(Dense(48, activation="relu"))
        model.add(Dense(24, activation="relu"))
        model.add(Dense(self.env.action_space.n))
        model.compile(loss="mean_squared_error",
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def act(self, state):
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        return np.argmax(self.model.predict(state)[0])

    def remember(self, state, action, reward, new_state, done):
        self.memory.append([state, action, reward, new_state, done])

    def replay(self):
        batch_size = 32
        if len(self.memory) < batch_size:
            return

        samples = random.sample(self.memory, batch_size)
        for sample in samples:
            state, action, reward, new_state, done = sample
            target = self.target_model.predict(state)
            if done:
                target[0][action] = reward
            else:
                Q_future = max(self.target_model.predict(new_state)[0])
                target[0][action] = reward + Q_future * self.gamma
            self.model.fit(state, target, epochs=1, verbose=0)

    def target_train(self):
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i] * self.tau + target_weights[i] * (1 - self.tau)
        self.target_model.set_weights(target_weights)

    def save_model(self, fn):
        self.model.save(fn)


# Example how to use:
# def main(self):
#     env = gym.make("MountainCar-v0")
#     gamma = 0.9
#     epsilon = .95
#
#     trials = 1000
#     trial_len = 500
#
#     # updateTargetNetwork = 1000
#     dqn_agent = DQN(env=env)
#     steps = []
#     for trial in range(trials):
#         cur_state = env.reset().reshape(1, 2)
#         for step in range(trial_len):
#             action = dqn_agent.act(cur_state)
#             new_state, reward, done, _ = env.step(action)
#
#             # reward = reward if not done else -20
#             new_state = new_state.reshape(1, 2)
#             dqn_agent.remember(cur_state, action, reward, new_state, done)
#
#             dqn_agent.replay()  # internally iterates default (prediction) model
#             dqn_agent.target_train()  # iterates target model
#
#             cur_state = new_state
#             if done:
#                 break
#         if step >= 199:
#             print("Failed to complete in trial {}".format(trial))
#             if step % 10 == 0:
#                 dqn_agent.save_model("trial-{}.model".format(trial))
#         else:
#             print("Completed in {} trials".format(trial))
#             dqn_agent.save_model("success.model")
#             break

