"""
Methods for evaluating models. The major method is compareModels, which takes a list of models
and a list of parameter configurations (which include a dataset), and evaluates each model
against each config+data combination.
"""

import sys
import matplotlib
# matplotlib.use("agg") # Try this if you get any matplotlib-related errors
import matplotlib.pyplot as plt
from os.path import join
from .Model import ANN, SoftmaxANN, KMeansModel, LogisticRegressionModel, \
    LinearRegressionModel, KNeighborsModel, SVMModel, \
    DoubleLayerSoftmaxANN, SKLearnModel, TFModel, PolynomialRegressionModel

import numpy as np
from .utils import toOneHot, dataSplit, accuracy, getMinibatches, printdb
from .DataLoader import RandomDataLoader, StanfordSentimentTreebankDataLoader, \
    SKLearnDataLoader, \
    ToyDataLoader, MnistDataLoader
from .params import *
import time
from pprint import pprint

import tflearn
import tensorflow as tf
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets
from copy import copy, deepcopy
import itertools

class Config(object):
    """Holds ANN hyperparams and data information.

    The config class is used to store various hyperparameters and dataset
    information parameters. ANN objects are passed a Config() object at
    instantiation.
    """
    def __init__(self, data, reg, batchSize, nEpochs, lr):
        self.data=data
        self.reg = reg
        self.nFeatures = data.nFeatures()
        self.nClasses = data.nClasses()
        self.nSamples = data.nSamples()
        self.batchSize = batchSize
        self.nEpochs = nEpochs
        self.lr = lr
        
        
    def __str__(self):
        return "\n".join([
                        "reg: " +str(self.reg), 
                       "nSamples: " +str(self.nSamples),
                       "nFeats: " + str(self.nFeatures),
                       "nClasses: " + str(self.nClasses),
                       "batchSize: " + str(self.batchSize),
                       "nEpochs: " + str(self.nEpochs),
                       "lr: " + str(self.lr),
                       ])
        
    def __repr__(self):
        return "-".join([
                        "reg:" +str(self.reg), 
                       "nSamples:" +str(self.nSamples),
                       "nFeats:" + str(self.nFeatures),
                       "nClasses:" + str(self.nClasses),
                       "batchSize:" + str(self.batchSize),
                       "nEpochs:" + str(self.nEpochs),
                       "lr:" + str(self.lr),
                       ])
    

def evaluateModel(modelClass, configs, debug=False):
    """Tries instantiates the given class with
    all given configurations and returns the 
    configuration which has best accuracy on the 
    dev set.
    
    @param modelClass: a subclass of Model.
    @param data: a subclass of DataLoader.
    
    @rtype: float. The test accuracy of @modelClass parametrized by the best 
    configuration in @configs.
    """
    if issubclass(modelClass, TFModel):
        results = evaluateTFModel(modelClass, configs, debug=debug)
    elif issubclass(modelClass, SKLearnModel):
        results = evaluateSKLearnModel(modelClass, configs, debug)
    else:
        raise NotImplementedError
    # Return results sorted dev by accuracy
    return sorted(results.items(), key=lambda x:x[1][0])

def evaluateSKLearnModel(modelClass, configs, debug=False):
    
    if debug:
        printdb("Testing " + modelClass.__name__)

    results = {}
    for config in configs:
        
        data = config.data
        
        if debug:
            printdb("Params:")
            printdb(str(config))
        model = modelClass(config)
        model.train(data.trainX, data.train_y)
        devAcc = model.eval(data.devX, data.dev_y)
        testAcc = model.eval(data.testX, data.test_y)
        
        if debug:
            printdb("Dev loss: " + str(devAcc))
            printdb("Test loss: " + str(testAcc))
            printdb("")
        results[config] = (devAcc, testAcc)

    return results

def evaluateTFModel(modelClass, configs, debug=False):
    
    results = {}
    for config in configs:
        if debug:
            print("Training %s with parameters:\n %s" % (modelClass.__name__, str(config)))
        
        # Creating a copy of the data that is one hot encoded (if needed)
        thisData = deepcopy(config.data)
        if issubclass(modelClass, ANN):
            thisData.encodeOneHot(False)
    
        model = modelClass(config)
           
        sess = tf.InteractiveSession()
        # tf.global_variables_initializer().run() # Try this if you get an error message
        sess.run(tf.global_variables_initializer())

        model.train(thisData.iterTraining(config.batchSize), sess)
        devAcc = model.eval(thisData.devX, thisData.dev_y, sess)
        testAcc = model.eval(thisData.testX, thisData.test_y, sess)
    
        if debug:
            printdb("Dev acc: " + str(devAcc))
            printdb("Test acc: " + str(testAcc))
        
        results[config] = (devAcc, testAcc)
        sess.close()
    return results

def compareModels(modelClasses, configs, debug=False, plotPath=None, plotParams=[]):
    """
    For each model, choose the best configuration from configs
    and then test each chosen model on the test set and report 
    accuracy results.
    dataOrDataFunction can be either a DataLoader object or a function which
    takes a config and returns a DataLoader object.
    
    @param modelClasses: a list of class objects which are assumed to be subclasses of Model.Model.
    @param debug: print debug info.
    @params plotParams: a list of parameters whose effect on the performance of the models
    will be plotted.
    @param plotPath: the directory to which plots will be saved.
    """

    accs = {}
    for modelClass in modelClasses:
        # List of tuples (config, (dev_accuracy, test_accuracy))
        results = evaluateModel(modelClass, configs, debug)
        accs[modelClass.__name__] = (results[-1][1][1], results[-1][0]) 

        # Plot if asked for it
        if plotPath:
            for plotParam in plotParams:
                X = []
                devY = []
                testY = []
                for res in results:
                    X.append(getattr(res[0], plotParam))
                    devY.append(res[1][0])
                    testY.append(res[1][1])

                plt.scatter(X, devY)
                plt.savefig(join(plotPath, modelClass.__name__ + "-" + plotParam+"-dev"))
                plt.close() 

                plt.scatter(X, testY)
                plt.savefig(join(plotPath, modelClass.__name__ + "-" + plotParam+"-test"))
                plt.close() 
    
    pprint (accs)
    
    

# @TODO: 
# - Make RandomDataLoader's labels follow some distributions
# - implement MSE
# - implement regularization for all models 
# - Save and recover trained models 
# - Fix MnistDataLoader and make sure that all DataLoaders are generalized

