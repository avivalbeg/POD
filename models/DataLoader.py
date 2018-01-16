"""
Classes that represent data. They are meant to be passed as parameters to the Config class.
Some classes corresponds to a single dataset, and some take init arguments, like 
files from which data are read.
"""

import sys

import pandas as pd
import numpy as np
from .utils import *


from sklearn import datasets

from sklearn.model_selection import train_test_split
from sklearn.datasets.base import load_boston
from numpy import genfromtxt


from .params import *
import os
from os.path import join
from pprint import pprint
from pandas.core.frame import DataFrame
from random import sample


class DataLoader:
    """
    
    The data can be accessed directly via:
    
    self.trainX, self.train_y
    self.devX, self.dev_y
    self.testX, self.test_y
    
    def iterTraining(self, int batchSize):
        Iterate over training examples, batchSize at a time.
    def nSamples(self):
        Return number of datapoints.
    def nFeatures(self):
        Return number of features for input vector.
    def nClasses(self):
        Returns number of classes, or np.nan if data is continuous.
    
    def encodeOneHot(self, bool trainOnly=True):
        Encodes labels as one hot vectors. If trainOnly is set 
        to true, then only training labels are encoded.
    def decodeOneHot(self, bool trainOnly=True):
        The inverse of encodeOneHot.
    """    
    
    def iterTraining(self, batchSize):
        return lambda: next(getMinibatches([self.trainX, self.train_y], batchSize))
    def nSamples(self):
        return self.trainX.shape[0]
    def nFeatures(self):
        return self.trainX.shape[1]

    def nClasses(self):
        # If not one hot encoded, return maximal value
        if len(self.train_y.shape)==1:
            return np.max(self.train_y) + 1
        # If one hot ended, return number of columns
        else:
            return self.train_y.shape[1]   
    
    def encodeOneHot(self, trainOnly=True):
        self.train_y = toOneHot(self.train_y)
        if not trainOnly:  
            self.dev_y = toOneHot(self.dev_y)  
            self.test_y = toOneHot(self.test_y)
    
    def decodeOneHot(self, trainOnly=True):
        self.train_y = fromOneHot(self.train_y)
        if not trainOnly:  
            self.dev_y = fromOneHot(self.dev_y)  
            self.test_y = fromOneHot(self.test_y)

    def __len__(self):
        return self.trainX.shape[0]+self.testX.shape[0]+self.devX.shape[0]
    
    def __str__(self):
        """
        Ugly default representation.
        """
        out = ""
        out += str(self.trainX) +"\n"
        out += str(self.train_y)+"\n"
        out += str(self.testX)+"\n"
        out += str(self.test_y)+"\n"
        out += str(self.devX)+"\n"
        out += str(self.dev_y)+"\n"
        out += str(self.trainX.shape)+"\n"
        out += str(self.train_y.shape)+"\n"
        out += str(self.testX.shape)+"\n"
        out += str(self.test_y.shape)+"\n"
        out += str(self.devX.shape)+"\n"
        out += str(self.dev_y.shape)+"\n"
        
        return out
    def __repr__(self):
        return str(self)

class ToyDataLoader(DataLoader):
    def __init__(self):
        
        self.trainX = np.array([[0, 0], [1, 1]])
        self.train_y  = np.array([0,1])
        
        self.devX = np.array([[0, 0], [1, 1]])
        self.dev_y  = np.array([0,1])

        self.testX = np.array([[2., 2.]])
        self.test_y  = np.array([1])
        
class ToyDataLoader2(DataLoader):
    def __init__(self):
        
        self.trainX = np.array([[0, 0], [1, 1]])
        self.train_y  = np.array([0,1])
        
        self.devX = np.array([[0, 0], [1, 1]])
        self.dev_y  = np.array([0,1])

        self.testX = np.array([[2., 2.]])
        self.test_y  = np.array([1])

class RandomDataLoader(DataLoader):
    """
    Generate purely random data. Useful for making sure that a model fails to 
    predict any pattern and only gets chance accuracy.
    """
    
    def __init__(self):
        
        
        nTrain, nTest = dataSplit(N_RAND_SAMPLES)
        nTrain, nDev = dataSplit(nTrain)
        
        self.trainX = np.random.rand(nTrain, N_RAND_FEATURES)
        self.train_y  = np.random.uniform(low=0,high=N_RAND_CLASSES,size=nTrain).astype(np.int32)
        
        self.devX = np.random.rand(nDev, N_RAND_FEATURES)
        self.dev_y  = np.random.uniform(low=0,high=N_RAND_CLASSES,size=nDev).astype(np.int32)

        self.testX = np.random.rand(nTest, N_RAND_FEATURES)
        self.test_y  = np.random.uniform(low=0,high=N_RAND_CLASSES,size=nTest).astype(np.int32)


class StanfordSentimentTreebankDataLoader(DataLoader):
    """
    A class for loading the Stanford Sentiment Treebank corpus.
    """
        
    def __init__(self):
        
        # Load the dataset
        dataset = StanfordSentiment()
        tokens = dataset.tokens()
        nWords = len(tokens)
        
        wordVectors = glove.loadWordVectors(tokens)
        dimVectors = wordVectors.shape[1]
        # Load the train set
        trainset = dataset.getTrainSentences()
        nTrain = len(trainset)
        self.trainX = np.zeros((nTrain, dimVectors))
        self.train_y = np.zeros((nTrain,), dtype=np.int32)
        for i in xrange(nTrain):
            words, self.train_y[i] = trainset[i]
            self.trainX[i, :] = getSentenceFeatures(tokens, wordVectors, words)
        
        # Prepare dev set features
        devset = dataset.getDevSentences()
        nDev = len(devset)
        self.devX = np.zeros((nDev, dimVectors))
        self.dev_y = np.zeros((nDev,), dtype=np.int32)
        for i in xrange(nDev):
            words, self.dev_y[i] = devset[i]
            self.devX[i, :] = getSentenceFeatures(tokens, wordVectors, words)
        
        # Prepare test set features
        testset = dataset.getTestSentences()
        nTest = len(testset)
        self.testX = np.zeros((nTest, dimVectors))
        self.test_y = np.zeros((nTest,), dtype=np.int32)
        for i in xrange(nTest):
            words, self.test_y[i] = testset[i]
            self.testX[i, :] = getSentenceFeatures(tokens, wordVectors, words)

class SKLearnDataLoader(DataLoader):
    def __init__(self, loader):
        X,y   = loader(True)
        self.devAndTrainX,self.testX,self.devAndTrain_y,self.test_y  = train_test_split(X,y,test_size=0.2,random_state=RANDOM_SEED)
        self.trainX,self.devX,self.train_y,self.dev_y = train_test_split(self.devAndTrainX,self.devAndTrain_y,test_size=.2, random_state=RANDOM_SEED)
        
        
class MnistDataLoader(DataLoader):
    def __init__(self):
        
        from tensorflow.examples.tutorials.mnist import input_data
        self._dataSet = input_data.read_data_sets("MNIST_data/", one_hot=True)
        self.testX = self._dataSet.test.images
        self.test_y = self._dataSet.test.labels
        
    def iterTraining(self, BATCH_SIZE):
        return lambda: self._dataSet.train.next_batch(BATCH_SIZE)
    
    def nSamples(self):
        return 0
    def nFeatures(self):
        return self._dataSet.test.images.shape[1]
    def nClasses(self):
        return self._dataSet.test.labels.shape[1]
            
    def encodeOneHot(self, trainOnly=True):
        pass
    def decodeOneHot(self, trainOnly=True):
        raise NotImplementedError
        
        
class TextVectorDataLoader(DataLoader):
    """
    Loads vectors from plain text files, where each line is a vector:
    
    0 1 0 -1 -1 -1 -1 10 10 10 0 1 1 10 0 1123 10 0 1 0 10 10 10 0
    
    All vectors are assumed to have equal dimensions. The last dimension 
    of the vector is assumed to be the golden label.
    """
    
    def __init__(self, root,nFiles = 0, nClasses=5):
        """
        @praam root: All files in this directory will be parsed.
        @param nFiles: Number of files in root that will be sampled. If 0 (default),
        then all files are taken. 
        @param nClasses: If greater than 0, then the golden labels
        will be partitioned into the given number of discrete classes. 
        """
        assert nClasses>=0
        
        paths = os.listdir(root)
        
        if nFiles:
            paths = sample(paths, nFiles)
            
        # Load data from all fils in dir
        Xy=pd.DataFrame.from_csv(join(root,paths[0]),sep=" ")
        for path in paths[1:]:
            Xy = np.vstack((Xy,pd.DataFrame.from_csv(join(root,path),sep=" ")))
        self._length = Xy.shape[0]
        
        # Split into features and label
        X,_,y = np.hsplit(Xy,[Xy.shape[1]-1,-1])
        
        # Turn into discrete classes
        if nClasses:
            y=doublesToClasses(y,nClasses)
            
        # Split into train, dev, test
        self.devAndTrainX,self.testX,self.devAndTrain_y,self.test_y  = train_test_split(X,y,test_size=.2)
        self.trainX,self.devX,self.train_y,self.dev_y = train_test_split(self.devAndTrainX,self.devAndTrain_y,test_size=.2)
        

     

