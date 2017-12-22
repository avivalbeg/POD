

import os

from constants import *
from itertools import product
import sys
from mining.DataMiner import IrcDataMiner
from mining.get_data import dlPokerBotData, mineGameData
 

runBenchmark = False
runAnn = True

def trainEquityPrediction():
    """
    Runs the machine learning models from the ml directory.
    Notice that there is also a train.py script in the bot directory, but that script is different because it trains the bot to make actual decisions. This script on the other hand just creates statistical models.
    
    
    Best results are achieved with a logistic regression model:
    
    'LogisticRegressionModel': (79.014084507042256, # Test accuracy
                                 reg:0.1-nSamples:2272-nFeats:22-nClasses:5-batchSize:64-nEpochs:10000-lr:0),
    
    Best results I got with a neural network:
    
    {'DoubleLayerSoftmaxANN': (0.6915493, # Test accuracy
                               reg:3.16227766017-nSamples:2272-nFeats:22-nClasses:5-batchSize:64-nEpochs:100000-lr:0.00501187233627)}
    
    
    
    """ 
    
    from models.DataLoader import TextVectorDataLoader, SKLearnDataLoader,\
    RandomDataLoader
    from models.eval_tools import Config, compareModels
    import itertools
    from models.params import REG_VALS, LRS, BATCH_SIZE, N_EPOCHS
    from models.Model import SVMModel, LogisticRegressionModel, KMeansModel, \
        KNeighborsModel, SoftmaxANN, DoubleLayerSoftmaxANN, TripleLayerSoftmaxANN
    from sklearn import datasets
    
    data = TextVectorDataLoader(HAND_DATA_PATH, nClasses=5)

    print("Training with %d examples" % len(data))
    if runBenchmark:
        
        configs = [Config(data, reg, BATCH_SIZE, N_EPOCHS, 0) \
                   for reg in REG_VALS]
        compareModels((
                        SVMModel,
                        LogisticRegressionModel,
                        KMeansModel,
                        KNeighborsModel,
                       ),
            configs,
            debug=False)
    
    if runAnn:
        configs = [Config(data, reg, BATCH_SIZE, N_EPOCHS, lr) \
                   for reg, lr in product(REG_VALS, LRS)]
        
        compareModels((
#             SoftmaxANN,
#             DoubleLayerSoftmaxANN,
            TripleLayerSoftmaxANN,
                       ),
            configs,
            debug=True)
    
def main(args):
    
    # Watchuwanna do?
    #getIrcData()
    dlPokerBotData()
#     mineGameData()
    #mineHandData()
    #trainEquityPrediction()
    
    pass

if __name__ == '__main__':
    main(sys.argv)
