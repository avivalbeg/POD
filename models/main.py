"""
@todo: 
    - Save models
    - Progress bar
    - Hidden layers as param
    - More models
    - More params for non-ann
    - say which code was salvaged from CS224N
    - write demo script
    - clean imports
    - k-fold cross validation
    - validation for ANNs
    - Save models
"""

from eval_tools import *


def main(args):

#     REG_VALS = REG_VALS[:1]
#     LRS = LRS[:1]
    N_EPOCHS = 100
#     data = RandomDataLoader()
#     data = TestDataLoader()
#     data = StanfordSentimentTreebankDataLoader()

    # data = SKLearnDataLoader(datasets.load_boston)
#     data = SKLearnDataLoader(datasets.load_diabetes)
    data = SKLearnDataLoader(datasets.load_iris)
#     data = SKLearnDataLoader(datasets.load_breast_cancer)
#     data = MnistDataLoader()
    
    # Choose all possible configurations
#     print list(data.trainX)
#     print list(data.train_y)
#     print list(data.testX)
#     print list(data.test_y)

    
    configs = [Config(data, reg, BATCH_SIZE, N_EPOCHS, lr)\
               for reg, lr in itertools.product(REG_VALS, LRS)]
    
    # compareModels((
    #           LinearRegressionModel,
    #           PolynomialRegressionModel,
    #           ), 
    #   configs, data)
    
    
#     compareModels((
#                     SVMModel,
#                     LogisticRegressionModel,
#                     KMeansModel,
#                     KNeighborsModel,
#                    ),
#         configs[:1], data,
#         debug=False)
#     quit()

    compareModels((
                    SoftmaxANN,
                    DoubleLayerSoftmaxANN,
                   ),
        configs, 
        debug=True)
     
      
if __name__ == "__main__":
    main(sys.argv)
