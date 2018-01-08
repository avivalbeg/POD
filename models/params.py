import numpy as np

RANDOM_SEED = 1

# Config params

BATCH_SIZE = 64
N_EPOCHS = 10000

N_HYPERPARAMS = 5 # Number of values to be tried for each hyperparam

REG_VALS = sorted(np.logspace(-4, 2, num=N_HYPERPARAMS, base=10))
LRS = sorted(np.logspace(-4, -.6, num=N_HYPERPARAMS, base=10))

# Random dataset params

N_RAND_SAMPLES = 3000
N_RAND_FEATURES = 100
N_RAND_CLASSES = 5