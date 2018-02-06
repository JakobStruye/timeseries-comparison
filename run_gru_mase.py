import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2' #Suppress build warnings
from run_gru_tf import GruSettings, run_gru
from sys import argv
import random

settings = GruSettings()
settings.max_verbosity = 0
settings.epochs = int(argv[7])
settings.online = False
nodes = int(argv[3])
retrain_interval = int(argv[4])
lr = float(argv[5])
lookback = int(argv[6])
settings.nodes = nodes
settings.batch_size = None
settings.retrain_interval = retrain_interval
settings.lr = lr
settings.lookback = lookback
settings.predictionStep = 5
settings.season = 48
#settings.ignore_for_error = 5500
settings.nTrain = 3000
settings.ignore_for_error = 3005
settings.limit_to = 5000
settings.normalization_type = 'default'
settings.implementation = 'keras'
settings.rnn_type = 'lstm'
settings.use_binary = False


settings.finalize()
mase = run_gru(settings)

print str(mase)
