import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2' #Suppress build warnings
from run_rnn import GruSettings, run_gru
from sys import argv
import random

"""
Command line args
(0: filename)
(1: -d)
(2: dataset name)
3: nodes
4: retrain
5: lr
6: lookback
7: epochs
8: online
9: batch
10: lb_as_features
11: feature_count
12: implementation 
13: adam epsilon
"""
settings = GruSettings()
settings.max_verbosity = 0
settings.epochs = int(argv[7])
settings.online = True if argv[8] == "True" else False
try:
    settings.lookback = int(argv[6])
except:
    settings.lookback = None
settings.nodes = int(argv[3])
try:
    settings.batch_size = int(argv[9])
except:
    settings.batch_size = None
settings.retrain_interval = int(argv[4])
settings.lookback_as_features =  True if argv[10] == "True" else False
settings.feature_count = int(argv[11])
settings.lr = float(argv[5])
settings.predictionStep = 5
settings.season = 48
settings.ignore_for_error = [5500]
settings.nTrain = 5000
settings.limit_to = 7500# if not
settings.normalization_type = 'default'
settings.implementation = 'keras'
settings.rnn_type = argv[12]
settings.use_binary = False
settings.stateful = False
settings.adam_eps = float(argv[13])
settings.refeed_on_retrain = True
settings.reset_on_retrain = False
settings.use_dropout = False

settings.finalize()
mase = run_gru(settings)

print str(mase)
