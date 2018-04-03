import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2' #Suppress build warnings
import sys
import shared

sys.path.append(".") #..welp
from run_rnn import  GruSettings, run_gru


settings = GruSettings()
settings.max_verbosity = 2
settings.epochs = 300
settings.epoch_retrain = 100
settings.online = True
nodes = 130
lr = 0.02#11389451319
settings.nodes = nodes
settings.batch_size = 1
settings.lr = lr
settings.loss = 'mae'
settings.stateful=False
settings.lookback = None
settings.lookback_as_features=False
settings.feature_count = 3
settings.predictionStep = 5
settings.season = 48
settings.adam_eps = 0.001
settings.retrain_interval = 1000
settings.reset_on_retrain = False
settings.refeed_on_retrain = True
settings.cutoff_normalize = True
settings.use_dropout = False
settings.nTrain = 5000
settings.ignore_for_error = [5500,10000]
settings.normalization_type = 'default'
settings.implementation = 'keras'
settings.rnn_type = 'lstm'
settings.use_binary = False
settings.limit_to = None
settings.finalize()
mase = run_gru(settings)

print mase
