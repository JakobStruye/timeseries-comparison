import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2' #Suppress build warnings
from run_gru_tf_stateful import GruSettings, run_gru

settings = GruSettings()
settings.max_verbosity = 2
settings.epochs = 100
settings.online = False
nodes = 20 #317
lr = 0.01#0.0031802801373# 0.0077
lookback = 1#74
settings.nodes = nodes
settings.batch_size = 1
settings.lr = lr
settings.lookback = lookback
settings.predictionStep = 5
settings.season = 1
settings.retrain_interval = 336
settings.nTrain = 5000
settings.ignore_for_error = 10000
settings.normalization_type = 'default'
settings.implementation = 'keras'
settings.rnn_type = 'lstm'
settings.use_binary = False
settings.limit_to = None
settings.finalize()
mase = run_gru(settings)

print mase
