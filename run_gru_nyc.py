import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2' #Suppress build warnings
from run_gru_tf import GruSettings, run_gru

settings = GruSettings()
settings.max_verbosity = 2
settings.epochs = 5
settings.online = False
nodes = 20 #317
lr = 0.003#0.0031802801373# 0.0077
lookback = 75#74
settings.nodes = nodes
settings.batch_size = None
settings.lr = lr
settings.lookback = lookback
settings.predictionStep = 5
settings.season = 48
settings.retrain_interval = 2500
settings.nTrain = 5000
settings.ignore_for_error = 5500
settings.normalization_type = 'default'
settings.implementation = 'keras'
settings.rnn_type = 'lstm'
settings.use_binary = False
settings.limit_to = None
settings.finalize()
mase = run_gru(settings)

print mase
