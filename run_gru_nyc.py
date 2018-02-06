import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2' #Suppress build warnings
from run_gru_tf import GruSettings, run_gru

settings = GruSettings()
settings.max_verbosity = 2
settings.epochs = 1000
settings.online = False
nodes = 100
lr = 0.0077
lookback = 50
settings.nodes = nodes
settings.batch_size = None
settings.lr = lr
settings.lookback = lookback
settings.predictionStep = 5
settings.season = 48
settings.retrain_interval = 2500
#settings.nTrain = 5000
settings.ignore_for_error = 5500
settings.normalization_type = 'default'
settings.implementation = 'tf'
settings.rnn_type = 'lstm'
settings.use_binary = True
settings.limit_to = None
settings.finalize()
mase = run_gru(settings)

print mase
