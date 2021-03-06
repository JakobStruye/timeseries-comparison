import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2' #Suppress build warnings
from run_gru_tf import GruSettings, run_gru

settings = GruSettings()
settings.max_verbosity = 2
settings.epochs = 20
settings.online = False
nodes = 300
lr = 0.003
lookback = 60
settings.nodes = nodes
settings.batch_size = 1
settings.lr = lr
settings.lookback = lookback
settings.predictionStep = 5
settings.season = 24
settings.retrain_interval = 4000
settings.nTrain = 5000
settings.normalization_type = 'AN'
settings.implementation = 'keras'
settings.rnn_type = 'lstm'
settings.use_binary = False
settings.limit_to = None
settings.finalize()
mase = run_gru(settings)

print mase
