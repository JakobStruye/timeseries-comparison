import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2' #Suppress build warnings
from run_gru_tf import GruSettings, run_gru

settings = GruSettings()
settings.max_verbosity = 2
settings.epochs = 50
settings.online = True
nodes = 100
lr = 0.001
lookback = 50
settings.nodes = nodes
settings.batch_size = None
settings.lr = lr
settings.lookback = lookback
settings.predictionStep = 5
settings.season = 24
settings.retrain_interval = 1500
settings.normalization_type = 'default'
settings.limit_to = None
settings.finalize()
mase = run_gru(settings)

print mase
