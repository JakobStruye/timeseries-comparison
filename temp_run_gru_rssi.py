import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2' #Suppress build warnings
from run_gru import GruSettings, run_gru

settings = GruSettings()
settings.max_verbosity = 2
settings.epochs = 85
settings.online = True
nodes = 53
lr = 0.00077
lookback = 52
settings.nodes = nodes
settings.batch_size = None
settings.lr = lr
settings.lookback = lookback
settings.predictionStep = 60
settings.season = 1440
settings.retrain_interval = 1180
settings.finalize()
mase, closer_rate = run_gru(settings)

print mase

