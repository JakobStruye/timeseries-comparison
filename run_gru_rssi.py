import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2' #Suppress build warnings
from run_gru import GruSettings, run_gru

settings = GruSettings()
settings.max_verbosity = 2
settings.epochs = 100
settings.online = True
nodes = 60
lr = 0.002
lookback = 50
settings.nodes = nodes
settings.batch_size = None
settings.lr = lr
settings.lookback = lookback
settings.predictionStep = 50
settings.season = 1440
settings.retrain_interval = 500
settings.finalize()
mase, closer_rate = run_gru(settings)

print mase