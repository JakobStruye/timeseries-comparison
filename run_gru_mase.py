import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2' #Suppress build warnings
from run_gru import GruSettings, run_gru
from sys import argv
import random

settings = GruSettings()
settings.max_verbosity = 0
settings.epochs = int(argv[7])
settings.online = True
nodes = int(argv[3])
retrain_interval = int(argv[4])
lr = float(argv[5])
lookback = int(argv[6])
settings.nodes = nodes
settings.batch_size = None
settings.retrain_interval = retrain_interval
settings.lr = lr
settings.lookback = lookback
settings.predictionStep = 60
settings.season = 1440
settings.finalize()
mase, closer_rate = run_gru(settings)

print str(mase) + " " + str(closer_rate)
