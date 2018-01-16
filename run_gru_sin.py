import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2' #Suppress build warnings
from run_gru import GruSettings, run_gru
from sys import argv
import random

settings = GruSettings()
settings.max_verbosity = 2
settings.epochs = 100
settings.online = False
nodes = 250
batch = 64
lr = 0.001
lookback = 1
settings.nodes = nodes
settings.batch_size = batch
settings.lr = lr
settings.lookback = lookback
settings.predictionStep = 50
settings.season = 629
settings.finalize()
mase = run_gru(settings)

print mase