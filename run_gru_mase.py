import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2' #Suppress build warnings
from run_gru import GruSettings, run_gru
from sys import argv
import random

settings = GruSettings()
settings.max_verbosity = 0
settings.epochs = 5
settings.online = False
nodes = int(argv[3])
batch = int(argv[4])
lr = float(argv[5])
lookback = int(argv[6])
settings.nodes = nodes
settings.batch_size = batch
settings.lr = lr
settings.lookback = lookback
settings.finalize()
mase = run_gru(settings)

print mase