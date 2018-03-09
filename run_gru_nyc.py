import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2' #Suppress build warnings
from run_gru_tf_stateful import GruSettings, run_gru

settings = GruSettings()
settings.max_verbosity = 2
settings.epochs = 10
settings.online = False
nodes = 20 #317
lr = 0.005#0.0031802801373# 0.0077
settings.nodes = nodes
settings.batch_size = 64
settings.lr = lr
settings.loss = 'mae'
settings.stateful=False
settings.lookback = 75  #None to not use lookbacks
settings.lookback_as_features=True #Considers each lookback value a separate feature if True, ignored if lookback is None
settings.feature_count = 3 #Uses the first `feature_count` selected columns as features, ignored if `lookback_as_features`
settings.predictionStep = 5
settings.season = 48
settings.adam_eps = 0.001
settings.retrain_interval = 2500
settings.nTrain = 5000
settings.ignore_for_error = 5500
settings.normalization_type = 'default'
settings.implementation = 'keras'
settings.rnn_type = 'gru'
settings.use_binary = False
settings.limit_to = None
settings.finalize()
mase = run_gru(settings)

print mase
