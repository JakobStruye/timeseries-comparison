# Description



# Example Usage

run modified HTM tests 

    # configure model_params/nyc_taxi_model_params.yaml:
    #   set spEnable, tmEnable
    # configure run_tm_model.py
    #   set nMultiplePass, call either runMultiplePass or runMultiplePassSPonly
	python run_tm_model.py -d nyc_taxi 

run LSTM or GRU on dataset

    # configure run_gru_nyc.py
    # python run_gru_nyc.py -d nyc_taxi


Tuning TODO DOCUMENT