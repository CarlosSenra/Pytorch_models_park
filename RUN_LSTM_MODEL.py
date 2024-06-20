import pandas as pd
import os
import warnings
import time
import config
import util_functions as functions
import utils_RNN_models as util
warnings.filterwarnings("ignore")

path = config.PATH_DATASETS
list_csv = functions.get_csv_file_list(path)


if __name__ == "__main__":
    for csv in list_csv:
        df = functions.get_csv(path,csv)
        util.run_RNN_model(df = df,
                           csv_file_name=csv,
                           cell_type = 'LSTM',
                           path_pred = config.PATH_PREDICTIONS_LSTM,
                           path_metrics_val = config.PATH_METRICS_VALUES_LSTM,
                           max_prediction_length = 168,
                           max_encoder_length = len(df) - 168,
                           batch_size = 128,
                           patience=1)
        
    
    functions.cleaning_eval_metrics_results(config.PATH_METRICS_VALUES_LSTM,
                                            config.PATH_RESULTS_LSTM, 
                                            "LSTM")
