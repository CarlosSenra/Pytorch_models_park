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
                           cell_type = 'GRU',
                           path_pred = config.PATH_PREDICTIONS_GRU,
                           path_metrics_val = config.PATH_METRICS_VALUES_GRU,
                           max_prediction_length = 20,
                           max_encoder_length = 20,
                           batch_size = 128,
                           patience=1)
        
    
    functions.cleaning_eval_metrics_results(config.PATH_METRICS_VALUES_GRU,
                                            config.PATH_RESULTS_GRU, 
                                            "GRU")
