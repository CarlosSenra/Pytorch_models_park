import pandas as pd
import os
import warnings
import time
import config
import util_functions as functions
import utils_TFT_models as util
warnings.filterwarnings("ignore")

path = config.PATH_DATASETS
list_csv = functions.get_csv_file_list(path)


if __name__ == "__main__":
    select_fatures = None

    for csv in list_csv:
        df = functions.get_csv(path,csv)
        util.run_TFT_model(df = df,
                           csv_file_name=csv,
                           max_prediction_length = 168,
                           max_encoder_length = len(df) - 168,
                           batch_size = 128,
                           patience=1,
                           select_fatures = select_fatures)
        
    if select_fatures:
        functions.cleaning_eval_metrics_results(config.PATH_METRICS_VALUES_TFT_SELECT_FEATURES,
                                                config.PATH_RESULTS_TFT_SELECT_FEATURES, 
                                                "TFT")

        functions.cleaning_attention_results(config.PATH_ATTENTION_SELECT_FEATURES,
                                            config.PATH_RESULTS_TFT_SELECT_FEATURES)
    else:
        print("Não está ocorrendo seleção de variáveis")
        functions.cleaning_eval_metrics_results(config.PATH_METRICS_VALUES_TFT,
                                                config.PATH_RESULTS_TFT, 
                                                "TFT")

        functions.cleaning_attention_results(config.PATH_ATTENTION,
                                            config.PATH_RESULTS_TFT)        