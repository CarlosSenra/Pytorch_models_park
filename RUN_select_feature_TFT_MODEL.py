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

encoder_select = ['month',
                'dayofweek_num',
                'hour',
                'bool_weather_missing_values',
                'precipType',
                'holiday']
decoder_select = ['month',
                'dayofweek_num',
                'hour',
                'bool_weather_missing_values',
                'precipType',
                'holiday']

encoder = ['month',
            'dayofweek_num',
            'hour',
            'holiday']
decoder = ['month',
            'dayofweek_num',
            'hour',
            'holiday']


i = 0

if __name__ == "__main__":
    select_fatures = True
    for csv in list_csv[:1]:
        print(i)
        i+=1       
        print(csv)
        df = functions.get_csv(path,csv)
        if select_fatures:
            print(encoder_select)
            util.run_TFT_model(df = df,
                           csv_file_name=csv,
                           encoder_list=encoder_select,
                           decoder_list=decoder_select,
                           max_prediction_length = 5,#168,
                           max_encoder_length = 5, #len(df) - 168,
                           batch_size = 128,
                           patience=2,
                           select_fatures = select_fatures)
        else:
            print(encoder)
            util.run_TFT_model(df = df,
                            csv_file_name=csv,
                            encoder_list=encoder,
                            decoder_list=decoder,
                            max_prediction_length = 5,#168,
                            max_encoder_length = 5, #len(df) - 168,
                            batch_size = 128,
                            patience=2,
                            select_fatures = select_fatures)
    if False:    
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
    
     