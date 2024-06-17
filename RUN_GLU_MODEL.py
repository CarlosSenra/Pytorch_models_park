import pandas as pd
import os
import warnings
import time
import config
import utils_models as util
warnings.filterwarnings("ignore")

path = config.PATH_DATASETS
list_csv = util.get_csv_file_list(path)


if __name__ == "__main__":
    for csv in list_csv:
        df = util.get_csv(path,csv)
        util.run_GLU_model(df = df,
                           csv_file_name=csv,
                           max_prediction_length = 168,
                           max_encoder_length = 720,
                           batch_size = 128,
                           path_pred = config.PATH_PREDICTIONS,
                           seed = 81)