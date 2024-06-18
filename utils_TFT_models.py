import pandas as pd
import os
import warnings
import time
import config
import util_functions
warnings.filterwarnings("ignore")

import copy
from pathlib import Path
import warnings

import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from pytorch_forecasting import Baseline, TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer, NaNLabelEncoder
from pytorch_forecasting.metrics import MAE, SMAPE, QuantileLoss, MAPE, RMSE

import pickle
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters

#torch.set_float32_matmul_precision("high")


##########################################################################################
                                       #TFT
##########################################################################################
def run_TFT_model(df,csv_file_name,
                  learning_rate = .1,
                  hidden_size = 15,
                  dropout = .2,
                  attention_head_size = 4,
                  hidden_continuous_size = 15,
                  loss = MAE(),
                  optimizer = "Ranger",
                  lstm_layers  = 2,
                  max_prediction_length = 168,
                  max_encoder_length = 720,
                  batch_size = 128,
                  patience=1,
                  path_pred = config.PATH_PREDICTIONS_TFT,
                  path_metrics_val = config.PATH_METRICS_VALUES_TFT,
                  seed = 81,
                  select_fatures =None):
    training_cutoff = df["time_idx"].max() - max_prediction_length

    pl.seed_everything(seed)

    training = TimeSeriesDataSet(
                    df[lambda x: df.time_idx <= training_cutoff],
                    time_idx = 'time_idx',
                    target = 'Energy_kwh',
                    group_ids = ['house_hold'],
                    time_varying_known_reals=['time_idx',
                                            'temperature',
                                            'windSpeed'],
                    time_varying_unknown_reals = ['Energy_kwh'],
                                                #'temperature',
                                                #'windSpeed'],
                    static_categoricals=['house_hold'],
                    time_varying_known_categoricals = ['year',
                                                    'month',
                                                    'day',
                                                    'dayofweek_num',
                                                    'hour',
                                                    'bool_weather_missing_values',
                                                    'precipType',
                                                    'icon',
                                                    'summary'],
                    #time_varying_unknown_categoricals= ['precipType',
                    #                                   'icon',
                    #                                   'summary'],
                    min_encoder_length = max_encoder_length // 2,
                    max_encoder_length = max_encoder_length,
                    min_prediction_length=1,
                    max_prediction_length = max_prediction_length,
                    categorical_encoders = {'house_hold': NaNLabelEncoder(add_nan=True, warn=True),
                                            'precipType': NaNLabelEncoder(add_nan=True, warn=True),
                                            'icon': NaNLabelEncoder(add_nan=True, warn=True),
                                            'summary': NaNLabelEncoder(add_nan=True, warn=True)},
                    target_normalizer=None,          
                    add_relative_time_idx=True,
                    add_target_scales=True,
                    add_encoder_length=True
    )


    validation = TimeSeriesDataSet.from_dataset(training, 
                                                df,
                                                predict = True,
                                                stop_randomization = True)



    train_dataloader = training.to_dataloader(train = True,
                                            batch_size = batch_size,
                                            num_workers = 1)


    val_dataloader = validation.to_dataloader(train = False,
                                            batch_size = batch_size,
                                            num_workers = 1)

    
    tft = TemporalFusionTransformer.from_dataset(training,
                                                learning_rate = learning_rate,
                                                hidden_size = hidden_size,
                                                attention_head_size = attention_head_size,
                                                hidden_continuous_size = hidden_continuous_size,
                                                dropout = dropout,
                                                loss = loss,#QuantileLoss(quantiles =[0.05, 0.5, 0.95]),
                                                optimizer = optimizer,
                                                lstm_layers = lstm_layers
                                            )
    
    early_stop_callback = EarlyStopping(monitor = "val_loss",
                                    min_delta = 0.00001,
                                    patience = patience,
                                    verbose = True,
                                    mode = "min")


    lr_logger = LearningRateMonitor()
    logger_GRU = TensorBoardLogger("TFT_logs")

    checkpoint_callback = ModelCheckpoint(monitor="val_loss")

    trainer_TFT = pl.Trainer(
                            max_epochs = 350,
                            accelerator = 'gpu',
                            enable_model_summary = True,
                            limit_train_batches = 300,
                            gradient_clip_val = 0.1,
                            callbacks = [lr_logger, early_stop_callback, checkpoint_callback],
                            logger = logger_GRU,
                            enable_progress_bar=False
            )
    start_time = time.time()
    trainer_TFT.fit(
                    tft,
                    train_dataloaders = train_dataloader,
                    val_dataloaders = val_dataloader
    )
    end_time = time.time()  
    execution_time = end_time - start_time  

    best_model_path = trainer_TFT.checkpoint_callback.best_model_path
    best_TFT = TemporalFusionTransformer.load_from_checkpoint(best_model_path)

    best_model_path = str(best_model_path)
    best_TFT = TemporalFusionTransformer.load_from_checkpoint(best_model_path)

    predictions = best_TFT.predict(val_dataloader, mode = "raw", return_x = True)
    raw_predictions = best_TFT.predict(val_dataloader, mode="raw", return_x=True)
    interpretation = best_TFT.interpret_output(raw_predictions.output, reduction="sum")

    if select_fatures:
        attention_path = config.PATH_ATTENTION_SELECT_FEATURES
        path_pred = config.PATH_PREDICTIONS_TFT_SELECT_FEATURES
        path_metrics_val = config.PATH_METRICS_VALUES_TFT_SELECT_FEATURES

        df_predictions = pd.DataFrame({'time_idx':predictions.x['decoder_time_idx'][0].to('cpu').numpy(),
                                    'Real':predictions.x['decoder_target'][0].to('cpu').numpy().round(3),
                                    'predict':predictions.output[0][0].to('cpu').numpy().round(3).squeeze()})

        
        df_predictions.to_csv(path_pred + "\\" + "predict_" + csv_file_name, index=False)

        eval_dict = util_functions.evaluation_metrics(val_dataloader,best_TFT,csv_file_name,execution_time)
        df_eval_metrics = pd.DataFrame(eval_dict)
        df_eval_metrics.to_csv(path_metrics_val + "\\" + "eval_metrics_" + csv_file_name, index=False)

        attention_encoder,  attention_decoder= util_functions.get_attention_values(interpretation,csv_file_name)
        attention_encoder_df = pd.DataFrame(attention_encoder)
        attention_decoder_df = pd.DataFrame(attention_decoder)
        attention_encoder_df.to_csv(attention_path + "\\" + "encoder_attention" + csv_file_name, index=False)
        attention_decoder_df.to_csv(attention_path + "\\" + "decoder_attention" + csv_file_name, index=False)
        
    else:    
        attention_path = config.PATH_ATTENTION
        df_predictions = pd.DataFrame({'time_idx':predictions.x['decoder_time_idx'][0].to('cpu').numpy(),
                                    'Real':predictions.x['decoder_target'][0].to('cpu').numpy().round(3),
                                    'predict':predictions.output[0][0].to('cpu').numpy().round(3).squeeze()})

        
        df_predictions.to_csv(path_pred + "\\" + "predict_" + csv_file_name, index=False)

        eval_dict = util_functions.evaluation_metrics(val_dataloader,best_TFT,csv_file_name,execution_time)
        df_eval_metrics = pd.DataFrame(eval_dict)
        df_eval_metrics.to_csv(path_metrics_val + "\\" + "eval_metrics_" + csv_file_name, index=False)

        attention_encoder,  attention_decoder= util_functions.get_attention_values(interpretation,csv_file_name)
        attention_encoder_df = pd.DataFrame(attention_encoder)
        attention_decoder_df = pd.DataFrame(attention_decoder)
        attention_encoder_df.to_csv(attention_path + "\\" + "encoder_attention" + csv_file_name, index=False)
        attention_decoder_df.to_csv(attention_path + "\\" + "decoder_attention" + csv_file_name, index=False)


    
##########################################################################################
##########################################################################################

