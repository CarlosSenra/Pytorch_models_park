import pandas as pd
import os
import matplotlib.pyplot as plt 
import numpy as np
from pytorch_forecasting.metrics import MAE, SMAPE, MAPE, RMSE


def get_csv_file_list(path):
    list_csv = os.listdir(path)
    return list_csv

def get_csv(path,csv_file_name):
    df = pd.read_csv(path + "\\" + csv_file_name)
    #estado.drop(columns='Unnamed: 0',inplace=True)
    #estado.data = pd.to_datetime(estado.data)
    #estado = estado.loc[estado.data >= '2022-01-01']

    df['temperature'].fillna(method='ffill', inplace=True)
    df['windSpeed'].fillna(method='ffill', inplace=True)
    df['year'] = df['year'].astype(str)
    df['hour'] = df['hour'].astype(str)
    df['month'] = df['month'].astype(str)
    df['day'] = df['day'].astype(str)
    df['dayofweek_num'] = df['dayofweek_num'].astype(str)
    df['house_hold'] = df['house_hold'].astype(str)
    df['precipType'] = df['precipType'].astype(str)
    df['icon'] = df['icon'].astype(str)
    df['summary'] = df['summary'].astype(str)
    df['bool_weather_missing_values'] = df['bool_weather_missing_values'].astype(str)

    return df 

def evaluation_metrics(val_dataloader,best_model,householde_name,execution_time):    
    predictions = best_model.predict(val_dataloader, return_y=True, trainer_kwargs=dict(accelerator="cpu"))
    mae = MAE()(predictions.output, predictions.y)
    mape = MAPE()(predictions.output, predictions.y)
    smape = SMAPE()(predictions.output, predictions.y)
    rmse = RMSE()(predictions.output, predictions.y)

    dict_ = {'House_Hold':householde_name.split(".")[0].upper(),
            'MAE':[mae.to('cpu').numpy().round(3)],
            'MAPE':[mape.to('cpu').numpy().round(3)],
            'SMAPE':[smape.to('cpu').numpy().round(3)],
            'RMSE':[rmse.to('cpu').numpy().round(3)],
            'Time_execution':f"{execution_time:.2f}s"}

    return dict_


def get_attention_values(interpretation,householde_name):
  att_encoder_values = interpretation['encoder_variables'].to('cpu').numpy()
  att_decoder_values = interpretation['decoder_variables'].to('cpu').numpy()
  tft_encoder = ['year', 'month', 'day', 'dayofweek_num', 'hour', 'bool_weather_missing_values', 'precipType', 'icon', 'summary','time_idx', 'relative_time_idx', 'Energy_kwh', 'temperature', 'windSpeed']
  tft_decoder = ['year', 'month', 'day', 'dayofweek_num', 'hour', 'bool_weather_missing_values', 'time_idx', 'relative_time_idx']
  encoder_dict = {}
  decoder_dict = {}
  for value, name in zip(att_encoder_values,tft_encoder):   
    encoder_dict[name] = [f"{np.round(value,4)*100:.2f}%"]
  encoder_dict['House_Hold'] = [householde_name.split(".")[0].upper()]
  for value, name in zip(att_decoder_values,tft_decoder):   
    decoder_dict[name] = [f"{np.round(value,4)*100:.2f}%"]
  decoder_dict['House_Hold'] = [householde_name.split(".")[0].upper()]
  return encoder_dict,decoder_dict


def cleaning_eval_metrics_results(path_origin, path_destiny,model_name):
    list_csv = os.listdir(path_origin)
    df_list = []
    for csv in list_csv:
        df_list.append(pd.read_csv(path_origin + "\\" + csv))
    concat_df = pd.concat(df_list)
    concat_df.to_csv(path_destiny + "\\" + f"{model_name}_metrics_results.csv",index=False)


def cleaning_attention_results(path_origin, path_destiny):
    list_csv = os.listdir(path_origin)
    df_encoder_list = []
    df_decoder_list = []
    for csv in list_csv:
        if csv.split("_")[0] == 'decoder':
            df_encoder_list.append(pd.read_csv(path_origin + "\\" + csv))
        else:
            df_decoder_list.append(pd.read_csv(path_origin + "\\" + csv))
    concat_df_encoder = pd.concat(df_encoder_list)
    concat_df_decoder = pd.concat(df_decoder_list)
    concat_df_encoder.to_csv(path_destiny + "\\" + "encoder_attention_results.csv",index=False)
    concat_df_decoder.to_csv(path_destiny + "\\" + "decoder_attention_results.csv",index=False)








####### DANGER !!!!!
#beta.version
#######
def save_graph_image(predictions,df_principal,figsize= (14, 4)):
    plt.figure(figsize=figsize)

    plt.plot(df_principal['time_idx'], df_principal['Energy_kwh'],color = 'blue')
    plt.plot(predictions.x['decoder_time_idx'][0].to('cpu').numpy(), predictions.x['decoder_target'][0].to('cpu').numpy().round(3),color = 'blue',label='Real')
    plt.plot(predictions.x['decoder_time_idx'][0].to('cpu').numpy(), predictions.output[0][0].to('cpu').numpy().round(3).squeeze(),color = 'orange',label='Predict')
    plt.xlabel("Time indexs")
    plt.ylabel("Energy KW/h")
    plt.title("Line Plot with Markers")
    plt.grid(True)
    plt.legend()
    plt.savefig('energy_prediction_plot.png', dpi=300, bbox_inches='tight')
#######
#beta.version
#######


