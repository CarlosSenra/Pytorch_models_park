import pandas as pd
import numpy as np
import random
import os
import functions_data_transformation as functions


weather_hourly = pd.read_csv('weather_hourly_darksky.csv')
weather_hourly.time = pd.to_datetime(weather_hourly.time)
weather = weather_hourly[['time','temperature','windSpeed','precipType','icon','summary']]

block_numbers_list = functions.get_random_blocks(num_blocks = 5, seed=42)

if __name__ == '__main__':
    print(os.getcwd())
    for num in block_numbers_list:
        df, house_select = functions.get_random_household(block_numer = num , seed = 42)
        df_block = functions.transform_half_in_hourly(df)
        df_block_weather = functions.add_weater_data(df_block,weather)
        df_block_weather = functions.add_bool_weather_missing_values(df_block_weather)

        blocks, dict_labels = functions.feature_eng_function(df_block_weather)
        blocks = blocks.reset_index(drop=True).reset_index()
        blocks = blocks.rename(columns={'index':'time_idx'})
        blocks = blocks.drop(columns='time')

        functions.save_json(dict_labels,house_select)
        blocks.to_csv(f'dataframe_model\{house_select}.csv',index=False)


