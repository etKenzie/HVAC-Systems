import os
import random
import gc
import datetime
# import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import math

# for reduce_mem usage
from pandas.api.types import is_datetime64_any_dtype as is_datetime
from pandas.api.types import is_categorical_dtype

def prepare():
    # Read Data Files
    path = os.getcwd() + '/data/'
    train_df = pd.read_csv(os.path.join(path, 'train.csv'))
    buildings_df = pd.read_csv(os.path.join(path, 'building_metadata.csv'))
    weather_train_df = pd.read_csv(os.path.join(path, 'weather_train.csv'))

    # Reduce memory usage
    train_df = reduce_mem_usage(train_df,use_float16=True)
    buildings_df = reduce_mem_usage(buildings_df,use_float16=True)
    weather_train_df = reduce_mem_usage(weather_train_df,use_float16=True)
    weather_train_df = fill_weather_dataset(weather_train_df)

    # Merge Data Based on building_id and time
    train_df = train_df.merge(buildings_df, on='building_id', how='left')
    train_df  = train_df.merge(weather_train_df, on=['site_id', 'timestamp'], how='left')

    # Remove outliers
    train_df = train_df [train_df['building_id'] != 1099 ]
    # train_df = train_df.query('not (building_id <= 104 & meter == 0 & timestamp <= "2016-05-20")')

    # Label encoder
    le = LabelEncoder()
    train_df["primary_use"] = le.fit_transform(train_df["primary_use"])

    # Deal with missing values

    # Columns to drop
    drop_cols = ["sea_level_pressure", "wind_speed", "timestamp"]
    train_df = train_df.drop(drop_cols, axis = 1)

    # Prepare the test df
    test_df = pd.read_csv(os.path.join(path, 'test.csv'))
    row_ids = test_df["row_id"]
    test_df.drop("row_id", axis=1, inplace=True)
    test_df = reduce_mem_usage(test_df)
    
    
    weather_test_df = pd.read_csv(os.path.join(path + 'weather_test.csv'))
    weather_test_df = fill_weather_dataset(weather_test_df)
    weather_test_df = reduce_mem_usage(weather_test_df)

    test_df = test_df.merge(buildings_df,left_on='building_id',right_on='building_id',how='left')
    test_df  = test_df.merge(weather_test_df, on=['site_id', 'timestamp'], how='left')
    del buildings_df, weather_test_df, weather_train_df
    gc.collect()

    # Label encoder
    le = LabelEncoder()
    test_df["primary_use"] = le.fit_transform(test_df["primary_use"])

    # Columns to drop
    drop_cols = ["sea_level_pressure", "wind_speed", "timestamp"]
    test_df = test_df.drop(drop_cols, axis = 1)

    # building_id,meter,meter_reading,site_id,primary_use,square_feet,year_built,floor_count,air_temperature,cloud_coverage,dew_temperature,precip_depth_1_hr,wind_direction
    # row_id,building_id,meter,site_id,primary_use,square_feet,year_built,floor_count,air_temperature,cloud_coverage,dew_temperature,precip_depth_1_hr,wind_direction

    return train_df, test_df, row_ids


def add_features():
    pass
    

# Original code from https://www.kaggle.com/aitude/ashrae-missing-weather-data-handling by @aitude

def fill_weather_dataset(weather_df):
    
    # Find Missing Dates
    time_format = "%Y-%m-%d %H:%M:%S"
    min_timestamp = pd.to_datetime(weather_df['timestamp']).min()
    timestamp_str = min_timestamp.strftime(time_format)
    start_date = datetime.datetime.strptime(timestamp_str, time_format)
    max_timestamp = pd.to_datetime(weather_df['timestamp']).max()
    timestamp_str = max_timestamp.strftime(time_format)
    end_date = datetime.datetime.strptime(timestamp_str,time_format)
    total_hours = int(((end_date - start_date).total_seconds() + 3600) / 3600)
    hours_list = [(end_date - datetime.timedelta(hours=x)).strftime(time_format) for x in range(total_hours)]
    missing_hours = []
    for site_id in range(16):
        site_hours = np.array(weather_df[weather_df['site_id'] == site_id]['timestamp'])
        new_rows = pd.DataFrame(np.setdiff1d(hours_list,site_hours),columns=['timestamp'])
        new_rows['site_id'] = site_id
        weather_df = pd.concat([weather_df,new_rows])

        weather_df = weather_df.reset_index(drop=True)           

    # Add new Features
    weather_df["datetime"] = pd.to_datetime(weather_df["timestamp"])
    weather_df["day"] = weather_df["datetime"].dt.day
    weather_df['week'] = weather_df['datetime'].dt.isocalendar().week
    weather_df["month"] = weather_df["datetime"].dt.month
    
    # Reset Index for Fast Update
    weather_df = weather_df.set_index(['site_id','day','month'])

    air_temperature_filler = pd.DataFrame(weather_df.groupby(['site_id','day','month'])['air_temperature'].mean(),columns=["air_temperature"])
    weather_df.update(air_temperature_filler,overwrite=False)

    # Step 1
    cloud_coverage_filler = weather_df.groupby(['site_id','day','month'])['cloud_coverage'].mean()
    # Step 2
    cloud_coverage_filler = pd.DataFrame(cloud_coverage_filler.fillna(method='ffill'),columns=["cloud_coverage"])

    weather_df.update(cloud_coverage_filler,overwrite=False)

    due_temperature_filler = pd.DataFrame(weather_df.groupby(['site_id','day','month'])['dew_temperature'].mean(),columns=["dew_temperature"])
    weather_df.update(due_temperature_filler,overwrite=False)

    # Step 1
    sea_level_filler = weather_df.groupby(['site_id','day','month'])['sea_level_pressure'].mean()
    # Step 2
    sea_level_filler = pd.DataFrame(sea_level_filler.fillna(method='ffill'),columns=['sea_level_pressure'])

    weather_df.update(sea_level_filler,overwrite=False)

    wind_direction_filler =  pd.DataFrame(weather_df.groupby(['site_id','day','month'])['wind_direction'].mean(),columns=['wind_direction'])
    weather_df.update(wind_direction_filler,overwrite=False)

    wind_speed_filler =  pd.DataFrame(weather_df.groupby(['site_id','day','month'])['wind_speed'].mean(),columns=['wind_speed'])
    weather_df.update(wind_speed_filler,overwrite=False)

    # Step 1
    precip_depth_filler = weather_df.groupby(['site_id','day','month'])['precip_depth_1_hr'].mean()
    # Step 2
    precip_depth_filler = pd.DataFrame(precip_depth_filler.fillna(method='ffill'),columns=['precip_depth_1_hr'])

    weather_df.update(precip_depth_filler,overwrite=False)

    weather_df = weather_df.reset_index()
    weather_df = weather_df.drop(['datetime','day','week','month'],axis=1)
        
    return weather_df


# Original code from https://www.kaggle.com/gemartin/load-data-reduce-memory-usage by @gemartin
def reduce_mem_usage(df, use_float16=False):
    """
    Iterate through all the columns of a dataframe and modify the data type to reduce memory usage.        
    """
    
    start_mem = df.memory_usage().sum() / 1024**2
    print("Memory usage of dataframe is {:.2f} MB".format(start_mem))
    
    for col in df.columns:
        if is_datetime(df[col]) or is_categorical_dtype(df[col]):
            continue
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if use_float16 and c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype("category")

    end_mem = df.memory_usage().sum() / 1024**2
    print("Memory usage after optimization is: {:.2f} MB".format(end_mem))
    print("Decreased by {:.1f}%".format(100 * (start_mem - end_mem) / start_mem))
    return df

if __name__ == "__main__":
    pass