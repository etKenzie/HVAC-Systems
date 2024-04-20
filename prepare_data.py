import os
import random
import gc

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
    test_df = pd.read_csv(os.path.join(path, 'test.csv'))
    buildings_df = pd.read_csv(os.path.join(path, 'building_metadata.csv'))
    weather_train_df = pd.read_csv(os.path.join(path, 'weather_train.csv'))

    # Reduce memory usage
    train_df = reduce_mem_usage(train_df,use_float16=True)
    buildings_df = reduce_mem_usage(buildings_df,use_float16=True)
    weather_train_df = reduce_mem_usage(weather_train_df,use_float16=True)

    # Merge Data Based on building_id and time
    train_df = train_df.merge(buildings_df, on='building_id', how='left')
    train_df  = train_df.merge(weather_train_df, on=['site_id', 'timestamp'], how='left')

    # Remove outliers
    train_df = train_df [train_df['building_id'] != 1099 ]
    # train_df = train_df.query('not (building_id <= 104 & meter == 0 & timestamp <= "2016-05-20")')

    # Add clearer variables regarding date/time
    train_df["timestamp"] = pd.to_datetime(train_df["timestamp"])
    train_df["hour"] = train_df["timestamp"].dt.hour
    train_df["day"] = train_df["timestamp"].dt.day
    train_df["weekend"] = train_df["timestamp"].dt.weekday
    train_df["month"] = train_df["timestamp"].dt.month

    # Label encoder
    le = LabelEncoder()
    train_df["primary_use"] = le.fit_transform(train_df["primary_use"])

    # Deal with missing values

    # Columns to drop
    drop_cols = ["sea_level_pressure", "wind_speed", "timestamp"]
    train_df = train_df.drop(drop_cols, axis = 1)

    return train_df, test_df


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