import os
import random
import gc

# import tqdm
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

def prepare():
    # Read Data Files
    path = os.getcwd() + '/data/'
    train_df = pd.read_csv(os.path.join(path, 'train.csv'))
    test_df = pd.read_csv(os.path.join(path, 'test.csv'))
    buildings_df = pd.read_csv(os.path.join(path, 'building_metadata.csv'))
    weather_test_df = pd.read_csv(os.path.join(path, 'weather_test.csv'))
    weather_train_df = pd.read_csv(os.path.join(path, 'weather_train.csv'))

    # Merge Data Based on building_id and time
    train_df = train_df.merge(buildings_df, on='building_id', how='left')
    train_df  = train_df.merge(weather_train_df, on=['site_id', 'timestamp'], how='left')

    # Add clearer variables regarding date/time
    train_df["timestamp"] = pd.to_datetime(train_df["timestamp"])
    train_df["hour"] = train_df["timestamp"].dt.hour
    train_df["day"] = train_df["timestamp"].dt.day
    train_df["weekend"] = train_df["timestamp"].dt.weekday
    train_df["month"] = train_df["timestamp"].dt.month

    return train_df


# if __name__ == "__main__":
#     root = 'input'
#     output = 'processed'
#     prepare(root, output)