from prepare_data import prepare
import os
import lightgbm as lgb
import gc
from sklearn.model_selection import KFold
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def fit_rf(train_df):
    sample_size = 2e5/len(train_df) 

def fit_lgbm(train_df):

    # features and target vars
    target = np.log1p(train_df["meter_reading"])
    features = train_df.drop('meter_reading', axis = 1)
    del train_df
    gc.collect()

    categorical_features = ["building_id", "site_id", "meter", "primary_use", "weekend"]
    params = {
        "objective": "regression",
        "boosting": "gbdt",
        "num_leaves": 1280,
        "learning_rate": 0.05,
        "feature_fraction": 0.85,
        "reg_lambda": 2,
        "metric": "rmse",
    }

    kf = KFold(n_splits=3)
    models = []
    for train_index,test_index in kf.split(features):
        train_features = features.loc[train_index]
        train_target = target.loc[train_index]
        test_features = features.loc[test_index]
        test_target = target.loc[test_index]
        d_training = lgb.Dataset(train_features, label=train_target,categorical_feature=categorical_features, free_raw_data=False)
        d_test = lgb.Dataset(test_features, label=test_target,categorical_feature=categorical_features, free_raw_data=False)
        model = lgb.train(params, train_set=d_training, num_boost_round=1000, valid_sets=[d_training,d_test], verbose_eval=25, early_stopping_rounds=50)
        models.append(model)
        del train_features, train_target, test_features, test_target, d_training, d_test
        gc.collect()
    return models

def apply_lgbm(train_df, test_df):
    
    # Applying
    models = fit_lgbm(train_df)
    
    # Showing feature importances
    for model in models:
        lgb.plot_importance(model)
        plt.show()    

    results = []
    for model in models:
        if  results == []:
            results = np.expm1(model.predict(test_df, num_iteration=model.best_iteration)) / len(models)
        else:
            results += np.expm1(model.predict(test_df, num_iteration=model.best_iteration)) / len(models)
        del model
        gc.collect()
    


if __name__ == "__main__":
    # print(os.getcwd())
    train, test = prepare()
    train.to_csv("data/train_with_features.csv")
    train = pd.read_csv("data/train_with_features.csv")
    print(train.head())
    apply_lgbm(train)

    

