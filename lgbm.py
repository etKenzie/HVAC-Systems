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
    print(features.head())
    del train_df
    gc.collect()

    categorical_features = ["building_id", "site_id", "meter", "primary_use"]

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
        model = lgb.train(params, train_set=d_training, num_boost_round=1000, valid_sets=[d_training,d_test], verbose_eval=True, early_stopping_rounds=50)
        models.append(model)
        del train_features, train_target, test_features, test_target, d_training, d_test
        gc.collect()
    return models

def apply_lgbm(train_df, test_df, row_ids):
    
    # Applying
    models = fit_lgbm(train_df)
    
    # Showing feature importances
    for i, model in enumerate(models):
        lgb.plot_importance(model)
        plt.savefig(f'plots/feature_importance{i+1}')

    results = []
    for model in models:
        if len(results) == 0:
            results = np.expm1(model.predict(test_df, num_iteration=model.best_iteration)) / len(models)
        else:
            results += np.expm1(model.predict(test_df, num_iteration=model.best_iteration)) / len(models)
        del model
        gc.collect()
    
    results_df = pd.DataFrame({"row_id": row_ids, "meter_reading": np.clip(results, 0, a_max=None).T})
    del row_ids,results
    gc.collect()
    results_df.to_csv("submission.csv", index=False)


if __name__ == "__main__":
    # print(os.getcwd())
    # train, test, row_ids = prepare()
    # train.to_csv("data/train_with_features.csv", index=False)
    # test.to_csv("data/test_with_features.csv", index = False)
    # row_ids.to_csv("data/row_ids.csv", index = False)
    train = pd.read_csv("data/train_with_features.csv")
    test = pd.read_csv("data/test_with_features.csv")
    row_ids = pd.read_csv("data/row_ids.csv")

    print(train.head())
    apply_lgbm(train, test, row_ids)

    

