from prepare_data import prepare
import os






def 






if __name__ == "__main__":
    # print(os.getcwd())
    train = prepare()
    print(train.head())
    # train[train["site_id"] == 3].plot("timestamp", "meter_reading")
