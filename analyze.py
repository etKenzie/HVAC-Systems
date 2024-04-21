from prepare_data import prepare
import os
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

def random_forest(training,target): 
    X_train, X_test, y_train, y_test = train_test_split(training, target, test_size=0.2, random_state=42)

    imputer = SimpleImputer(strategy='mean')
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    y_pred = rf_model.predict(X_test)

    # Evaluate model
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

    # Feature Importance
    feature_importance = rf_model.feature_importances_
    print("Feature Importance:", feature_importance)

    return rf_model

    

if __name__ == "__main__":
    # print(os.getcwd())
    train = prepare()
    X = train[['temp_difference', 'is_weekend', 'building_age', 'relative_humidity', 'hour', 'day', 'month', 'square_feet']]  # plus encoded 'primary_use' columns
    y = train['meter_reading']
    

    random_forest(X,y)
    # train[train["site_id"] == 3].plot("timestamp", "meter_reading")
