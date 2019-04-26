# Import Libraries
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from xgboost.sklearn import XGBRegressor
from keras import models
from keras import layers
from sklearn.metrics import mean_squared_error


def evaluate_on_test(model, df_prepared, df_target):
    """
            Evaluate the performance of the model on test set
            :params: Any of the trained models
            :params: Test Features
            :params: Test Target
            :return: RMSE of predicted values
    """

    # Make predictions on the data
    df_prediction = model.predict(df_prepared)

    # Evaluate the predictions against the actual targets
    df_score = mean_squared_error(df_target, df_prediction)

    return np.sqrt(df_score)
