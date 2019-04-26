# Import Libraries
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score


def stack_models(df_prepared, df_target, model_1, model_2, model_3, model_4):
    """
            Stack all the four models and blend their predictions in a Random Forest Regressor; Train and evaluate the
            Blender's performance on Stacking set
            :params: Stacking Set Features,
            :params: Stacking Set Target
            :params: All the different models we want to stack
            :return: A tuple of Trained stacked Model and the Mean of it's Cross-validated RMSE
    """

    # Bring together the best estimators of all the three ML models and the deep neural network model
    estimators = [model_1, model_2, model_3, model_4]

    # Creating training set for the Stacker/Blender
    stack_predictions = np.empty((df_prepared.shape[0], len(estimators)), dtype=np.float32)
    for index, estimator in enumerate(estimators):
        stack_predictions[:, index] = np.reshape(estimator.predict(df_prepared), (df_prepared.shape[0],))

    # Initializing the Stacker/Blender (Random Forest Regressor)
    rf_blender = RandomForestRegressor(n_estimators=20, random_state=123)

    # Evaluate the Blender on stacking set using cross-validation (# cross validation sets =3)
    val_scores = cross_val_score(rf_blender, stack_predictions, df_target, scoring='neg_mean_squared_error', n_jobs=-1)

    return rf_blender, np.mean(np.sqrt(np.array(val_scores)*-1))
