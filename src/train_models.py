# Import Libraries
import numpy as np
import h5py
import pickle
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from xgboost.sklearn import XGBRegressor
from keras import models
from keras import layers
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split


def train_lasso_reg(df_prepared, df_target):
    """
            Trains a LASSO Regression with alpha = 1
            :param: df_prepared
            :param: df_target
            :return: A tuple of trained Model and best cross_validated RMSE
    """

    lasso_reg = Lasso()

    params = {'alpha': [1]}

    gridsearch_lr = GridSearchCV(lasso_reg, params, cv=4, verbose=1, n_jobs=-1, scoring='neg_mean_squared_error')
    gridsearch_lr.fit(df_prepared, df_target)

    # save the model to disk
    lr_filename = r'models\lasso_model.sav'
    pickle.dump(gridsearch_lr, open(lr_filename, 'wb'))
    print('LASSO Regression Model saved to disk')

    return gridsearch_lr.best_estimator_, np.sqrt(-gridsearch_lr.best_score_)


def train_random_forest_reg(df_prepared, df_target):
    """
            Trains a Random Forest Regressor
            :param: df_prepared
            :param: df_target
            :return: A tuple of trained Model and best cross_validated RMSE
    """

    forest_reg = RandomForestRegressor(max_features='auto', random_state=123)

    params = {'max_depth': [10],
              'min_samples_split': [1000],
              'n_estimators': [10]}

    gridsearch_rf = GridSearchCV(forest_reg, params, cv=4, verbose=1, n_jobs=-1, scoring='neg_mean_squared_error')
    gridsearch_rf.fit(df_prepared, df_target)

    # save the model to disk
    rf_filename = r'models\rnd_forest_model.sav'
    pickle.dump(gridsearch_rf, open(rf_filename, 'wb'))
    print('Random Forest Model saved to disk')

    return gridsearch_rf.best_estimator_, np.sqrt(-gridsearch_rf.best_score_)


def train_xgboost_reg(df_prepared, df_target):
    """
            Trains an XGBoost Regressor
            :param: df_prepared
            :param: df_target
            :return: A tuple of trained Model and best cross_validated RMSE
    """

    xgb_reg = XGBRegressor()

    params = {'nthread': [4],
              'objective': ['reg:linear'],
              'learning_rate': [0.07],  # so called `eta` value
              'max_depth': [7],
              'min_child_weight': [4],
              'silent': [1],
              'subsample': [0.7],
              'colsample_bytree': [0.7],
              'n_estimators': [500]}

    gridsearch_xgb = GridSearchCV(xgb_reg, params, cv=4, verbose=1, n_jobs=-1, scoring='neg_mean_squared_error')
    gridsearch_xgb.fit(df_prepared, df_target)

    # save the model to disk
    xgb_filename = r'models\xgboost_model.sav'
    pickle.dump(gridsearch_xgb, open(xgb_filename, 'wb'))
    print('XGBoost Model saved to disk')

    return gridsearch_xgb.best_estimator_, np.sqrt(-gridsearch_xgb.best_score_)


def train_deep_neural_network(df_prepared, df_target):
    """
            Trains a Multi-Level Perceptron (Deep Neural Network) with Dense Layers using activation function
            'ReLU' implemented in Keras
            :param: df_prepared
            :return: A tuple of trained Model and best cross_validated RMSE
    """

    # Build a network architecture of type Sequential
    model = models.Sequential()

    # Add 5 hidden layers to the network
    model.add(layers.Dense(512, activation='relu', input_shape=(df_prepared.shape[1],)))
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(512, activation='relu'))

    # Add a single unit output layer (since it's a scalar regression problem)
    model.add(layers.Dense(1))

    # Compile the model (loss function is 'Mean Squared Error')
    model.compile(optimizer='Adam',
                  loss='mse',
                  metrics=['mse'])

    # Validating Our Model Using Hold-Out Validation
    partial_train_data, val_data, partial_train_target, val_target = train_test_split(df_prepared, df_target,
                                                                                      test_size=0.2, random_state=123)

    model.fit(partial_train_data, partial_train_target, epochs=10, batch_size=1000, verbose=1)
    val_mse = model.evaluate(val_data, val_target, verbose=0)

    # save the model to disk
    model.save(r'models\dnn_model.h5')
    print('DNN Model saved to disk')

    return model, np.sqrt(val_mse[0])
