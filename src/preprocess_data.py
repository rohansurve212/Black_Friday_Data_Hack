# Import Libraries
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline


def transform_num_features(df_features, df_num_feature_columns):
    """
            Takes a pandas Dataframe (part of the black_friday_data_hack project) as input and
            transforms the numerical features (imputing missing values with median value and
            normalizing the values in each column such that minimum is 0 and maximum is 1)
            Arguments: data - DataFrame
            Returns: A numpy array of transformed numerical features
    """

    # Let's build a pipeline to transform numerical features
    df_num = df_features[df_num_feature_columns]

    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('normalizer', MinMaxScaler())
    ])

    df_num_tr = num_pipeline.fit_transform(df_num)

    return df_num_tr


def transform_cat_features(df_features, df_cat_feature_columns):
    """
            Takes a pandas DataFrame (part of the black_friday_data_hack project) as input and
            transforms the categorical features (imputing missing values with most frequent occurence
            value and performing one-hot encoding)
            Arguments: data - DataFrame
            Returns: A numpy array of transformed categorical features
    """

    # Let's build a pipeline to transform categorical features
    df_cat = df_features[df_cat_feature_columns]

    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('one_hot_encoder', OneHotEncoder())
    ])

    df_cat_tr = cat_pipeline.fit_transform(df_cat)

    return df_cat_tr


def transform_all_features(df_features, df_num_feature_columns, df_cat_feature_columns):
    """
            Takes a pandas Dataframe (part of the black_friday_data_hack project) as input and
            transforms all the features (uses both the pipelines)
            Arguments: data - DataFrame
            Returns: A numpy array of transformed features
    """

    # Let's create the full pipeline
    try:
        from sklearn.compose import ColumnTransformer
    except ImportError:
        from future_encoders import ColumnTransformer

    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('normalizer', MinMaxScaler())
    ])

    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('one_hot_encoder', OneHotEncoder())
    ])

    full_pipeline = ColumnTransformer([
        ('num', num_pipeline, df_num_feature_columns),
        ('cat', cat_pipeline, df_cat_feature_columns)
    ])

    df_prepared = full_pipeline.fit_transform(df_features)

    return df_prepared
