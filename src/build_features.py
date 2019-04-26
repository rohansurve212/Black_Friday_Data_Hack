# Import Libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def train_stack_test_split(data, train_size, stacking_size, random_state=123):

    """
        Takes a pandas Dataframe as input and splits it into three parts: train set,
        stacking set and test set.
        Arguments: data - DataFrame
                   train_size - Train set as a fraction of the input data
                   stacking_size - Stacking set as a fraction of the remaining data
                   random_state - numpy.random.seed value
        Returns: tuple of train_set, stacking_set, test_set
    """

    # First split the input data into train set and the rest
    full_train_set, full_test_set = train_test_split(data, train_size=train_size, random_state=random_state)

    # Then split the rest into stacking set and test set
    stacking_set, test_set = train_test_split(full_test_set, train_size=stacking_size, random_state=random_state)

    return full_train_set, stacking_set, test_set


# Create a function to convert data types of different columns

def convert_data_types(df):
    """
            Takes a pandas Dataframe (part of the black_friday_data_hack project) as input
            and convert the data types of some of its features.
            Arguments: data - DataFrame
            Returns: same DataFrame with converted data types
    """

    # Convert categorical features into numeric type
    df['Age'] = df['Age'].map({'0-17': 15, '18-25': 21, '26-35': 30, '36-45': 40, '46-50': 48, '51-55': 53, '55+': 55})
    df['Stay_In_Current_City_Years'] = df['Stay_In_Current_City_Years'].map({'0': 0, '1': 1, '2': 2, '3': 3, '4+': 4})

    # Convert numeric features into categorical type
    df['Occupation'] = df['Occupation'].astype('category')
    df['Product_Category_1'] = df['Product_Category_1'].astype('category')
    df['Product_Category_2'] = df['Product_Category_2'].astype('category')
    df['Product_Category_3'] = df['Product_Category_3'].astype('category')
    df['Marital_Status'] = df['Marital_Status'].astype('category')

    # Convert Product_ID to numerical type by discarding the 1st letter 'P'
    df['Product_ID'] = df['Product_ID'].map(lambda x: x[1:])
    df['Product_ID'] = df['Product_ID'].astype('int64')

    # Convert Purchase to numerical type
    df['Purchase'] = df['Purchase'].astype('int64')

    return df


def separate_features_target(df):
    """
            Takes a pandas Dataframe (part of the black_friday_data_hack project) as input
            and drops the 'Purchase' column
            Arguments: data - DataFrame
            Returns: tuple of features and targets
    """

    df_features = df.drop('Purchase', axis=1)
    df_target = df['Purchase'].copy()
    return df_features, df_target


def num_cat_feature_columns(df):
    """
            Takes a pandas Dataframe (part of the black_friday_data_hack project) as input and
            separates numerical features from categorical features
            Arguments: data - DataFrame
            Returns: tuple of numerical features and categorical features
    """

    df_features = list(df.drop('Purchase', axis=1).columns)

    df_num_feature_columns = ['User_ID', 'Product_ID', 'Age', 'Stay_In_Current_City_Years']
    df_cat_feature_columns = list(set(df_features) - set(df_num_feature_columns))

    return df_num_feature_columns, df_cat_feature_columns
