# Library imports
import numpy as np
import pandas as pd
import os


def make_dataset(dataframe):
    
    """ Pre-processes raw data (../data/raw) to turn it into
        interim data ready to be processed (saved in ../data/interim).
    """
    # Set random number seed
    np.random.seed(123)

    # Convert the dataframe into an array
    df_array = np.array(dataframe)
    columns = dataframe.columns.tolist()

    # shuffle the instances within the array
    shuffled_array = np.random.permutation(df_array)
    
    # take the first 100k instances of the shuffled array
    sliced_array = shuffled_array[:100000]

    # convert it into a pandas dataframe to further pre-process it
    interim_data = pd.DataFrame(data=sliced_array,
                                columns=columns)

    #destination_path = os.path.join(os.path.abspath(path), 'data', 'interim', 'interim_data.csv')
    export_csv = interim_data.to_csv(r'C:\Users\visha\Documents\GitHub\Black_Friday_Data_Hack\Black_Friday_Data_Hack\data\interim\interim_data.csv', index=None, header=True)

    return interim_data
