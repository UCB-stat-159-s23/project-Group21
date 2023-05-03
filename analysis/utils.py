from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from seaborn import load_dataset


def clean_data(raw_data):
    '''
    This function one hot encodes the categorical columns, and uses sklearn's preprocessing function MinMaxScaler() to scale the data
    Input: raw_data file in .csv format
    Output: Returns the features X, and the corresponding labels Y
    '''
    # create an encoder and fit the dataframe
    categorical_columns= ['Sex', 'ChestPainType','RestingECG', 'ExerciseAngina', 'ST_Slope' ]
    categorical_data = raw_data[categorical_columns]
    enc = OneHotEncoder(sparse=False).fit(categorical_data)
    encoded = enc.transform(categorical_data);
    # convert it to a dataframe
    encoded_df = pd.DataFrame(
         encoded, 
         columns=enc.get_feature_names_out()
    );
    modified_df = pd.concat([raw_data[raw_data.columns.difference(categorical_columns)], encoded_df], axis = 1)
    x = modified_df.values #returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    df = pd.DataFrame(x_scaled, columns = modified_df.columns)
    features = df.columns[:-1]
    return df[features], df['HeartDisease']