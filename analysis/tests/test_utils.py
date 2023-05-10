import pytest
import utils as ul
import sklearn
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from seaborn import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from seaborn import load_dataset
import seaborn as sns
from sklearn import metrics
import os
from os.path import exists
from sklearn.metrics import recall_score, precision_score


# read in data and add code that is necessary to test fit_models function
raw_data = pd.read_csv('./data/heart.csv')
categorical_columns= ['Sex', 'ChestPainType','RestingECG', 'ExerciseAngina', 'ST_Slope' ]
categorical_data = raw_data[categorical_columns]
enc = OneHotEncoder(sparse=False).fit(categorical_data)
encoded = enc.transform(categorical_data)
encoded_df = pd.DataFrame(
     encoded, 
     columns=enc.get_feature_names_out())

modified_df = pd.concat([raw_data[raw_data.columns.difference(categorical_columns)], encoded_df], axis = 1)
x = modified_df.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
df = pd.DataFrame(x_scaled, columns = modified_df.columns)

features = ['Age', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak',
       'RestingBP', 'Sex_F', 'Sex_M', 'ChestPainType_ASY', 'ChestPainType_ATA',
       'ChestPainType_NAP', 'ChestPainType_TA', 'RestingECG_LVH',
       'RestingECG_Normal', 'RestingECG_ST', 'ExerciseAngina_N',
       'ExerciseAngina_Y', 'ST_Slope_Down', 'ST_Slope_Flat', 'ST_Slope_Up']
X,y = df[features], df['HeartDisease']
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)


# tests
def test_clean_data():
    '''
    Check the type of the outputs, and that the features and labels are same as the input.
    '''
    features, labels = ul.clean_data(raw_data)
    right_features =['Age',
				 'Cholesterol',
				 'FastingBS',
				 'MaxHR',
				 'Oldpeak',
				 'RestingBP',
				 'Sex_F',
				 'Sex_M',
				 'ChestPainType_ASY',
				 'ChestPainType_ATA',
				 'ChestPainType_NAP',
				 'ChestPainType_TA',
				 'RestingECG_LVH',
				 'RestingECG_Normal',
				 'RestingECG_ST',
				 'ExerciseAngina_N',
				 'ExerciseAngina_Y',
				 'ST_Slope_Down',
				 'ST_Slope_Flat',
				 'ST_Slope_Up']
    
    assert type(features) == pd.DataFrame
    assert type(labels) == pd.core.series.Series
    assert features.columns.values.tolist() == right_features
    assert raw_data['HeartDisease'].astype(float).equals(labels)

logisticRegr, clf, RF = ul.fit_models(x_train, y_train, x_test, y_test)
best_model, best_test_predictions, best_model_name = ul.choose_best_model(logisticRegr, clf, RF, x_test, y_test)
    ### doesn't actually work, but think it is a problem in utils
    
def test_fit_models():
    '''
    Check that the outputs are a logistic regression model, MLP model, and RF model, respectively.
    '''
    assert type(logisticRegr) == sklearn.linear_model._logistic.LogisticRegression
    assert type(clf) == sklearn.neural_network._multilayer_perceptron.MLPClassifier
    assert type(RF) == sklearn.ensemble._forest.RandomForestClassifier


def test_choose_best_model():
    ''' 
    Check that a valid model type is returned.
    '''
    model_names = ['logistic regression', 'mlp', 'random forest']
    assert best_model_name in model_names 
    
def test_graph_confusion_roc():
    '''
    Check that the figure was saved in the figures folder, and it is named correctly.
    '''
    ul.graph_confusion_roc(best_model, x_test, y_test, best_test_predictions, best_model_name, root_dir = '.')
    assert os.path.exists('./figures/'+ best_model_name + '_ROC_curve.png')
