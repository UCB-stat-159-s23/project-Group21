import pytest
from analysis import utils as ul
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
import seaborn as sns
from sklearn import metrics

raw_data = pd.read_csv('../data/heart.csv')

## clean data test
def test_clean_data():
    '''
    Check the type of the outputs, and that the features and labels are same as the input.
    '''
    features, labels = ul.clean_data(raw_data)
    assert type(features) == pd.core.indexes.base.Index
    assert type(labels) == pd.core.series.Series
    #now check the actual values of features and labels

## fit models test

## choose best model test