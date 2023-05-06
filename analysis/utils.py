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
import seaborn as sns
from sklearn import metrics


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


def fit_models(x_train, y_train, x_test, y_test):
    
    '''
    this function fits logistic regression, MLP, and random forest 
    and returns the 3 models. 
    
    inputs:
        - x_train, y_train, x_test, y_test 
        
    outputs:
        - logistic regression model, MLP model, and RF model
    
    '''
    
    logisticRegr = LogisticRegression()
    logisticRegr.fit(x_train, y_train)
    
    clf = MLPClassifier(random_state=1, max_iter=300).fit(x_train, y_train)

    RF = RandomForestClassifier()
    RF.fit(x_train,y_train)
    
    return logisticRegr, clf, RF
    
    
from sklearn import datasets, metrics, model_selection, svm
def choose_best_model(logisticRegr, clf, RF):
    '''
    this function chooses the model with the highest accuracy and returns it
    
    inputs:
        - 3 models: logisticRegr, clf, RF
    outputs:
        - best_model = given by choose_best_model
		- best_test_predictions = list of test predictions given by best model
		- best_model_name = str of name of best model
    
    '''
    models = [logisticRegr, clf, RF]
    log_test_accuracy = logisticRegr.score(x_test, y_test)
    clf_test_accuracy = clf.score(x_test, y_test)
    rf_test_accuracy = RF.score(x_test, y_test)
    
    accuracies = [log_test_accuracy, clf_test_accuracy, rf_test_accuracy]
    best_accuracy = np.max(accuracies)
    best_model = models[np.argmax(accuracies)]

    best_test_predictions = best_model.predict(x_test)
   
    
    model_names = ['logistic regression', 'mlp', 'random forest']
    best_model_name = model_names[np.argmax(accuracies)]
    
    
	
    return best_model, best_test_predictions, best_model_name
    

	
def graph_confusion_roc(best_model, x_test, y_test, best_test_predictions, best_model_name):
    '''
    this function
    creates graphs of roc curve and confusion matrix for the best model
    
    inputs:
        - best_model = given by choose_best_model
		- best_test_predictions = list of test predictions given by best model
		- best_model_name = str of name of best model
    outputs:
        - graph of roc curve, saved to ./figures
        - graph of confusion matrix, saved to ./figures

    '''


    print('Best Model is: {0}'.format(best_model_name))
    cm = metrics.confusion_matrix(y_test, best_test_predictions)
    
    plt.figure(figsize=(9,9))
    sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r')
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    all_sample_title = 'Accuracy Score of Best Model: {0}'.format(score)
    plt.title(all_sample_title, size = 15)
    plt.savefig('../figures/'+ best_model_name + '_confusion_matrix')
    plt.show()

    
    fpr, tpr, thresholds = metrics.roc_curve(list(y_test), best_test_predictions)
    print( "Best model AUC is: {0}".format(metrics.auc(fpr, tpr)))
    plt.plot(fpr, tpr)
    plt.title("ROC curve")
    plt.xlabel("TPR")
    plt.ylabel("FPR")
    plt.savefig('../figures/'+ best_model_name + '_ROC_curve')
    plt.show()
    