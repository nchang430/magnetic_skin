import pandas as pd
import operator
import pickle
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import math

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate
import itertools  

from sklearn.model_selection import GridSearchCV

# enter file name of data you want to analyze
filename = ''
full_dataset = pd.read_pickle("./" + str(filename) + " SVD.pkl")
print(full_dataset.head())

X = full_dataset.drop(['X', 'Y', 'Z', 'F'], axis=1)
y = full_dataset[['X','Y','Z']]

scaler = StandardScaler()
mlp = MLPRegressor(early_stopping=True)
pipeline = Pipeline([('transformer', scaler), ('estimator', mlp)])

parameters={
'estimator__learning_rate': ['constant'],
'estimator__learning_rate_init': [0.005],
'estimator__hidden_layer_sizes': [x for x in itertools.product((20,50,100,150,200),repeat=2)],
'estimator__activation': ['tanh', 'relu', 'logistic'],
'estimator__max_iter': [5000],
'estimator__batch_size': [20, 50, 100, 150, 200]
}

clf = GridSearchCV(pipeline, parameters, cv=5)

clf.fit(X,y)
print("Best parameter (CV score=%0.3f):" % clf.best_score_)
print(clf.best_params_)

pickle.dump(clf, open(filename + ' GS Object.pkl', 'wb'))