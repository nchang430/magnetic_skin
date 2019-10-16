import pandas as pd
import operator
import pickle
import numpy as np
import matplotlib.pyplot as plt
import math

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate

filename = ''
full_dataset = pd.read_pickle("./" + str(filename) + " SVD.pkl")
print(full_dataset.head())


X = full_dataset.drop(['X', 'Y', 'Z', 'F'], axis=1)
y = full_dataset[['X','Y','Z']]

scaler = StandardScaler()
# choose parameters from preivous grid search
mlp = MLPRegressor(hidden_layer_sizes=(150,200), solver='adam', max_iter=5000, learning_rate_init=0.005, activation='tanh', batch_size=20, random_state=7, early_stopping=True)
pipeline = Pipeline([('transformer', scaler), ('estimator', mlp)])

scores = cross_validate(pipeline, X, y, cv=5, scoring=('r2','neg_mean_squared_error','neg_mean_absolute_error'), return_estimator=True, return_train_score=True)

np.savez(filename + ' KFold Model.npz', scores=scores, scaler=scaler)

print("Training RMSE: ")
print(scores['train_neg_mean_squared_error'])
print("Testing RMSE: ")
print(scores['test_neg_mean_squared_error'])

print("Training R2: ")
print(scores['train_r2'])
print("Testing R2: ")
print(scores['test_r2'])

print("Training MAE: ")
print(scores['train_neg_mean_absolute_error'])
print("Testing MAE: ")
print(scores['test_neg_mean_absolute_error'])
