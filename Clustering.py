
import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd
import seaborn as sns
from sklearn.datasets import fetch_covtype
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
def sigmoid(x):
    return 1/(1+np.exp(-x))
def standardisecols(x):
    XMean = x.mean(axis=0, keepdims=True)
    XStd = x.std(axis=0, keepdims=True)
    return (x - XMean) / XStd

Data, Target = fetch_covtype(return_X_y=True, shuffle=True)
print(Data.shape[0], 'examples')
print(Data.shape[1], "features")
featureNames = fetch_covtype().feature_names
print(featureNames)

print(Data[0][0])
print(Target[0])

scaler = StandardScaler().fit(Data)
X_scaled = scaler.transform(Data)
print(X_scaled.mean(axis=0))
print(X_scaled.std(axis=0))
X_train, X_test, y_train, y_test = train_test_split(X_scaled, Target, test_size=0.982788, random_state=42)
print('Training set size:', len(X_train))
print('Test set size:', len(X_test))
#0.982788
