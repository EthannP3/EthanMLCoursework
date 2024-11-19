
import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd
import seaborn as sns
import sklearn.cluster
from kneed import KneeLocator
from sklearn.datasets import fetch_covtype
from itertools import combinations
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
def sigmoid(x):
    return 1/(1+np.exp(-x))
def standardisecols(x):
    xmean = x.mean(axis=0, keepdims=True)
    xstd = x.std(axis=0, keepdims=True)
    return (x - xmean) / xstd
def TestValues(y, z):
    n = len(y)
    CorrectCounter = 0
    PairNumber = 0
    for i, j in combinations(range(n), 2):
        if (y[i] == y[j]):
            CorrectCounter += (z[i] == z[j])
            PairNumber += 1
    IncorrectCounter = PairNumber - CorrectCounter
    return IncorrectCounter, CorrectCounter
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

Kmean = KMeans(n_clusters=7).fit(X_train)
print(Kmean.labels_)

IncorrectCounter, CorrectCounter = TestValues(y_train, Kmean.labels_)
print(IncorrectCounter)
print(CorrectCounter/IncorrectCounter)
print(CorrectCounter)


GausMix = GaussianMixture(n_components=7).fit(X_train)
Predicted = GausMix.predict(X_train)
IncorrectCountGmm, CorrectCountGmm = TestValues(y_train, Predicted)
print(IncorrectCountGmm)
print(CorrectCountGmm/IncorrectCountGmm)
print(CorrectCountGmm)

random_labels = np.random.choice(range(7), size=len(y_train), replace=True)
IncorrectCountRand, CorrectCountRand = TestValues(y_train, random_labels)

print(IncorrectCountRand)
print(CorrectCountRand/IncorrectCountRand)
print(CorrectCountRand)
#Classification
X_train, X_test, y_train, y_test = train_test_split(X_scaled, Target, test_size=0.2, random_state=42)