
import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd
import sklearn.cluster
from kneed import KneeLocator
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import fetch_covtype
from itertools import combinations
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LogisticRegression
def sigmoid(x):
    return 1/(1+np.exp(-x))

def TestValues(y, z):
    n = len(y)
    CorrectCounter = 0
    PairNumber = 0
    for i, j in combinations(range(n), 2):
        if (y[i] == y[j]):
            CorrectCounter += (z[i] == z[j])
            PairNumber += 1
    IncorrectCounter = PairNumber - CorrectCounter
    return IncorrectCounter, CorrectCounter, PairNumber

Data, Target = fetch_covtype(return_X_y=True, shuffle=True)
# print(Data.shape[0], 'examples')
# print(Data.shape[1], "features")
featureNames = fetch_covtype().feature_names
# print(featureNames)
# print(Data[0][0])
# print(Target[0])

scaler = StandardScaler().fit(Data)
X_scaled = scaler.transform(Data)
# print(X_scaled.mean(axis=0))
# print(X_scaled.std(axis=0))
X_train, X_test, y_train, y_test = train_test_split(X_scaled, Target, test_size=0.982788, random_state=42)
# print('Training set size:', len(X_train))
# print('Test set size:', len(X_test))
#0.982788

Kmean = KMeans(n_clusters=7).fit(X_train)
# print(Kmean.labels_)

IncorrectCounter, CorrectCounter, PairAmount = TestValues(y_train, Kmean.labels_)
Accuracy = round((CorrectCounter/PairAmount)*100, 4)
print("K-Means Analysis:")
print("Total Pairs:", PairAmount)
print("Correct Pairs:", CorrectCounter)
print("Incorrect Pairs:", IncorrectCounter)
print("Accuracy:",Accuracy,"%")
print("Error Rate:",100-Accuracy,"%")

GausMix = GaussianMixture(n_components=7).fit(X_train)
Predicted = GausMix.predict(X_train)
IncorrectCountGmm, CorrectCountGmm, PairAmountGmm = TestValues(y_train, Predicted)
AccuracyGmm = round((CorrectCountGmm/PairAmountGmm)*100, 4)
print("Gaussian Analysis:")
print("Total Pairs:", PairAmountGmm)
print("Correct Pairs:", CorrectCountGmm)
print("Incorrect Pairs:", IncorrectCountGmm)
print("Accuracy:",AccuracyGmm,"%")
print("Error Rate:",100-AccuracyGmm,"%")

random_labels = np.random.choice(range(7), size=len(y_train), replace=True)
IncorrectCountRand, CorrectCountRand, PairAmountRand = TestValues(y_train, random_labels)
AccuracyRand= round((CorrectCountRand/PairAmountRand)*100, 4)
print("Random Analysis:")
print("Total Pairs:", PairAmountRand)
print("Correct Pairs:", CorrectCountRand)
print("Incorrect Pairs:", IncorrectCountRand)
print("Accuracy:",AccuracyRand,"%")
print("Error Rate:",100-AccuracyRand,"%")

#Classification
# X_train, X_test, y_train, y_test = train_test_split(X_scaled, Target, test_size=0.2, random_state=42)
# Classifiers = LogisticRegression(random_state=0).fit(X_train, y_train)
# Prediction = Classifiers.predict(X_test)
# LogAccuracy = accuracy_score(y_test, Prediction)

# model = DecisionTreeClassifier(random_state=0)
# model.fit(X_train, y_train)
# predictions = model.predict(X_test)
# DecAccuracy = accuracy_score(y_test, predictions)
#
# modell = RandomForestClassifier(n_estimators=100, random_state=0)
# modell.fit(X_train, y_train)
# predictions = model.predict(X_test)
# ForAccuracy = accuracy_score(y_test, predictions)