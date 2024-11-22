from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import fetch_covtype
from itertools import combinations
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np
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
X_train, X_test, y_train, y_test = train_test_split(X_scaled, Target, test_size=0.2, random_state=42)
Classifiers = LogisticRegression(random_state=0).fit(X_train, y_train)
Prediction = Classifiers.predict(X_test)
LogAccuracy = accuracy_score(y_test, Prediction)
print(LogAccuracy)
model = DecisionTreeClassifier(random_state=0)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
DecAccuracy = accuracy_score(y_test, predictions)
print(DecAccuracy)
modell = RandomForestClassifier(n_estimators=100, random_state=0)
modell.fit(X_train, y_train)
predictions = model.predict(X_test)
ForAccuracy = accuracy_score(y_test, predictions)
print(ForAccuracy)