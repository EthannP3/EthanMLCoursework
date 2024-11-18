import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from skimage import io
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
def sigmoid(x):
    return 1/(1+np.exp(-x))
def standardiseCols(x):
    x_means = x.mean(axis=0, keepdims=True)
    x_std = x.std(axis=0, keepdims=True)
    return (x - x_means) / x_std
A = np.array([[2, 3], [4, -1], [5, 6]])
B = np.array([[5, 2], [8, 9], [2, 1]])
D = np.array([-5, 0, 5])
F = np.array([
    [0, 3, 5],
    [1, 6, 4],
    [3, -2, 8],
    [-1, 1, 10]
])

C = 3 * A
print('C1:')
print(C)

C = A + B
print('C2:')
print(C)

C = np.dot(A, B.T)
print('C3:')
print(B.T)

C = np.multiply(A, B)
print('C4:')
print(C)
ASum = np.sum(A)
print('A sum:', ASum)

Amean = A.mean()
print('A mean:', Amean)

Avar = A.var()
print('A var: ', Avar)

Asum_row = np.sum(A, axis=1)
print('A sum (row):', Asum_row)

Asum_col = A.sum(axis=0)
print('A sum (column):', Asum_col)
F_standard = standardiseCols(F)
print(F_standard.mean(axis=0))
print(F_standard.std(axis=0))

G = np.arange(5)
G = G.reshape(-1, 1)
print(G)
assert(G.shape == (5,1))

X, y = fetch_california_housing(return_X_y=True)
print(X.shape[0], 'examples')
print(X.shape[1], 'features')
feature_names = fetch_california_housing().feature_names
print(feature_names)

scaler = StandardScaler().fit(X)
X_scaled = scaler.transform(X)
print(X_scaled.mean(axis=0))
print(X_scaled.std(axis=0))

data = pd.DataFrame(X_scaled, columns=feature_names)
data['MedHouseValue'] = y
plot = sns.pairplot(data)

E = np.linspace(-10, 10, 100)
fig, ax = plt.subplots()
ax.plot(E, sigmoid(E))
ax.set_xlabel('x')
ax.set_ylabel('sigmoid(x)')
plt.show()