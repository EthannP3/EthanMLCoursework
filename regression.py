import matplotlib.pyplot as plt
import torch
import pymc as pm
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
class FeedForwardNN(nn.Module):
    def __init__(self, inputSize, outputSize, hiddenSize):
        super(FeedForwardNN, self).__init__()
        self.fc1 = nn.Linear(inputSize, hiddenSize)  # First fully connected layer
        self.fc2 = nn.Linear(hiddenSize, outputSize)  # Second fully connected layer
        #self.dropout = nn.Dropout(0.5)  # Dropout for regularization
    def forward(self, x):
        x = F.relu(self.fc1(x))  # Apply ReLU activation to first layer
        #x = self.dropout(x)  # Apply dropout
        x = self.fc2(x)  # Apply second layer (output)
        return x
f = open("regression.train.txt", "r")
g = open("regression.test.txt", "r")
num = 0
RegrDataX_train = []
RegrDataY_train = []
RegrDataX_test = []
RegrDataY_test = []
for x in f:
      XYsplit = x.split(" ")
      if len(XYsplit[0]) > 1:
          RegrDataX_train.append(float(XYsplit[0]))
          RegrDataY_train.append(float(XYsplit[1]))
          print(num, "x=", RegrDataX_train[num],
                     "y=", RegrDataY_train[num])
          num += 1
      else:
          break
f.close()
num = 0
for y in g:
    XYsplit2 = y.split(" ")
    if len(XYsplit2[0]) > 1:
        RegrDataX_test.append(float(XYsplit2[0]))
        RegrDataY_test.append(float(XYsplit2[1]))
        num += 1
    else:
        break
g.close()
print(RegrDataY_train[75])
ArrXTrain = np.array(RegrDataX_train)
ArrYTrain = np.array(RegrDataY_train)
ArrXTest = np.array(RegrDataX_test)
ArrYTest = np.array(RegrDataY_test)
print(ArrXTrain[75])
ArrXTrain = ArrXTrain.reshape(-1, 1)
print(ArrXTrain[75])
print(ArrYTrain[75])
ArrYTrain = ArrYTrain.reshape(-1, 1)
print(ArrYTrain[75])
ArrXTest = ArrXTest.reshape(-1, 1)
ArrYTest = ArrYTest.reshape(-1, 1)
print(ArrXTrain)
# Regressed = linear_model.LinearRegression()
# Regressed.fit(ArrXTrain, ArrYTrain)
# YPrediction = Regressed.predict(ArrXTest)
# print("Coefficients: \n", Regressed.coef_)
# print("Mean squared error: %.2f" % mean_squared_error(ArrYTest, YPrediction))
# print("Coefficient of determination: %.2f" % r2_score(ArrYTest, YPrediction))
# inputSize = 1
# outputSize = 1
# hiddenSize = 512
# learnRate = 0.001
# epochs = 500
# Neural = FeedForwardNN(inputSize,
#                        outputSize,
#                        hiddenSize)
# lossFunction = nn.MSEloss()
# Optimizer = torch.optim.Adam(Neural.parameters(), learnRate)
# trainDataset = TensorDataset(torch.from_numpy(ArrXTrain), torch.from_numpy(ArrYTrain))
# trainLoader = DataLoader(trainDataset, batch_size=128, shuffle=True)
# for epoch in range(epochs):
#     Neural.train()
#     trainLoss = 0.0
#     for batchInput, batchOutput in trainLoader:
#         Output = Neural(batchInput)
#         Loss = lossFunction(Output, batchOutput)
#         Optimizer.zero_grad()
#         Loss.backward()
#         Optimizer.step()
#         trainLoss += Loss.item()
#     trainLoss /= len(trainLoader)
# Neural.eval()
# with torch.no_grad():
#     xTorch = torch.from_numpy(ArrXTest)
#     YPrediction = Neural(xTorch)
#     predLoss = lossFunction(YPrediction, torch.from_numpy(ArrYTest))
#     print(predLoss)
degree = 3  # Degree of the polynomial
XTrainPoly = np.vstack([ArrXTrain ** d for d in range(degree + 1)]).T

# Bayesian Polynomial Regression
with pm.Model() as bayesian_poly_model:
    # Priors for polynomial coefficients
    coefficients = pm.Normal("coefficients", mu=0, sigma=10, shape=(degree + 1,))
    sigma = pm.HalfNormal("sigma", sigma=10)

    # Likelihood
    mu = pm.math.dot(XTrainPoly, coefficients)  # Polynomial prediction
    y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=ArrYTrain)

    # Inference
    trace = pm.sample(2000, return_inferencedata=True)

# Plot posterior distributions
pm.plot_posterior(trace)
plt.show()

# plt.plot(ArrXTest, YPrediction, color="blue", linewidth=3)
# plt.xticks(())
# plt.yticks(())
# plt.show()
