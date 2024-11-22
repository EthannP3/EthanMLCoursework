import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
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
          RegrDataX_train.append(XYsplit[0])
          RegrDataY_train.append(XYsplit[1])
          num += 1
      else:
          break
f.close()
num = 0
for y in g:
    XYsplit2 = y.split(" ")
    if len(XYsplit2[0]) > 1:
        RegrDataX_test.append(XYsplit2[0])
        RegrDataY_test.append(XYsplit2[1])
        num += 1
    else:
        break
g.close()

print(RegrDataX_test[87])