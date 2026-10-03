import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

data = pd.read_csv(r'house_pricing.txt', header = None, delimiter = ",") #read from dataset
X = data.iloc[:, 0:2]
print('X.shape: ', X.shape)
y = data.iloc[:,2]
print('y.shape: ', y.shape)
m = len(y)
print('Number of samples:', m)
print(data.head()) 

model = LinearRegression()
model.fit(X,y)

y_pred_train = model.predict(X)
mse= mean_squared_error(y, y_pred_train)
cost = mse/2
print(f'cost: {cost}')

value_to_be_predicted = [5500, 6]
new_X = np.array([value_to_be_predicted])
pred = model.predict(new_X)[0]
print(f'Predicted y for X={value_to_be_predicted}: ${pred:.2f}')