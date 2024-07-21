import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

ds=pd.read_csv(".\\archive\\Real estate.csv")


ds=ds.drop(columns=['No', 'X1 transaction date', 'X5 latitude' ,'X6 longitude']) 

X= ds[['X2 house age', 'X3 distance to the nearest MRT station' ,'X4 number of convenience stores']]
y=ds['Y house price of unit area']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

norm = MinMaxScaler()
X_train_normal = norm.fit_transform(X_train)

X_test_normal = norm.transform(X_test)

X_train_normal_ds = pd.DataFrame(X_train_normal, columns=X.columns)
X_test_normal_ds = pd.DataFrame(X_test_normal, columns=X.columns)


#Sklearn Linear Model
lm = LinearRegression()

lm.fit(X_train, y_train)
linear_predictions = lm.predict(X_test_normal)
linear_mse = mean_squared_error(y_test, linear_predictions)

linear_r2 = r2_score(y_test, linear_predictions)

print(linear_mse)
print(linear_r2)