import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

class LinearRegression:
    def __init__(self, learning_rate, epochs):
        self.lr=learning_rate
        self.epochs=epochs

    def fit(self, X_train, y_train):
        n_samples, n_features = X_train.shape
        y_train=y_train.reshape(-1,1)
        # init parameters
        self.weights = np.zeros((n_features,1))
        self.bias = np.zeros((1,1))

        # gradient descent
        for i in range(self.epochs):
            delta= -(y_train-np.dot(X_train,self.weights)-self.bias)/n_samples
            dw= np.dot(X_train.T,delta)
            db= np.sum(delta).reshape(1,1)

            #update weights and biases
            self.weights-= self.lr * dw
            self.bias-= self.lr* db

    def predict(self, X_test):
        y_predicted = np.dot(X_test,self.weights)+self.bias
        print(self.weights, self.bias)
        return y_predicted
    

ds=pd.read_csv(".\\archive\\Real estate.csv")

#print("The number of null values in the datasets is",ds.isnull().sum().sum())

#for i in range(1,7):
    #plt.scatter(ds.iloc[:,i], ds.iloc[:,7])
    #plt.xlabel(f"Column {i}")
    #plt.ylabel("Last Column")
    #plt.title(f"Figure {i}")
    #plt.show()

ds=ds.drop(columns=['No', 'X1 transaction date', 'X2 house age', 'X5 latitude' ,'X6 longitude']) 
print(ds)   

   


    

        

