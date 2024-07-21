from sklearn.datasets import load_breast_cancer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error


def sigmoid(z):
    z1 = np.clip(z, -700, 700)
    return 1 / (1 + np.exp(-z1))
    

def sigmoid_derivative(z):
    return (sigmoid(z)*(1-sigmoid(z)))

class LogisticRegression2:
    def __init__(self, learning_rate, epochs):
      #Initialise the hyperparameters of the model
        self.lr = learning_rate
        self.epochs = epochs

    def fit(self, X, y):

        y = y.reshape(-1, 1)
        n_samples, n_features = X_train.shape

        self.weights = np.random.randn(n_features,1)/np.sqrt(n_features)
        self.bias = np.random.randn(1,1)

        #Implement the GD algortihm
        for i in range(self.epochs):
            z = np.dot(X,self.weights) + self.bias
            y_pred = sigmoid(z)

            dw = -np.dot(X.T,(y - y_pred))/n_samples
            db = -np.sum(y - y_pred)/n_samples

            self.weights -= self.lr* dw
            self.bias-= self.lr* db

    def predict(self, X):
      #Write the predict function
        y_pred = np.dot(X,self.weights)+self.bias

        for i in range(len(y_pred)):
            if y_pred[i]<= 0.5:
                y_pred[i] = 0
            else:
                y_pred[i] = 1
        return y_pred
    

data=load_breast_cancer()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

norm = MinMaxScaler()
X_train_normal = norm.fit_transform(X_train)

X_test_normal = norm.transform(X_test)

X_train_normal_ds = pd.DataFrame(X_train_normal)
X_test_normal_ds = pd.DataFrame(X_test_normal)


# Self LogisticRegression
logreg = LogisticRegression2(learning_rate=0.05, epochs=100)#0.005,10
logreg.fit(X_train, y_train)
y_pred2 = logreg.predict(X_test)

mse_test_self = mean_squared_error(y_test, y_pred2)
r2_self = r2_score(y_test, y_pred2)

print(mse_test_self)
print(r2_self)

#LogisticRegression function
logr=LogisticRegression()
logr.fit(X_train, y_train)
y_pred = logr.predict(X_test)

linear_predictions = logr.predict(X_test_normal)
linear_mse = mean_squared_error(y_test, linear_predictions)

linear_r2 = r2_score(y_test, linear_predictions)

#print(linear_mse)
#print(linear_r2)





