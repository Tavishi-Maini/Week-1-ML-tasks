import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


class LinearRegression:
    def __init__(self, learning_rate, epochs):
        self.lr = learning_rate
        self.epochs = epochs

    def fit(self, X_train, y_train):
        #covert to np array
        if isinstance(y_train, pd.Series):
            y_train = y_train.to_numpy()

        n_samples, n_features = X_train.shape
        
        # Ensure y_train is a column vector
        if y_train.ndim == 1:
            y_train = y_train.reshape(-1, 1)
        
        # Initialize parameters
        self.weights = np.zeros((n_features, 1))
        self.bias = np.zeros((1, 1))

        # Gradient descent
        for _ in range(self.epochs):
            # Linear prediction
            y_predicted = np.dot(X_train, self.weights) + self.bias
            
            # Calculate gradients
            dw = (1 / n_samples) * np.dot(X_train.T, (y_predicted - y_train))
            db = (1 / n_samples) * np.sum(y_predicted - y_train)
            
            # Update weights and biases
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X_test):
        y_predicted = np.dot(X_test, self.weights) + self.bias
        return y_predicted


    

ds=pd.read_csv(".\\archive\\Real estate.csv")

#print("The number of null values in the datasets is",ds.isnull().sum().sum())

#for i in range(1,7):
    #plt.scatter(ds.iloc[:,i], ds.iloc[:,7])
    #plt.xlabel(f"Column {i}")
    #plt.ylabel("Last Column")
    #plt.title(f"Figure {i}")
    #plt.show()

ds=ds.drop(columns=['No', 'X1 transaction date', 'X5 latitude' ,'X6 longitude']) 

X= ds[['X2 house age', 'X3 distance to the nearest MRT station' ,'X4 number of convenience stores']]
y=ds['Y house price of unit area']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

norm = MinMaxScaler()
X_train_normal = norm.fit_transform(X_train)

X_test_normal = norm.transform(X_test)

X_train_normal_ds = pd.DataFrame(X_train_normal, columns=X.columns)
X_test_normal_ds = pd.DataFrame(X_test_normal, columns=X.columns)



lr_model = LinearRegression(learning_rate=0.001, epochs=850)
lr_model.fit(X_train_normal, y_train.to_numpy())

y_predicted = lr_model.predict(X_test_normal)

mse_train = mean_squared_error(y_train, lr_model.predict(X_train_normal))
mse_test = mean_squared_error(y_test, y_predicted)
r2 = r2_score(y_test, y_predicted)

print(mse_train)
print(mse_test)
print(r2)

    

        

