import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


def euclidean_distance(x1,x2):
    return np.sqrt(np.sum((x1-x2)**2))

class KNN(object):
    def __init__(self,k):
        self.k=k
    def fit(self,x_train,y_train):
        #covert to np array
        if isinstance(y_train, pd.Series):
            y_train = y_train.to_numpy()
        if isinstance(x_train, pd.DataFrame):
            x_train = x_train.to_numpy()    

        self.x_train = np.asarray(x_train)
        self.y_train = np.asarray(y_train)

    def _helper(self,x):
        prediction=[euclidean_distance(x,x1) for x1 in self.x_train]
        indices= np.argsort(prediction)[:self.k]
        labels= [self.y_train[i] for i in indices]
        c=Counter(labels).most_common()
        return c[0][0]
        
    def predict(self,x_test):
        predictions=[self._helper(x) for x in x_test]
        return np.array(predictions)
    

ds=pd.read_csv(".\\glass.csv")

X= ds[['Na','Mg','Al', 'Si', 'K', 'Ca', 'Ba', 'Fe']]
y= ds['Type']

# Ensure data is numerical
X = np.asarray(X)
y = np.asarray(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf=KNN(k=3)
clf.fit(X_train,y_train)
predictions=clf.predict(X_test)

print(accuracy_score(predictions,y_test))

# Initialize and train the sklearn KNeighborsClassifier model
knn_sklearn = KNeighborsClassifier(n_neighbors=3)
knn_sklearn.fit(X_train, y_train)

# Make predictions with the sklearn KNeighborsClassifier model
y_pred_sklearn = knn_sklearn.predict(X_test)

# Evaluate the sklearn KNeighborsClassifier model
print("Sklearn KNN Classification Report:")
print(classification_report(y_test, y_pred_sklearn, target_names=ds.Type_names))

print("Sklearn KNN Confusion Matrix:")
cm_sklearn = confusion_matrix(y_test, y_pred_sklearn)
print(cm_sklearn)

