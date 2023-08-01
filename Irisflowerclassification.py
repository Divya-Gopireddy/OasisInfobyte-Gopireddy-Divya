import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
data=pd.read_csv("d:/OasisInfobyte/Iris.csv")
data.head()
"""Structure of the Data"""
data.shape
"""Central Tendency of the Data"""
data.describe()
data.info()
data.columns
"""Missing Values"""
data.isnull().sum()
"""It is clear the there is no missing values in the dataset"""
data['Species']
data.Species.value_counts()
"""Training and Testing the data"""
X= data.iloc[:,1:5]
print(X)
y=data.iloc[:,5:]
print(y)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print("X_train: ",X_train.shape)
print("X_test:  ",X_test.shape)
print("y_train: ",y_train.shape)
print("y_test:  ",y_test.shape)
from sklearn.linear_model import LogisticRegression
regressor = LogisticRegression()
regressor.fit(X,y)
regressor.fit(X_train,y_train)
predictions = regressor.predict(X)
y_pred=regressor.predict(X_test)
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,y_pred))
from sklearn import tree
classifier=tree.DecisionTreeClassifier()
classifier.fit(X_train,y_train)
predictions=classifier.predict(X_test)
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,predictions))
