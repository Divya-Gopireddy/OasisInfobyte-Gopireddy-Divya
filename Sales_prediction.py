import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

"""Load the Dataset and assign the dataset to a variable data"""

data=pd.read_csv("D:/OasisInfobyte/Advertising.csv")
data.head()

"""Preliminary Analysis"""

data.shape

data.describe()

data.columns

data=data.drop(columns=['Unnamed: 0'])

data.columns

data.info()

"""Check null values"""

data.isnull().sum()

"""No Null Values

Data Visualization
"""

import seaborn as sns
plot1=plt.figure(figsize=(4,4))
plot1=sns.scatterplot(data=data,x=data['TV'],y=data['Sales'],color='#fc8803')
plt.show()

plot3=plt.figure(figsize=(4,4))
plot3=sns.scatterplot(data=data,x=data['Radio'],y=data['Sales'])
plt.show()

plot2=plt.figure(figsize=(4,4))
plot2=sns.scatterplot(data=data,x=data['Newspaper'],y=data['Sales'],color='#d203fc')
plt.show()

"""Spliting the data"""

data.columns

X=data.iloc[:,:3]
X.head()

y=data.iloc[:,3:]
y.head()

from sklearn.model_selection import train_test_split

X_train, X_test, y_train,y_test = train_test_split(X,y,test_size = 0.20, random_state = 0)

from sklearn.linear_model import LinearRegression

model= LinearRegression()

model.fit(X_train,y_train)

#predictions
y_pred=model.predict(X_test)
y_pred

from sklearn import metrics

print('MAE:',metrics.mean_absolute_error(y_pred,y_test))

from sklearn.metrics import r2_score

r2_score(y_test,y_pred)

plt.scatter(y_test,y_pred,c='#f54275')
plt.show()
