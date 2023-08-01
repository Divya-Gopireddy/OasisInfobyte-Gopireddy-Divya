import numpy as np
import pandas as pd

import calendar
from datetime import datetime

"""Load the dataset and assign it to the variable data"""

data=pd.read_csv("d:/OasisInfobyte/UnemploymentinIndia.csv")
data.head()

"""Preliminary analysis"""

data.info()

data.describe()

data.shape

data.columns

data.isnull().sum()

"""The dataset contains null values so use any of the methods to treat the null values. Here I'm using drop() method."""

data=data.dropna()
data.isnull().sum()

"""Now, you can observe that the data contains no null values."""

data.shape

data.dtypes

"""EDA

Barplot

Import the plotly.express module for the visualization. Normally we can use any other python modules for visualization such as matplotlib.pylot and seaborn
"""

import plotly.express as px

"""We are using animation to show the bar plot as per date attribute."""

plot1= px.bar(data,x='Region',y=' Estimated Unemployment Rate (%)',color='Region',title='Unemployment rate',animation_frame=' Date',template='plotly')
plot1.show()

"""* This is awesome.
I'm wondering that I am a good learner now.

Scatterplot

To show the relationship, scatterplot plays a main role to show the relatioship in such a way that everyone can understand the concept.
"""

plot2= px.scatter(data,x='Region',y=' Estimated Unemployment Rate (%)',color='Region',title='Unemployment rate',animation_frame=' Date',template='plotly')
plot2.show()

data.Region.unique()

data.columns

data.Area.unique()

"""* Upto now we used visualization to show the unemployment in each State by the Region attribute.
* Now, we are about to know ---- Employment in each state by Region
"""

plot3=px.bar(data,x=' Estimated Employed',y='Region',title='Estimated Employment by Region',color='Region')
plot3.update_layout(xaxis={'categoryorder':'total ascending'})

plot3.show()

"""As we can see that Uttarpradesh is the leading state for Employees.
What about the people who are unemployed belogs to which Area????

Let's get started ...!
"""

unemployment = data.groupby(['Region','Area'])[' Estimated Unemployment Rate (%)'].mean().reset_index()

plot4= px.sunburst(unemployment, path=['Region','Area'], values=' Estimated Unemployment Rate (%)',
                  title= 'Unemployment Rate in Every State and Area', height=750)
plot4.show()

"""* As per our Sunbrust plot we can say that the most unemployed people belongs to Rural Area. How sad the have to move to Urban Area in order to get Employment 😢

# Conclusion

This all about the Umployment Project.... I hope the Visualzation part is clear enough to understand the variations in the unemployment according to the States.
Here is the simple explanation....
* From Barplot we can observe that here is huge difference in the unemployment in
Tripura,Haryana,Pducherry and jharkhand.
* In 31-04-20 we can observe the high unemployment in every state.
* Uttarpradesh, Maharastra and westbengal are the highly employed States.
* when we consider the 'Area' that means Rural & urban.... Urban Area is the highly employed Area compared to Rural Area.

I hope you all like it 😀 .....
"""
