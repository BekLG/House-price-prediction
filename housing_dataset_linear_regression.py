# Importing Libraries
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns


# Importing USA Housing.csv
data = pd.read_csv('../input/USA_Housing.csv')
# The dataset can be found here: https://www.kaggle.com/datasets/kanths028/usa-housing

"""### EDA"""

data.head()

# Checking for Null Values
data.info()

"""### Data Preparation

1. There are no null values, so there is no need of deleting or replacing the data.
2. There is no necessity of having Address column/feature, so i am dropping it.
"""

# Dropping some Columns
data.drop(['Address','Avg. Area Number of Bedrooms','Area Population'],axis=1,inplace=True)

data.head()

"""#### Creating a Base Model"""

from sklearn import preprocessing
pre_process = preprocessing.StandardScaler()

# Putting feature variable to X
X = data[['Avg. Area Income','Avg. Area House Age','Avg. Area Number of Rooms']]

# Putting response variable to y
y = data['Price']

X = pd.DataFrame(pre_process.fit_transform(X))

X.head()

y.head()

#random_state is the seed used by the random number generator, it can be any integer.
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8 ,test_size = 0.2, random_state=1000)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

# Importing RFE and LinearRegression
from sklearn.linear_model import LinearRegression

# Representing LinearRegression as lr(Creating LinearRegression Object)
lm = LinearRegression()

# fit the model to the training data
lm.fit(X_train, y_train)

# print the intercept
print(lm.intercept_)

# Making predictions using the model
y_pred = lm.predict(X_test)

from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y_test, y_pred)
r_squared = r2_score(y_test, y_pred)

print('Mean_Squared_Error :' ,mse)
print('r_square_value :',r_squared)
