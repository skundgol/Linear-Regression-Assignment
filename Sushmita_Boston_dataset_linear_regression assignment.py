#!/usr/bin/env python
# coding: utf-8

# In[5]:


#import data and libraries

import numpy as np
import matplotlib.pyplot as plt 

import pandas as pd  
import seaborn as sns 

get_ipython().run_line_magic('matplotlib', 'inline')


# In[24]:


#import data from scikit libraries

from sklearn.datasets import load_boston
boston_dataset = load_boston()


# In[25]:


print(boston_dataset.keys())#check what data set contains


# In[26]:


boston_dataset.DESCR


# In[27]:


boston = pd.DataFrame(boston_dataset.data, columns=boston_dataset.feature_names)#load data into pandas dataframe
boston.head()


# In[28]:


boston['MEDV'] = boston_dataset.target#add new column of target values


# In[29]:


boston.isnull().sum()#data preprocessing check for null values


# In[36]:



plt.figure(figsize=(20, 5))

features = ['LSTAT', 'RM']
target = boston['MEDV']

for i, col in enumerate(features):
    plt.subplot(1, len(features) , i+1)
    x = boston[col]
    y = target
    plt.scatter(x, y, marker='o')
    plt.title(col)
    plt.xlabel(col)
    plt.ylabel('MEDV')


# In[37]:


#preparing data for training model

X = pd.DataFrame(np.c_[boston['LSTAT'], boston['RM']], columns = ['LSTAT','RM'])
Y = boston['MEDV']


# In[38]:


#split data into training and testing sets
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=5)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)


# In[39]:


#scikit-learn’s LinearRegression to train our model on both the training and test sets.

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

lin_model = LinearRegression()
lin_model.fit(X_train, Y_train)


# In[41]:


# model evaluation for training set
from sklearn.metrics import r2_score
y_train_predict = lin_model.predict(X_train)
rmse = (np.sqrt(mean_squared_error(Y_train, y_train_predict)))
r2 = r2_score(Y_train, y_train_predict)

print("The model performance for training set")
print("--------------------------------------")
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))
print("\n")

# model evaluation for testing set
y_test_predict = lin_model.predict(X_test)
rmse = (np.sqrt(mean_squared_error(Y_test, y_test_predict)))
r2 = r2_score(Y_test, y_test_predict)

print("The model performance for testing set")
print("--------------------------------------")
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))


# In[ ]:




