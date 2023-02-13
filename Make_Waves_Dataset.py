#!/usr/bin/env python
# coding: utf-8

# #### Kaggle Dataset Tsunami Causes and Waves
# 

# In[2]:


import mglearn


# In[12]:


from sklearn.neighbors import KNeighborsRegressor
X, y = mglearn.datasets.make_wave(n_samples=40)


# In[17]:


from sklearn.model_selection import train_test_split
# split the wave dataset into a training and a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)


# In[18]:


# instantiate the model and set the number of neighbors to consider to 3
reg = KNeighborsRegressor(n_neighbors=3)
# fit the model using the training data and training targets
reg.fit(X_train, y_train)


# In[19]:


print("Test set predictions:\n{}".format(reg.predict(X_test)))


# We can also evaluate the model using the score method, which for regressors returns
# the R^2 score. The R^2
#  score, also known as the coefficient of determination, is a meas‐
# ure of goodness of a prediction for a regression model, and yields a score between 0
# and 1. A value of 1 corresponds to a perfect prediction, and a value of 0 corresponds
# to a constant model that just predicts the mean of the training set responses, y_train:
# 

# In[20]:


print("Test set R^2: {:.2f}".format(reg.score(X_test, y_test)))


# #### Analyzing KNeighborsRegressor
# 

# In[23]:


from matplotlib import pyplot as plt
import numpy as np

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
# create 1,000 data points, evenly spaced between -3 and 3
line = np.linspace(-3, 3, 1000).reshape(-1, 1)
for n_neighbors, ax in zip([1, 3, 9], axes):
    
    # make predictions using 1, 3, or 9 neighbors
    reg = KNeighborsRegressor(n_neighbors=n_neighbors)
    reg.fit(X_train, y_train)
    ax.plot(line, reg.predict(line))
    ax.plot(X_train, y_train, '^', c=mglearn.cm2(0), markersize=8)
    ax.plot(X_test, y_test, 'v', c=mglearn.cm2(1), markersize=8)
    ax.set_title("{} neighbor(s)\n train score: {:.2f} test score: {:.2f}".format(
         n_neighbors, reg.score(X_train, y_train),
         reg.score(X_test, y_test)))
    ax.set_xlabel("Feature")
    ax.set_ylabel("Target")
axes[0].legend(["Model predictions", "Training data/target","Test data/target"], loc="best")


# In[ ]:




