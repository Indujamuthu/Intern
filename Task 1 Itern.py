#!/usr/bin/env python
# coding: utf-8

# In[26]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error


# In[27]:


df = pd.read_csv("C:/Users/hp/Downloads/USA_Housing.csv")
df


# In[28]:


df.info()


# In[29]:


df.describe()


# In[30]:


import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# In[31]:


df.head()


# In[32]:


df.isnull().sum()


# In[33]:


correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()


# In[34]:


df.columns


# In[36]:


# Preprocessing: Selecting features and target variable
X = df[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
       'Avg. Area Number of Bedrooms', 'Area Population']]
y = df['Price']


# In[37]:


X


# In[38]:


y


# In[39]:


# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[40]:


X_train


# In[41]:


X_test


# In[42]:


y_train


# In[43]:


y_test


# In[51]:


from sklearn import metrics


# In[54]:


from sklearn.linear_model import LinearRegression


# In[55]:


lm = LinearRegression()


# In[56]:


lm.fit(X_train,y_train)


# In[57]:


# print the intercept
print(lm.intercept_)


# In[58]:


coeff_df = pd.DataFrame(lm.coef_,X.columns,columns=['Coefficient'])
coeff_df


# In[59]:


#Predictions from our Model
predictions = lm.predict(X_test)


# In[60]:


plt.scatter(y_test,predictions)


# In[61]:


#Residual Histogram
sns.distplot((y_test-predictions),bins=50);


# In[62]:


#Regression Evaluation Metrics
from sklearn import metrics


# In[63]:


print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))


# In[ ]:




