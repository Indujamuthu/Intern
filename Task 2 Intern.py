#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score

get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


wine = pd.read_csv("C:/Users/hp/Downloads/winequality-red.csv")
wine


# In[4]:


wine.head()


# In[5]:


wine.info()


# In[6]:


wine.describe()


# In[7]:


wine.columns


# In[8]:


fig = plt.figure(figsize=(15,10))

plt.subplot(3,4,1)
sns.barplot(x='quality',y='fixed acidity',data=wine)


# In[9]:


fig = plt.figure(figsize=(15,10))

plt.subplot(3,4,2)
sns.barplot(x='quality',y='volatile acidity',data=wine)


# In[10]:


plt.subplot(3,4,11)
sns.barplot(x='quality',y='alcohol',data=wine)


# In[11]:


wine['quality'].value_counts()


# In[12]:


ranges = (2,6.5,8) 
groups = ['bad','good']
wine['quality'] = pd.cut(wine['quality'],bins=ranges,labels=groups)


# In[14]:


le = LabelEncoder()
wine['quality'] = le.fit_transform(wine['quality'])


# In[15]:


wine.head()


# In[16]:


wine['quality'].value_counts()


# In[17]:


good_quality = wine[wine['quality']==1]
bad_quality = wine[wine['quality']==0]


# In[18]:


bad_quality = bad_quality.sample(frac=1)
bad_quality = bad_quality[:217]


# In[19]:


new_df = pd.concat([good_quality,bad_quality])
new_df = new_df.sample(frac=1)
new_df


# In[20]:


new_df['quality'].value_counts()


# In[21]:


new_df.corr()['quality'].sort_values(ascending=False)


# In[22]:


from sklearn.model_selection import train_test_split

X = new_df.drop('quality',axis=1) 
y = new_df['quality']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# In[23]:


X_train


# In[24]:


X_test


# In[25]:


y_train


# In[26]:


y_test


# In[27]:


param = {'n_estimators':[100,200,300,400,500,600,700,800,900,1000]}

grid_rf = GridSearchCV(RandomForestClassifier(),param,scoring='accuracy',cv=10,)
grid_rf.fit(X_train, y_train)

print('Best parameters --> ', grid_rf.best_params_)


# In[29]:


# Wine Quality Prediction
pred = grid_rf.predict(X_test)

print(confusion_matrix(y_test,pred))
print('\n')
print(classification_report(y_test,pred))
print('\n')
print(accuracy_score(y_test,pred))


# In[ ]:




