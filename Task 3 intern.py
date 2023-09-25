#!/usr/bin/env python
# coding: utf-8

# In[13]:


#importing necessary libraries
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[14]:


df = pd.read_csv ("C:/Users/hp/Downloads/Iris.csv")
df


# In[15]:


df.head()


# In[16]:


df.tail()


# In[17]:


df.describe()


# In[18]:


df.shape


# In[19]:


df.isnull().sum()


# In[20]:


count =  df.Species.value_counts()
print(count)


# In[21]:


lab = df.Species.unique().tolist()
lab


# In[22]:


#DATA VISUALIZATION
plt.pie(count,labels=lab)
plt.title("Count of Species",fontsize=20)
plt.show()


# In[24]:


plt.subplots(figsize=(7,7))
sns.scatterplot(x="SepalLengthCm",y="SepalWidthCm",data=df,hue="Species")
plt.show()


# In[26]:


plt.subplots(figsize=(7,7))
sns.scatterplot(x="PetalLengthCm",y="PetalWidthCm",data=df,hue="Species")
plt.show()


# In[27]:


#Relation of all feature with each other
data1 = df.drop("Id",axis=1)
plot=sns.pairplot(data1,hue="Species",diag_kind="hist")
plot.fig.suptitle("Relation of all feature with each other",y=1.1,fontsize=20)
plt.show()


# In[28]:


#MODEL BUILD(REGRESSION)
from sklearn.model_selection import train_test_split


# In[30]:


X = df.drop(["Species","Id"],axis=1)
X


# In[32]:


Y = df["Species"]
Y


# In[33]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler


# In[34]:


model = LogisticRegression(max_iter=1000)


# In[35]:


x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.3,random_state=1)
model.fit(x_train,y_train)


# In[36]:


predictions = model.predict(x_test)


# In[37]:


print(classification_report(y_test,predictions))


# In[38]:


print("Confusion Matrix\n",confusion_matrix(y_test,predictions))


# In[39]:


#The accuracy of the model 
print(accuracy_score(y_test,predictions))


# In[ ]:




