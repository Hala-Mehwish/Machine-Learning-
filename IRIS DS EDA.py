#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")


# In[3]:


sns.get_dataset_names()


# In[12]:


iris = sns.load_dataset("iris")
iris.shape


# In[7]:


iris.head(5)


# In[8]:


iris.tail(5)


# In[9]:


iris.info()


# In[14]:


iris.describe()


# In[15]:


#sepals are usually longer than petals


# In[16]:


iris.species.unique()


# In[19]:


iris.species.value_counts()


# In[21]:


iris.species.value_counts(normalize=True)*100 


# In[22]:


iris.isnull().sum()


# In[23]:


iris.isnull().sum().sort_values(ascending=False)


# In[25]:


iris[iris.duplicated()]  #duplicated rows


# In[32]:


iris.duplicated().sum()


# In[28]:


#will show row that is same as 142 
iris[(iris.sepal_length==5.8) & (iris.sepal_width==2.7) & (iris.petal_length==5.1) & (iris.petal_width==1.9) & (iris.species=="virginica")]


# In[30]:


iris.drop_duplicates(inplace=True)


# In[33]:


iris.duplicated().sum()


# <h3>Data Visulization</h3>

# In[37]:


sns.countplot(iris.species)
plt.title("Species Count")
plt.show()


# In[41]:


iris.species.value_counts().plot(kind="bar", title="Species Count")


# In[42]:


sns.distplot(iris.sepal_length)
plt.show


# In[54]:


for i in iris.describe().columns:
    #print(i)
    sns.distplot(iris[i])
    plt.title(print("Histogram of", i))
    plt.show()
    print("===========================================================================================================")


# In[58]:


sns.scatterplot(iris.sepal_length, iris.sepal_width)
plt.show()


# In[59]:


sns.scatterplot(iris.sepal_length, iris.sepal_width, hue=iris.species)
plt.show()


# In[60]:


#setosa has smaller sepal_length but heigher sepal_width

#versicolor has almost same sepal length and width

#virginica has largest sepal_length and small sepal_width


# In[61]:


sns.scatterplot(iris.petal_length, iris.petal_width, hue=iris.species)
plt.show()


# In[62]:


#setosa has smallest petal length and width
#versicolor has medium petal length and width
#virginica has heighest petal length and width


# In[67]:


sns.pairplot(iris, hue='species')
plt.show()


# In[ ]:




