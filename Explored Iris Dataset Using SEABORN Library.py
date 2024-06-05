#!/usr/bin/env python
# coding: utf-8

# # Data Visualization - Seaborn

# # Load DataSet

# In[1]:


import numpy as np
import pandas as pd
from sklearn import datasets
import seaborn as sns


# In[2]:


iris = datasets.load_iris() 

df= pd.DataFrame(data= np.c_[iris['data'], iris['target']], 
                 columns= iris['feature_names'] + ['target'])
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

df.head()


# ## 1. Pair Plot

# In[3]:


# Pair plot works with neumerical data only:

sns.pairplot(df)


# In[6]:


sns.pairplot(df, diag_kind="kde")


# In[8]:


# Assign color accorting to categories - - which are not neumerical

sns.pairplot(df, hue="species")


# In[9]:


sns.pairplot(df, diag_kind="hist", hue="species")


# In[4]:


sns.pairplot(df, hue="species", palette="husl")


# In[11]:


sns.pairplot(df, hue="species", markers=["o", "s", "d"])


# In[13]:


#Regression:

sns.pairplot(df, kind="reg")


# ## 2. Heatmap

# In[5]:


#It defines the correlations: [-1, 0, 1]

corr_df = df[["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"]].corr(method="pearson")
sns.heatmap(corr_df, annot=True)


# In[16]:


sns.heatmap(corr_df, annot=True, linewidth=2, linecolor="yellow")


# In[17]:


sns.heatmap(corr_df, annot=True, cmap="PiYG")


# ## 3. Density Plot

# In[18]:


# Only takes neumericals:

sns.kdeplot(df["sepal width (cm)"])


# In[19]:


sns.kdeplot(df["sepal width (cm)"], shade=True)


# In[20]:


sns.kdeplot(df["sepal width (cm)"], shade=True, color="yellow")


# In[22]:


sns.kdeplot(df["sepal width (cm)"], shade=True, vertical=True, color="skyblue")


# ## 4. Bar Plot

# In[3]:


iris = datasets.load_iris() 

df= pd.DataFrame(data= np.c_[iris['data'], iris['target']], 
                 columns= iris['feature_names'] + ['target'])
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

df.head()


# In[4]:


sns.barplot(x="species", y="sepal width (cm)", data=df)


# In[5]:


sns.barplot(x="species", y="sepal length (cm)", data=df)


# In[7]:


# The number of 'sepal length' for each species:


sns.barplot(x="species", y="petal width (cm)", data=df)


# ## 5. Catplot

# In[8]:


sns.catplot(x="species", y="sepal width (cm)", data=df)


# In[9]:


sns.catplot(x="species", y="petal length (cm)", data=df)


# In[11]:


# Can be box plot: Reprpesent both categorical/ neumerical data

sns.catplot(x="species", y="sepal width (cm)", kind="box", data=df)


# In[12]:


sns.catplot(x="species", y="sepal width (cm)", kind="violin", data=df)


# In[14]:


sns.catplot(x="species", y="sepal width (cm)", kind="violin", inner=None, data=df)


# In[15]:


sns.catplot(y="sepal width (cm)", kind="count", palette="pastel", edgecolor="0.6", data=df)


# In[17]:


#Shows the average#

sns.catplot(x="species", y="sepal width (cm)", kind="point", data=df)


# ## 6. Factor Plot

# In[18]:


iris = datasets.load_iris() 

df= pd.DataFrame(data= np.c_[iris['data'], iris['target']], 
                 columns= iris['feature_names'] + ['target'])
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

df.head()


# In[19]:


g = sns.factorplot(x="sepal width (cm)",
                  y="sepal length (cm)",
                  data=df,
                  hue="species",
                  col="species",
                  kind="swarm")    #Swarmplot#


# In[6]:


g = sns.factorplot(x="petal width (cm)",
                  y="petal length (cm)",
                  data=df,
                  hue="species",
                  col="species",
                  kind="swarm")


# ## 7. Joint Distribution Plot

# In[7]:


iris = datasets.load_iris() 

df= pd.DataFrame(data= np.c_[iris['data'], iris['target']], 
                 columns= iris['feature_names'] + ['target'])
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

df.head()


# In[25]:


# Scatter and histogram together:

sns.jointplot(x="sepal width (cm)", y="sepal length (cm)", data=df)


# In[24]:


sns.jointplot(x="petal width (cm)", y="petal length (cm)", data=df)


# #### 7.1 Customization plot

# In[27]:


sns.jointplot(x="sepal width (cm)", y="sepal length (cm)", kind="kde", data=df)


# In[30]:


#Recognise how they are similer -- No need to joint usually#

(sns.jointplot(x="sepal width (cm)", y="sepal length (cm)", 
               data=df, color="k")
 .plot_joint(sns.kdeplot, zorder=0, n_levels=6))


# In[ ]:




