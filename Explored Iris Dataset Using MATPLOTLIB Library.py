#!/usr/bin/env python
# coding: utf-8

# # Data Visualization - Matplotlib

# In[1]:


import numpy as np
import pandas as pd
from sklearn import datasets   #'sklearn' has datasets already #
import matplotlib.pyplot as plt


from matplotlib import markers
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
get_ipython().run_line_magic('matplotlib', 'inline')


# ### Import Dataset

# In[2]:


#Copied#

iris = datasets.load_iris()   #Get a dataset from 'sklearn'#
df = pd.DataFrame(iris['data'], columns=iris['feature_names'])
df['target'] = iris['target']

df.head()


# In[3]:


iris = datasets.load_iris() 

df= pd.DataFrame(data= np.c_[iris['data'], iris['target']], 
                 columns= iris['feature_names'] + ['target'])
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

df.head()


# ### 1. Scatter Plot

# ##### 1.1 Basic Plot

# In[4]:


plt.plot("sepal length (cm)", "sepal width (cm)", data=df, linestyle="none", marker="o")


# In[5]:


plt.plot("sepal length (cm)", "sepal width (cm)", data=df, linestyle="none", marker="o")
plt.xlabel("sepal length (cm)")
plt.ylabel("sepal width (cm)")


# In[6]:


plt.plot("sepal length (cm)", "sepal width (cm)", data=df, linestyle="none", marker="o")
plt.xlabel("sepal length (cm)")
plt.ylabel("sepal width (cm)")
plt.title("Relationship Between Sepal Length & Width")


# In[7]:


# There is no relationship between sepal lenght and width

plt.plot("sepal length (cm)", "sepal width (cm)", data=df, linestyle="none", marker="x")
plt.xlabel("sepal length (cm)")
plt.ylabel("sepal width (cm)")
plt.title("Relationship Between Sepal Length & Width")


# In[8]:


# There is a weak co-relation between sepal length & petal length:

plt.plot("sepal length (cm)", "petal length (cm)", data=df, linestyle="none", marker="x")


# In[4]:


#No Co-relation#

plt.plot("sepal length (cm)", "sepal width (cm)", data=df, linestyle="none", marker="o")


# In[7]:


#Weak Co-relation#

plt.plot("petal length (cm)", "petal width (cm)", data=df, linestyle="none", marker="o")


# ### 2. 2D Density Plot

# In[8]:


## Sub-plot##

# Create a figure with 6 plot areas
fig, axes = plt.subplots(ncols=3, nrows=1, figsize=(21, 5))


#The figure statrs with a 'Scatter Plot':
axes[0].set_title("Scatterplot")
axes[0].plot(df["sepal length (cm)"], df["sepal width (cm)"], "ko")


#Same data through 'Hexbins':    #Use for large data#
nbins=20
axes[1].set_title("Hexbin")
axes[1].hexbin(df["sepal length (cm)"], df["sepal width (cm)"], gridsize=nbins)

# 2D Histogram    #Use for large data#
axes[2].set_title("2D Histogram")
axes[2].hist2d(df["sepal length (cm)"], df["sepal width (cm)"], bins=nbins)

fig.show()


# ### 3. Box Plot

# ##### 3.1 Basic Plot

# In[26]:


## Sepal length:
    # Has no outliers 
    # Positively Skewed: Median closest to the min
    # Not Symetric: Whiskers are not equal


plt.boxplot(df["sepal length (cm)"])
plt.ylabel("sepal length (cm)")
plt.title("Sepal Length Box Plot")


# In[27]:


## Sepal Width:
    # Has outliers 
    # Positively Skewed: Median closest to the min
    # Not Symetric: Whiskers are not equal


plt.boxplot(df["sepal width (cm)"])
plt.ylabel("sepal width (cm)")
plt.title("Sepal Width Box Plot")


# In[29]:


## Petal length:
    # Has no outliers 
    # Negatively Skewed: Median closest to the max
    # Not Symetric: Whiskers are not equal


plt.boxplot(df["petal length (cm)"])
plt.ylabel("petal length (cm)")
plt.title("Petal Length Box Plot")


# In[30]:


## Petal Width:
    # Has no outliers 
    # Negatively Skewed: Median closest to the max
    # Not Symetric: Whiskers are not equal


plt.boxplot(df["petal width (cm)"])
plt.ylabel("petal width (cm)")
plt.title("Petal Width Box Plot")


# In[31]:


## See all the plots on the same figure:

plt.boxplot([df["sepal length (cm)"], df["sepal width (cm)"], df["petal length (cm)"], df["petal width (cm)"]],
           labels=["Sepal Vength", "Sepal Width", "Petal Length", "Petal Width"])

plt.show()


# In[50]:


## See all the plots on the same figure:
# Change Color

plt.boxplot([df["sepal length (cm)"], df["sepal width (cm)"], df["petal length (cm)"], df["petal width (cm)"]],
           notch=True, patch_artist=True, labels=["Sepal Vength", "Sepal Width", "Petal Length", "Petal Width"])
colors = ["blue", "green", "purple", "tan"]

for patch, color in zip(box["boxes"], colors):
    patch.set_facecolor(color)

plt.show()


# ### 4. Violin Plot

# ##### 4.1 Basic Plot

# In[34]:


print(df["sepal length (cm)"].describe())

plt.violinplot(df["sepal length (cm)"])
plt.ylabel("Sepal Length")


# In[37]:


# Show Mean, Extreme & Median:

print(df["sepal length (cm)"].describe())

plt.violinplot(df["sepal length (cm)"], showmeans=True, showextrema=True, showmedians=True)
plt.ylabel("Sepal Length")


# In[38]:


# Horizontal Violin:


plt.violinplot(df["sepal length (cm)"], showmeans=True, showextrema=True, showmedians=True, vert=False)
plt.ylabel("Sepal Length")


# In[45]:


# Multible Violins on 1 Figure:

fig, ax = plt.subplots(figsize=(10,10))

ax.violinplot([df["sepal length (cm)"], df["sepal width (cm)"], df["petal length (cm)"], df["petal width (cm)"]])
ax.set_xticks([1,2,3,4])
ax.set_xticklabels(["Petal Width", "Petal Length", "Sepal Width", "Sepal Length"])


# ### 5. 1D Historam

# In[52]:


plt.hist(df["sepal length (cm)"], 16)   #16 is the number of bins#
plt.title("Sepal Length Histogram")

plt.show()


# In[55]:


plt.hist(df["sepal length (cm)"], 16, facecolor="green", alpha=0.4)   #Alpha is transperancy - [0-1]
plt.title("Sepal Length Histogram")

plt.show()


# ## 6. 3D Scatterplot

# In[9]:


fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

ax.scatter(df["sepal width (cm)"], df["sepal length (cm)"], df["petal width (cm)"], c="skyblue", s=50,)  #'s' is the marker's size#
ax.view_init(30,185)
            
plt.show()


# In[10]:


fig = plt.figure()
ax1 = fig.add_subplot(111, projection="3d")

#Define Axes#
dx = np.ones(len(df["sepal width (cm)"]))
dy = np.ones(len(df["sepal length (cm)"]))
dz = np.ones(len(df["petal width (cm)"]))

ax1.bar3d(df["sepal width (cm)"], df["sepal length (cm)"], df["petal width (cm)"], dx, dy, dz, color="purple", alpha=0.5)

plt.show()


# ## 7. 3D Surface Plot

# In[11]:


fig = plt.figure()
ax = fig.gca(projection="3d")

ax.plot_trisurf(df["sepal width (cm)"], df["sepal length (cm)"], df["petal width (cm)"], cmap=plt.cm.viridis)

plt.show()


# In[78]:


fig = plt.figure()
ax = fig.gca(projection="3d")

surf=ax.plot_trisurf(df["sepal width (cm)"], df["sepal length (cm)"], df["petal width (cm)"], cmap=plt.cm.viridis)

fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()


# In[ ]:




