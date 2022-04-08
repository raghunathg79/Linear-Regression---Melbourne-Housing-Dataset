#!/usr/bin/env python
# coding: utf-8

# Melb Data Frame Exercise

# In[449]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[450]:


df=pd.read_csv("melb_data.csv")


# # Data Preparation 

# # Pandas Creating a Data Frame

# In[451]:


df.head()


# # Variable = Price is a dependent variable

# In[452]:


df.tail()


# ['Address'	'Type'	'Method'	'SellerG'	'Date'	'Distance'	'Postcode' 'Car' 'YearBuilt''CouncilArea''Lattitude''Longtitude''Propertycount']
# 
# Few of the above variables may not be required for analysis hence, removing

# # Categorical variables ['Suburb','Address','Type','Method','SellerG','CouncilArea','Regionname']

# In[453]:


df.describe()


# In[454]:


df.info()


# In[455]:


df_new = df.drop(columns=['Address','Type', 'Method', 'SellerG', 'Date', 'Distance', 'Postcode', 'Car', 'YearBuilt','CouncilArea','Lattitude','Longtitude','Propertycount'])


# # missing values exist for Bedrooms, Bathroom, landsize,Building Area.

# In[457]:


df_new.info()


# In[458]:


df.hist('Bedroom2', bins=[0,2,4,6,8,10])


# In[459]:


df.hist('Bathroom', bins=[0,2,4,8,10,])


# In[460]:


df.hist('Landsize', bins=[0,50,100,200,300,500,600,700,800,1000,2000,3000,5000])


# In[461]:


df.hist('BuildingArea', bins=[0,100,200,300,400,500,1000])


# In[462]:


df.describe()


# # Univariate analysis based on individual Variables one by one

# In[463]:


df.boxplot('Price')


# In[464]:


df.hist('Price', bins=[100000,500000,1000000,1500000,2000000,10000000])


# In[465]:


df.boxplot('Rooms')


# In[466]:


df.hist('Rooms', bins=[0,2,4,6,10])


# # Observation - no. of houses are with 2 to 4 rooms are high

# In[467]:


df.hist('Landsize', bins=[5000,10000,15000,20000,50000])


# # Observation - no. of houses are bewteen 5000sq to 10000sq are high

# # Bi-Variate Analysis

# In[468]:


ax=df.plot.scatter(x='Landsize',y='Price')
ax.set_xlim(0,8000)
ax.set_ylim(0,7000000)


# In[469]:


df['Price'].fillna(df['Price'].median)


# In[470]:


ax=df.plot.scatter(x='Price',y='Bedroom2')
ax.set_xlim(0,7000000)
ax.set_ylim(0,10)


# In[471]:


h=df_new.corr()


# In[472]:


sns.heatmap(h,annot=True)


# In[473]:


df_new.info()


# # Linear Regression

# In[474]:


from sklearn.linear_model import LinearRegression


# In[475]:


LinearReg=LinearRegression()


# In[476]:


X=df[['Rooms']]
Y=df[['Price']]


# In[477]:


LinearReg.fit(X,Y)


# In[478]:


LinearReg.coef_


# In[479]:


X_test = np.arange(1,12, 1.5)


# In[480]:


X_test=X_test.reshape(-1,1)


# In[481]:


X_test


# In[482]:


Y_pred=LinearReg.predict(X_test)


# In[483]:


Y_pred


# In[484]:


plt.scatter(df["Rooms"],df["Price"])
plt.xlabel("Rooms", fontsize=24, color='red')
plt.ylabel("Price", fontsize=24, color='green')
plt.scatter(X_test,Y_pred)
plt.plot(X_test,Y_pred)
fig= plt.figure(figsize=(14,7))


# In[485]:


df['New_Price'] = df['Price'] + 2*df['Rooms']*df['Rooms']*1000000


# In[486]:


df.head()


# In[448]:


plt.xlabel("Rooms", fontsize=24, color='red')
plt.ylabel("Price", fontsize=24, color='green')
plt.scatter(df["Rooms"],df["Price"])
plt.scatter(df["Rooms"],df["New_Price"])


# In[ ]:





# In[ ]:





# In[ ]:




