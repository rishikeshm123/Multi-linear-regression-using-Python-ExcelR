#!/usr/bin/env python
# coding: utf-8

# ## Multi Linear Regression
# 

# Prepare a prediction model for profit of 50_startups data.
# Do transformations for getting better predictions of profit and
# make a table containing R^2 value for each prepared model.
# 
# R&D Spend -- Research and develop spend in the past few years</br>
# Administration -- spend on administration in the past few years</br>
# Marketing Spend -- spend on Marketing in the past few years</br>
# State -- states from which data is collected</br>
# Profit  -- profit of each state in the past few years</br>

# In[1]:


import pandas as pd 
import scipy.stats as stats
import seaborn as sns
from statsmodels.graphics.regressionplots import influence_plot
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import statsmodels.api as sm
import numpy as np


# #### Importing Dataset, EDA and Visualisation

# In[3]:


#read the dataset
data = pd.read_csv('50_startups.csv')
data.head()


# In[4]:


data.shape


# In[5]:


data[data.duplicated()]


# In[42]:


#check for missing values
data.isna().sum()


# In[6]:


data.info()


# In[7]:


data.describe()


# ### Correlation Matrix

# In[8]:


data.corr()


# ### Scatterplot between variables along with histograms

# In[9]:


sns.pairplot(data)


# ## Model Building

# In[13]:


data =data.rename({'R&D Spend':'RNDspend','Marketing Spend':'MarketingSpend'},axis =1)
data.head()


# In[14]:


model1= smf.ols('Profit ~ RNDspend + Administration + MarketingSpend',data=data).fit()
model1.summary()


# In[15]:


(model1.rsquared, model1.aic)


# #### As we can see p values for Administration and marketing spend do not satisfy ie. they are greater than 0.05 alpha value.

# ## Creating Simple Linear Regression Models

# In[16]:


#Simple linear regression model for Profit vs Administration
m1_adm = smf.ols('Profit~Administration',data= data).fit()
m1_adm.summary()


# #### Both P-value and Rsquared values are not satisfactory.

# In[17]:


#Simple linear regression model for Profit vs Marketing Spend
m1_mark = smf.ols('Profit~MarketingSpend',data = data).fit()
m1_mark.summary()


# #### P-value has improved but Rsquared values are not satisfactory.

# #### Creating Multi linear regression model using Marketing spend and administration vs Proft

# In[18]:


#Marketing spend + Administration vs Profit
m1_MVSA = smf.ols('Profit ~ Administration + MarketingSpend',data = data).fit()
m1_MVSA.summary()


# #### Both Pvalue are now satisfactory and Rsquared  Values have also been improved.

# ## Calculating VIF 

# In[19]:


rsq_rd = smf.ols('RNDspend ~ Administration + MarketingSpend + Profit',data=data).fit().rsquared
vif_rd = 1/(1-rsq_rd)

rsq_ad = smf.ols('Administration ~ RNDspend + MarketingSpend + Profit',data=data).fit().rsquared
vif_ad = 1/(1-rsq_ad)

rsq_ms = smf.ols('MarketingSpend ~ RNDspend + Administration + Profit',data=data).fit().rsquared
vif_ms = 1/(1-rsq_ms)

rsq_pr = smf.ols('Profit ~ RNDspend + Administration + MarketingSpend',data=data).fit().rsquared
vif_pr = 1/(1-rsq_pr)

        #Storing Vif values in a data frame
    
d1 = {'Variables':['RNDspend','Administration','MarketingSpend','Profit'],'VIF':[vif_rd,vif_ad,vif_ms,vif_pr]} 
vif_frame = pd.DataFrame(d1)
vif_frame


# ## Residual Analysis

# ### Test for Normality of Residuals (Q-Q plot)

# In[20]:


res = model1.resid
res


# In[21]:


res.mean()


# In[22]:


qqplot= sm.qqplot(res,line= 'q')
plt.title("Test for Normaltiy of Residuals (Q-Q Plot)")
plt.show


# In[23]:


list(np.where(model1.resid<-30000))


# ## Residual Plots for Homoscedasticity

# In[24]:


def get_standardized_values(vals):
        return (vals - vals.mean())/vals.std()


# In[25]:


plt.scatter(get_standardized_values(model1.fittedvalues),get_standardized_values(model1.resid))
plt.titles("Residual plot")
plt.xlabel("Standardized fitted values")
plt.ylabel("Standardized residual values")
plt.show()


# ## Residual vs Regressors

# In[26]:


fig = plt.figure(figsize = (15,8))
fig = sm.graphics.plot_regress_exog(model1, "RNDspend",fig=fig)
plt.show()


# In[27]:


fig = plt.figure(figsize = (15,8))
fig = sm.graphics.plot_regress_exog(model1, "Administration",fig=fig)
plt.show()


# In[28]:


fig = plt.figure(figsize = (15,8))
fig = sm.graphics.plot_regress_exog(model1, "MarketingSpend",fig=fig)
plt.show()


# ## Model Deletion Diagnostics

# ### Detecting influenecrs/outliers

# ### Cook's Distance

# In[29]:


#Calculating Cooks distance and plotting it
(c,_) = model1.get_influence().cooks_distance
c


# In[30]:


#Plot the influencers values using stem plot
fig = plt.subplots(figsize=(20,7))
plt.stem(np.arange(len(data)),c)
plt.xlabel('Row index')
plt.ylabel('Cooks distane')
plt.show()


# In[31]:


#Index and value of influencer where c is more than 0.20
np.argmax(c),np.max(c)


# ### High Influence Points

# In[32]:


data[data.index.isin([48,49])]


# ### Dropping the outliers

# In[33]:


#dropping the outliers
data_new = data.drop(data.index[[48,49]],axis=0).reset_index()


# In[34]:


data_new


# In[35]:


data_new1 = data_new.drop(['index'],axis=1)
data_new1.shape


# ## Building our final model on new cleaned dataset.

# In[36]:


final_model = smf.ols('Profit ~ RNDspend + Administration + MarketingSpend',data= data_new1).fit()
final_model.summary()


# In[37]:


#Rsquared and Akaike information criterion
(final_model.rsquared, final_model.aic)


# ## Model Predictions

# In[38]:


newdata = pd.DataFrame({"RNDspend" : 80000,"Administration": 90000,"MarketingSpend": 390900},index=[1])
newdata


# In[39]:


final_model.predict(newdata)


# In[40]:


pred_y  = final_model.predict(data_new)
pred_y


# ## Table with both RSquared values of original model and final model

# In[41]:


df = {'Prepared model':['Model','Final Model'],'Rsquared':[model1.rsquared,final_model.rsquared]}
table = pd.DataFrame(df)
table


# In[ ]:





# In[ ]:





# In[ ]:




