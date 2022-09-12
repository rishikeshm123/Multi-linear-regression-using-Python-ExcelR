#!/usr/bin/env python
# coding: utf-8

# # Multi Linear Regression

# In[88]:


import pandas as pd
import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
import seaborn as sns


# ## Import Dataset, EDA and Visualisation

# In[89]:


data = pd.read_csv('ToyotaCorolla.csv',encoding='latin1')
data.head()


# In[90]:


data.shape


# In[91]:


col= ["Price","Age_08_04","KM","HP","cc","Doors","Gears","Quarterly_Tax","Weight"]
data2 = data[col]
data2.head()


# In[98]:


data2.isna().sum()


# In[99]:


data2.describe()


# In[100]:


data2[data2.duplicated()]


# In[101]:


data2.info()


# ## Correlation Matrix

# In[102]:


data2.corr()


# ## Scatterplot between variables along with histograms

# In[103]:


sns.pairplot(data2)


# ## Model Building 

# In[106]:


model = smf.ols('Price~ Age_08_04+KM+HP+cc+Doors+Gears+Quarterly_Tax+Weight',data=data2).fit()
model.summary()


# In[107]:


(model.rsquared,model.aic)


# ### According to the model summary, Even though model accuracy is 86.37%,we can see that pvalues of cc and doors do no satisy the 0.05 alpha value.

# ### Building simple linear regression models wth cc,doors and both combined.

# In[108]:


m1_cc= smf.ols('Price ~ cc',data =data2).fit()
m1_cc.summary()


# In[109]:


m1_doors= smf.ols('Price ~ Doors',data =data2).fit()
m1_doors.summary()


# In[110]:


m1_ccd= smf.ols('Price ~ cc+Doors',data =data2).fit()
m1_ccd.summary()


# ## Calculating VIF

# In[111]:


# 1) Collinearity Problem Check
# Calculate VIF = 1/(1-Rsquare) for all independent variables
rsq_wt = smf.ols('Weight ~ Price+Age_08_04+KM+HP+cc+Doors+Gears+Quarterly_Tax',data = data2).fit().rsquared
vif_wt = 1/(1-rsq_wt)

rsq_age = smf.ols(' Age_08_04 ~ Price+Weight+KM+HP+cc+Doors+Gears+Quarterly_Tax',data = data2).fit().rsquared
vif_age = 1/(1-rsq_age)

rsq_km = smf.ols('KM~ Price+Age_08_04+Weight+HP+cc+Doors+Gears+Quarterly_Tax',data = data2).fit().rsquared
vif_km = 1/(1-rsq_km)

rsq_hp = smf.ols('HP ~ Price+Age_08_04+KM+Weight+cc+Doors+Gears+Quarterly_Tax',data = data2).fit().rsquared
vif_hp = 1/(1-rsq_hp)

rsq_cc = smf.ols('cc ~ Price+Age_08_04+KM+HP+Weight+Doors+Gears+Quarterly_Tax',data = data2).fit().rsquared
vif_cc = 1/(1-rsq_cc)

rsq_do = smf.ols('Doors ~ Price+Age_08_04+KM+HP+cc+Weight+Gears+Quarterly_Tax',data = data2).fit().rsquared
vif_do = 1/(1-rsq_do)

rsq_ge = smf.ols('Gears ~ Price+Age_08_04+KM+HP+cc+Doors+Weight+Quarterly_Tax',data = data2).fit().rsquared
vif_ge = 1/(1-rsq_ge)

rsq_wt = smf.ols(' Quarterly_Tax~ Price+Age_08_04+KM+HP+cc+Doors+Gears+Weight',data = data2).fit().rsquared
vif_wt = 1/(1-rsq_wt)

#Storing VIF Values in a data frame
    
d1 = {'Variables':['Weight','Age_08_04','KM','HP','cc','Doors','Gears','Quarterly_Tax'],'VIF':[vif_wt,vif_age,vif_km,vif_hp,vif_cc,vif_do,vif_ge,vif_wt]}
vif_frame = pd.DataFrame(d1)
vif_frame


# ## Residual Analysis

# ## Test for Normality of Residuals (Q-Q Plot)

# In[112]:


# 2) Residual analysis check
res= model.resid
res


# In[113]:


res.mean()


# In[114]:


qqplot = sm.qqplot(res,line= 'q')
plt.title("Test for Normality of Residuals (Q-Q Plot)")
plt.show


# In[119]:


list(np.where(model.resid<-4500,)) ,list(np.where(model.resid>4500))


# ## Residual plots for homoscedasticity

# In[120]:


def get_standardized_values(vals):
    return  (vals-vals.mean())/vals.std()


# In[121]:


# Residual plot for Homoscedasticity
plt.scatter(get_standardized_values(model.fittedvalues),get_standardized_values(model.resid))
plt.titles("Residual plot")
plt.xlabel("Standardized fitted values")
plt.ylabel("Standardized residual values")
plt.show()


# ## Residuals vs Regressors

# In[122]:


fig = plt.figure(figsize = (15,8))
fig = sm.graphics.plot_regress_exog(model, "Age_08_04",fig=fig)
plt.show()


# In[123]:


fig = plt.figure(figsize = (15,8))
fig = sm.graphics.plot_regress_exog(model, "KM",fig=fig)
plt.show()


# In[124]:


fig = plt.figure(figsize = (15,8))
fig = sm.graphics.plot_regress_exog(model, "HP",fig=fig)
plt.show()


# In[125]:


fig = plt.figure(figsize = (15,8))
fig = sm.graphics.plot_regress_exog(model, "cc",fig=fig)
plt.show()


# In[126]:


fig = plt.figure(figsize = (15,8))
fig = sm.graphics.plot_regress_exog(model, "Doors",fig=fig)
plt.show()


# In[127]:


fig = plt.figure(figsize = (15,8))
fig = sm.graphics.plot_regress_exog(model, "Gears",fig=fig)
plt.show()


# In[128]:


fig = plt.figure(figsize = (15,8))
fig = sm.graphics.plot_regress_exog(model, "Quarterly_Tax",fig=fig)
plt.show()


# In[129]:


fig = plt.figure(figsize = (15,8))
fig = sm.graphics.plot_regress_exog(model, "Weight",fig=fig)
plt.show()


# ## Calculating Cooks distance to identify outliers/influencers

# In[130]:


#Calculating Cooks distance and plotting it
(c,_) = model.get_influence().cooks_distance
c


# In[131]:


#Plot the influencers values using stem plot
fig = plt.subplots(figsize=(20,7))
plt.stem(np.arange(len(data)),c)
plt.xlabel('Row index')
plt.ylabel('Cooks distance')
plt.show()


# In[132]:


#index of the data points where c is more than 0.5
np.argmax(c) , np.max(c)


# In[133]:


data2[data2.index.isin([80])] 


# ## Improving the model

# In[134]:


#Removing 80 observation and Reset the index,re arrange the row values
data3 = data2.drop(data2.index[[80]],axis = 0).reset_index(drop = True)
data3


# ## Model Deletion diagnostics and final model

# In[135]:


while model.rsquared < 0.90:
    for c in [np.max(c)>0.5]:
        model=smf.ols('Price~Age_08_04+KM+HP+cc+Doors+Gears+Quarterly_Tax+Weight',data=data3).fit()
        (c,_)=model.get_influence().cooks_distance
        c
        np.argmax(c) , np.max(c)
        data3=data3.drop(data3.index[[np.argmax(c)]],axis=0).reset_index(drop=True)
        data3
    else:
        final_model=smf.ols('Price~Age_08_04+KM+HP+cc+Doors+Gears+Quarterly_Tax+Weight',data=data3).fit()
        final_model.rsquared , final_model.aic
        print("Thus model accuracy is improved to",final_model.rsquared)


# In[136]:


final_model.rsquared


# In[137]:


data3


# ## Model Predictions

# In[159]:


new_data = pd.DataFrame({"Age_08_04":30, "KM":3000,"HP":70 ,"cc":2000 ,"Doors":3 ,"Gears":5 ,"Quarterly_Tax":210 ,"Weight":1000},index = [1])
new_data


# In[161]:


final_model.predict(new_data)


# In[162]:


# Automatic Prediction of Price with 90 % accuracy
pred_y=final_model.predict(data3)
pred_y

