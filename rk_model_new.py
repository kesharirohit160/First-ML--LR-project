#!/usr/bin/env python
# coding: utf-8

# In[74]:


# import liberary
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import pickle


# In[75]:


# read the data set
salary = pd.read_csv(r"C:\Users\Admin\Downloads\Deployment-flask-master\ML Model Basic\hiring.csv")


# In[76]:


# data part
salary.head(10)


# In[77]:


# shape of the data set
salary.shape


# In[78]:


# info part
salary.info()


# In[79]:


# data stats part
salary.describe()


# In[80]:


# nan cal
salary.isnull().sum()


# In[81]:


sns.pairplot(data = salary, x_vars= ['test_score','interview_score'], y_vars = 'salary')


# In[82]:


# nan impute for 
salary['experience'].fillna(0, inplace=True)


# In[83]:


# head part
salary.head(10)


# In[84]:


#test score nan impute
salary["test_score"].fillna(salary["test_score"].mean(), inplace = True)


# In[85]:


salary.head(10)


# In[86]:


sns.pairplot(data = salary, x_vars= ['test_score','interview_score'], y_vars = 'salary')


# In[87]:


# Indepent variable store
X = salary.iloc[:, :3]


# In[88]:


# experience year to number funtion
def covert_to_int(word):
    word_dict = {'one':1, 'two':2, 'three':3, 'four':4, 'five':5, 'six':6, 'seven':7, 'eight':8,
                'nine':9, 'ten':10, 'eleven':11, 'twelve':12, 'zero':0, 0: 0}
    return(word_dict[word])


# In[89]:


# applying the function
X["experience"] = X["experience"].apply(lambda x : covert_to_int(x))


# In[90]:


# X head part
X.head()


# In[91]:


# tarhet variable
y = salary.iloc[:,-1:]


# In[92]:


# head party
y.head(10)


# In[93]:


LR = LinearRegression()


# In[94]:


# fit the model with data given (Training data)

LR.fit(X, y)


# In[95]:


# coff.
LR.coef_


# In[96]:


# intercept
LR.intercept_


# In[97]:


# Saving model to disk
pickle.dump(LR, open('rk_model.pkl','wb'))


# In[98]:


# Loading model to compare the results
rk_model = pickle.load(open('rk_model.pkl','rb'))
print(rk_model.predict([[12, 9, 6]]))


# In[99]:


# Thank you

