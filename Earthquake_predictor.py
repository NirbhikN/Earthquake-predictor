#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing
import seaborn as sn
import os
print(os.listdir("../Earthquake predictor"))


# In[2]:


from subprocess import check_output
# print(check_output(["ls", "../earthquake predictor/earthquake.csv"]).decode("utf8"))


# In[3]:


veri=pd.read_csv(('../Earthquake predictor/earthquake.csv'), encoding='utf-8', engine='python',sep=',', error_bad_lines=False)
veri.describe()


# In[4]:


veri.corr()


# In[5]:


veri=veri[['date','time','lat','long','depth','xm']]    #index 1: Kocaeli earthquake is our prediction 
data=veri.head(10)  
print(data)


# In[6]:


veri=veri[['lat','long','depth','xm']]    #index 1 is our prediction earthquake
data=veri.head(10)  
print(data)


# In[7]:


import numpy as np #use numpy for y variable for Linear Regression
y=np.array(veri['xm'])


# In[8]:


X=np.array(veri.drop('xm',axis=1)) #prepare attribute (X variable) and drop the predicted value


# In[9]:


from sklearn.model_selection import train_test_split  #lets split data set
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=2)


# In[10]:


from sklearn.linear_model import LinearRegression  #lets set LR model


# In[11]:


linear=LinearRegression() #taking an example


# In[12]:


linear.fit(X_train,y_train) #training data


# In[13]:


data=linear.score(X_test,y_test)  # score of test datas
print(data)


# In[14]:


data=linear.score(X_train,y_train) #score of training datas
print(data)


# In[15]:


print('coefficients: \n',linear.coef_) 
print('intercepts: \n',linear.intercept_)


# In[16]:


predict_data=np.array([[28.05,84.80,50.0]])   #kocaeli earthquake prediction is 4.51, quiet close to the real value
data2=linear.predict(predict_data)
print(data2)   


# In[17]:


import pickle
pickle.dump(linear, open('model.pkl', 'wb'))
model = pickle.load(open('model.pkl', 'rb'))

# %%
