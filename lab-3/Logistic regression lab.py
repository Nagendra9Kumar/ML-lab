#!/usr/bin/env python
# coding: utf-8

# # implementation of Logistic Regression

# # Step-1: Data Pre-processing
# 
# # we will import the three important libraries, for loading the dataset, plotting the graphs, and creating the Simple Linear Regression model.
# 
# 

# In[1]:



import numpy as nm  
import matplotlib.pyplot as mtp  
import pandas as pd  


# 
# 
# # Load the dataset in our code
# 
# 

# In[2]:



# In[3]:


data_set= pd.read_csv('car_data.csv')  


# In[4]:


data_set


# # Extracting Independent and dependent Variable
# # we will predict the purchased variable (Dependent Variable) by using age and salary (Independent variables).

# In[5]:


x= data_set.iloc[:, [2,3]].values  
y= data_set.iloc[:, 4].values  


# In[6]:


x


# In[7]:


y


# # Splitting the dataset into training and test set.

# In[8]:


from sklearn.model_selection import train_test_split  
x_train, x_test, y_train, y_test= train_test_split(x, y, test_size= 0.25, random_state=0)  


# # Test-dataset:

# In[9]:


x_test


# In[10]:


y_test


# # Training Dataset:

# In[11]:


x_train


# In[12]:


y_train


# # In logistic regression, we will do feature scaling because we want accurate result of predictions. Here we will only scale the independent variable because dependent variable have only 0 and 1 values. 

# In[13]:


from sklearn.preprocessing import StandardScaler    
st_x= StandardScaler()    
x_train= st_x.fit_transform(x_train)    
x_test= st_x.transform(x_test)  


# In[14]:


x_train


# In[15]:


x_test


# # STEP 2: Fitting Logistic Regression to the Training set:

# In[16]:


from sklearn.linear_model import LogisticRegression  
classifier= LogisticRegression(random_state=0)  
classifier.fit(x_train, y_train)  


# In[17]:


LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,  
                   intercept_scaling=1, l1_ratio=None, max_iter=100,  
                   multi_class='warn', n_jobs=None, penalty='l2',  
                   random_state=0, solver='warn', tol=0.0001, verbose=0,  
                   warm_start=False)  


# # STEP 3: Predicting the Test Result

# In[18]:


y_pred= classifier.predict(x_test)  


# In[19]:


y_pred


# # step 4: Test Accuracy of the result

# In[20]:


from sklearn.metrics import confusion_matrix  
cm= confusion_matrix(y_test,y_pred)  


# In[21]:


cm


# # We can find the accuracy of the predicted result by interpreting the confusion matrix. By above output, we can interpret that 65+24= 89 (Correct Output) and 8+3= 11(Incorrect Output).

# # STEP 5: Visualizing the training set result

# In[22]:


from matplotlib.colors import ListedColormap  
x_set, y_set = x_train, y_train  
x1, x2 = nm.meshgrid(nm.arange(start = x_set[:, 0].min() - 1, stop = x_set[:, 0].max() + 1, step  =0.01),  
nm.arange(start = x_set[:, 1].min() - 1, stop = x_set[:, 1].max() + 1, step = 0.01))  
mtp.contourf(x1, x2, classifier.predict(nm.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),  
alpha = 0.75, cmap = ListedColormap(('purple','green' )))  
mtp.xlim(x1.min(), x1.max())  
mtp.ylim(x2.min(), x2.max())  
for i, j in enumerate(nm.unique(y_set)):  
    mtp.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],  
        c = ListedColormap(('purple', 'green'))(i), label = j)  
mtp.title('Logistic Regression (Training set)')  
mtp.xlabel('Age')  
mtp.ylabel('Estimated Salary')  
mtp.legend()  
mtp.show() 


# # visualizing the testing set result

# In[23]:


from matplotlib.colors import ListedColormap  
x_set, y_set = x_test, y_test  
x1, x2 = nm.meshgrid(nm.arange(start = x_set[:, 0].min() - 1, stop = x_set[:, 0].max() + 1, step  =0.01),  
nm.arange(start = x_set[:, 1].min() - 1, stop = x_set[:, 1].max() + 1, step = 0.01))  
mtp.contourf(x1, x2, classifier.predict(nm.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),  
alpha = 0.75, cmap = ListedColormap(('purple','green' )))  
mtp.xlim(x1.min(), x1.max())  
mtp.ylim(x2.min(), x2.max())  
for i, j in enumerate(nm.unique(y_set)):  
    mtp.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],  
        c = ListedColormap(('purple', 'green'))(i), label = j)  
mtp.title('Logistic Regression (Test set)')  
mtp.xlabel('Age')  
mtp.ylabel('Estimated Salary')  
mtp.legend()  
mtp.show()  


# # OBSERVATION:

# # The above graph shows the test set result. As we can see, the graph is divided into two regions (Purple and Green). And Green observations are in the green region, and Purple observations are in the purple region. So we can say it is a good prediction and model. Some of the green and purple data points are in different regions, which can be ignored as we have already calculated this error using the confusion matrix (11 Incorrect output).
# 
# Hence our model is pretty good and ready to make new predictions for this classification problem.

# In[ ]:




