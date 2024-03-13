#!/usr/bin/env python
# coding: utf-8

# In[11]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import cross_val_score


# In[5]:


df = pd.read_csv("dataset/train.csv")


# In[8]:


df.head()


# In[7]:


# Drop the Loan_ID column as it is of no use for model.
data = df.drop(columns=["Loan_ID"])


# In[9]:


data.info()


# In[29]:


data.columns


# In[28]:


categorical_data = [i for i in data.columns if data[i].dtype=="object"]
categorical_data


# In[13]:


numerical_data = [i for i in data.columns if data[i].dtype!="object"]
numerical_data


# In[26]:


discrete_numerical_data = [i for i in numerical_data if len(data[i].unique())<16]
discrete_numerical_data


# In[27]:


continuous_numerical_data = [i for i in numerical_data if len(data[i].unique())>=16]
continuous_numerical_data


# In[ ]:





# ## Data Visualisation

# In[16]:


# For Categotical Data.
# for i in categorical_data:
#     data[i].value_counts().plot(kind="bar")
#     plt.xlabel(i)
#     plt.ylabel("Counts")
#     plt.show()


# In[30]:


# For Discrete Numerical Data
# for i in discrete_numerical_data:
#     data[i].value_counts().plot(kind="bar")
#     plt.xlabel(i)
#     plt.ylabel("Counts")
#     plt.show()


# In[31]:


# For Continuous Numerical Data
# for i in continuous_numerical_data:
#     sns.histplot(data[i])
#     plt.xlabel(i+" Distribution")
#     plt.show()


# In[32]:


# Check for outliers(Since the above distribution graphs are skewed, hence outliers are present
# for i in continuous_numerical_data:
#     sns.boxplot(data = data, y=i)
#     plt.show()


# In[15]:


# From the above box plot we can know that outliers are present, hence we need to handle missing values by replacing with median value.


# In[ ]:





# ## Handling Missing Values

# In[33]:


#sns.heatmap(data.isnull(), cbar=False)


# In[34]:


# Since there are null values in categorical values and discrete numerical values, so we replace them with mode of that feature.
for i in categorical_data+discrete_numerical_data:
    data[i] = data[i].fillna(data[i].mode().iloc[0])


# In[35]:


# Now We replace the loan amount column from numerical category.
data["LoanAmount"] = data["LoanAmount"].fillna(data["LoanAmount"].median())


# In[36]:


#sns.heatmap(data.isnull(), cbar=False)


# In[37]:


# Correlation Matrix
#data.corr()


# In[38]:


#sns.heatmap(data.corr())


# In[ ]:





# ## Feature Engineering

# In[39]:


# We will encode the categorical data using label Encoder.
le = preprocessing.LabelEncoder()

for i in categorical_data:
    data[i] = le.fit_transform(data[i])


# In[40]:


data.head()


# In[41]:


# If we want to apply log transformation for all the numerical variables, then majority of CoapplicantIncome values are 0.
# Hence we will create a new variable called TotalIncome = ApplicantIncome + CoapplicantIncome.

data["TotalIncome"] = data["ApplicantIncome"]+data["CoapplicantIncome"]


# In[42]:


data.drop(["ApplicantIncome","CoapplicantIncome"],axis=1,inplace=True)


# In[43]:


continuous_numerical_data


# In[44]:


continuous_numerical_data.remove("ApplicantIncome")
continuous_numerical_data.remove("CoapplicantIncome")
continuous_numerical_data.append("TotalIncome")


# In[45]:


continuous_numerical_data


# In[ ]:





# In[46]:


# Log Transformation
for i in continuous_numerical_data+["Loan_Amount_Term"]:
    data[i] = np.log(data[i])


# In[42]:


data.head()


# In[ ]:





# ## Model Building

# In[31]:


# Splitting the data.


# In[47]:


X,y = data.drop(columns = "Loan_Status"),data["Loan_Status"]


# In[48]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# In[ ]:





# In[49]:


# Logistic Regression
model = LogisticRegression()


# In[51]:


model.fit(X_train,y_train)


# In[52]:


print("Accuracy of Logistic Regression Model is ",model.score(X_test,y_test)*100)


# In[53]:


score = cross_val_score(model, X, y, cv=5)
print("Cross validation is",np.mean(score)*100)


# In[ ]:





# In[39]:

import pickle
 # open a file, where you want to store the data
file = open('model.pkl', 'wb')

# # dump information to that file
pickle.dump(model, file)


# In[ ]:




