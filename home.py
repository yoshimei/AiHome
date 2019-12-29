#!/usr/bin/env python
# coding: utf-8

# In[12]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics


# ### Loading the Dataset

# In[13]:


heart = pd.read_csv("dvc_data/train.csv")
heart1 = pd.read_csv("dvc_data/test.csv")
su = pd.read_csv("dvc_data/submit_samples.csv")
heart.head()


# In[ ]:





# ### Training and Validation Partition

# In[14]:



heart1 = heart1.drop(['ID'], axis=1)
#heart1.fillna(0, inplace = True)
#heart1 = heart1.astype(float)

X = heart.drop(['Y','ID'], axis=1)
y =heart['Y'] 
#X.fillna(0, inplace = True)
#X = X.astype(float)
X.head()


# In[23]:


from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn.model_selection import train_test_split
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size = 0.4)
# 建立分類器
clf = tree.DecisionTreeClassifier()
h_clf = clf.fit(train_X, train_y)

# 預測
test_y_predicted = h_clf.predict(test_X)
print(accuracy_score(test_y, test_y_predicted))
#print(test_y_predicted)

# 標準答案
print(h_clf.predict_proba(test_X)[:300])


# ### Prediction

# In[13]:


data = pd.DataFrame({'ID':su.ID ,'LEVEL':pre})
data.to_csv("ocean/12102.csv",index=False,sep=',')
print(data)


# In[149]:


import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
#from sklearn.grid_search import GridSearchCV
#from sklearn import cross_validation, metrics
import matplotlib.pylab as plt
#%matplotlib inline
heart = pd.read_csv("dvc_data/train.csv")
heart1 = pd.read_csv("dvc_data/test.csv")
su = pd.read_csv("dvc_data/submit_samples.csv")
heart.head()

heart1 = heart1.drop(['ID'], axis=1)
X = heart.drop(['Y','ID'], axis=1)
y =heart['Y'] 

 
#不管任何参数，都用默认的，拟合下数据看看
rf0 = RandomForestClassifier(oob_score=True,max_depth=100,max_leaf_nodes=60,n_estimators=110,max_features=0.5, 
                             min_samples_leaf=2, min_samples_split=4, random_state=105,n_jobs=1,bootstrap=True)
rf0.fit(X,y)
print(rf0.oob_score_)
y_predprob = rf0.predict_proba(heart1)
print(y_predprob)


# In[ ]:





# In[ ]:




