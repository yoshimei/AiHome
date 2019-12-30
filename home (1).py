#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics


# ### Loading the Dataset

# In[3]:


heart = pd.read_csv("dvc_data/train.csv")
heart1 = pd.read_csv("dvc_data/test.csv")
su = pd.read_csv("dvc_data/submit_samples.csv")
heart.head()


# In[ ]:





# ### Training and Validation Partition

# In[4]:



heart1 = heart1.drop(['ID'], axis=1)
#heart1.fillna(0, inplace = True)
#heart1 = heart1.astype(float)

X = heart.drop(['Y','ID'], axis=1)
y =heart['Y'] 
#X.fillna(0, inplace = True)
#X = X.astype(float)
X.head()


# ### Prediction

# In[6]:





# In[8]:


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

 

rf0 = RandomForestClassifier(oob_score=True,max_depth=100,max_leaf_nodes=60,n_estimators=110,max_features=0.5, 
                             min_samples_leaf=2, min_samples_split=4, random_state=105,n_jobs=1,bootstrap=True)
rf0.fit(X,y)
print(rf0.oob_score_)
y_predprob = rf0.predict_proba(heart1)
print(y_predprob)


# In[13]:


data = pd.DataFrame({'LEVEL':y_predprob})
data.to_csv("dvc_data/ans.csv")
print(data)


# In[ ]:




