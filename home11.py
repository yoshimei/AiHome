#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[5]:


from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn.model_selection import train_test_split
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size = 0.2)
# 建立分類器
clf = tree.DecisionTreeClassifier()
h_clf = clf.fit(train_X, train_y)

# 預測
test_y_predicted = h_clf.predict(test_X)
print(accuracy_score(test_y, test_y_predicted))
#print(test_y_predicted)

# 標準答案
print(h_clf.predict_proba(test_X)[:300])


# In[8]:


from sklearn.linear_model import LogisticRegression
import numpy as np

x_train = X
y_train = y
 
x_test = heart1
 
clf = LogisticRegression()
clf.fit(x_train, y_train)
 
# 返回预测标签
print(clf.predict(x_test))

# 返回预测属于某标签的概率
print(clf.predict_proba(x_test))


# In[92]:


from sklearn.ensemble import GradientBoostingClassifier
train_x, test_x, train_y, test_y = train_test_split(X, y, test_size = 0.2)
clf = GradientBoostingClassifier(n_estimators=200, max_depth=3,max_features=0.5,max_leaf_nodes=45)
clf.fit(train_x, train_y)

# print(clf.feature_importances_)
pred = clf.predict(test_x)
pred1 = clf.predict_proba(heart1)
print(accuracy_score(test_y, pred))
#print(clf.predict(heart1))
print(pred1)


# ### Prediction

# In[94]:


print(pred1.shape)


# In[95]:


data = pd.DataFrame({'ID':su.ID ,'C1':pred1[:300,0],'C2':pred1[:300,1],'C3':pred1[:300,2]})
data.to_csv("dvc_data/010522.csv",index=False,sep=',')
print(data)


# In[1]:


import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pylab as plt
heart = pd.read_csv("train.csv")
heart1 = pd.read_csv("test.csv")
su = pd.read_csv("submit_samples.csv")
heart.head()
heart1 = heart1.drop(['ID'], axis=1)
X = heart.drop(['Y','ID'], axis=1)
y =heart['Y'] 
#不管任何参数，都用默认的，拟合下数据看看
rf0 = RandomForestClassifier(oob_score=True,max_depth=8,max_leaf_nodes=45,n_estimators=300,max_features=0.7, 
                             min_samples_leaf=1, min_samples_split=3, random_state=1000,n_jobs=1,bootstrap=True)
rf0.fit(X,y)
print(rf0.oob_score_)
y_predprob = rf0.predict_proba(heart1)
print(y_predprob)


# In[ ]:





# In[ ]:




