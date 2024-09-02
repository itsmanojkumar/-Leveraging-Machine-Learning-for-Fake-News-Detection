#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import re
import string
from sklearn.metrics import classification_report


# In[2]:


true=pd.read_csv("true.csv")
fake=pd.read_csv("fake.csv")


# In[3]:


fake.head()
true.head()


# In[4]:


fake["class"]=0
true["class"]=1


# In[5]:


fake.shape,true.shape


# In[6]:


fake_manuals_testing=fake.tail(10)
for i in range(23480,23470,-1):
    fake.drop([i],axis=0,inplace=True)


# In[7]:


true_manuals_testing=true.tail(10)
for i in range(21416,21406,-1):
    true.drop([i],axis=0,inplace=True)


# In[8]:


fake.shape,true.shape


# In[9]:


fake_manuals_testing["class"]=0
true_manuals_testing["class"]=1


# In[10]:


fake_manuals_testing.head()
true_manuals_testing.head()


# In[11]:


merge=pd.concat([true,fake],axis=0)


# In[12]:


merge.columns


# In[13]:


data=merge.drop(['title','subject','date'],axis=1)


# In[14]:


data.isnull().sum()


# In[15]:


data=data.sample(frac=1)
data.head()


# In[16]:


data.reset_index(inplace=True)
data.drop(['index'],axis=1,inplace=True)


# In[17]:


data.head()


# In[21]:


def wordopt(text):
    text=text.lower()
    text=re.sub('\[.*?\]','',text)
    text=re.sub("\\W","",text)
    text=re.sub('https?://\S+|www\.\S+','',text)
    text=re.sub('<.*?>+','',text)
    text=re.sub('[%s]' % re.escape(string.punctuation),'',text)
    text=re.sub('\n','',text)
    text=re.sub('\w*\d\w*','',text)
    return text
        


# In[22]:


data['text'] =data['text'].apply(wordopt)


# In[23]:


x=data['text']
y=data['class']


# In[24]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25)


# In[25]:


from sklearn.feature_extraction.text import TfidfVectorizer
vectorization=TfidfVectorizer()
xv_train=vectorization.fit_transform(x_train)
xv_test=vectorization.transform(x_test)


# In[26]:


from sklearn.linear_model import LogisticRegression
LR=LogisticRegression()
LR.fit(xv_train,y_train)


# In[27]:


pred_lr=LR.predict(xv_test)


# In[28]:


LR.score(xv_test,y_test)


# In[29]:


print(classification_report(y_test, pred_lr))


# In[30]:


from sklearn.tree import DecisionTreeClassifier
DT=DecisionTreeClassifier()
DT.fit(xv_train,y_train)


# In[31]:


pred_dt=DT.predict(xv_test)


# In[32]:


DT.score(xv_test,y_test)


# In[33]:


print(classification_report(y_test, pred_lr))


# In[62]:


def output_lable(n):
    if n==0:
        return "fake news"
    elif n==1:
        return "not a fake news"
    
def manual_testing(news):
    testing_news={"text":[news]}
    new_def_test=pd.DataFrame(testing_news)
    new_def_test['text']=new_def_test["text"].apply(wordopt)
    new_x_test=new_def_test["text"]
    new_xv_test=vectorization.transform(new_x_test)
    pred_LR=LR.predict(new_xv_test)
    pred_DT=DT.predict(new_xv_test)
    
    return print("\n\nLR prediction: {} \nDT prediction: {}".format(output_lable(pred_LR[0]),output_lable(pred_DT[0])))
    


# In[63]:


news=str(input())
manual_testing(news)


# In[ ]:





# In[ ]:





# In[ ]:




