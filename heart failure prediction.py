#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
colors = ['#0000FF','#FF0000']

import imblearn
from collections import Counter
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import RocCurveDisplay
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import precision_recall_curve
import warnings
warnings.filterwarnings("ignore")


# In[2]:


pip install --upgrade scipy


# In[3]:


pip install imbalanced-learn


# In[5]:


pip install lightgbm


# In[2]:


data=pd.read_csv("C:\\Users\\sendm\\Downloads\\archive (3)\\heart_failure_clinical_records_dataset.csv")


# In[3]:


data.head()


# In[4]:


data.info


# In[5]:


data.shape


# In[6]:


data.columns


# In[7]:


data.describe().T


# In[8]:


data.isnull().mean()*100


# In[9]:


data['age'] = data['age'].astype(int)
data['platelets'] = data['platelets'].astype(int)
df = data.copy(deep = True)


# In[10]:


df.loc[df['DEATH_EVENT']==0,'Status']='Survived'
df.loc[df['DEATH_EVENT']==1,'Status']='Not Survived'


# In[11]:


col = list(data.columns)
categorical_features = []
numerical_features = []
for i in col:
    if len(data[i].unique()) > 6:
        numerical_features.append(i)
    else:
        categorical_features.append(i)
print('Categorical Features :',*categorical_features)
print('Numerical Features :',*numerical_features)


# In[12]:


sns.set(style='white')

fig, axes = plt.subplots(1, 2, figsize=(13, 4))

plt.subplot(1, 2, 1)
df['Status'].value_counts().plot.pie(explode=[0.1, 0.1], autopct='%1.1f%%', shadow=True)

plt.subplot(1, 2, 2)
ax = sns.countplot(data=df, x='Status', palette=colors, edgecolor='k')
ax.bar_label(ax.containers[0])

plt.suptitle('Death Event')


# In[13]:


def catplot(df, x): 
    sns.set(style='white')
    fig = plt.subplots(1, 3, figsize=(15, 4))
    plt.subplot(1, 3, 1)
    df[x].value_counts().plot.pie(explode=[0.1, 0.1], autopct='%1.1f%%', shadow=True)  # Removed 'text' parameter

    plt.subplot(1, 3, 2)
    ax = sns.histplot(data=df, x=x, kde=True, color=colors[0], edgecolor='k')
    ax.bar_label(ax.containers[0])

    plt.subplot(1, 3, 3)
    ax = sns.countplot(data=df, x=x, hue='Status', palette=colors, edgecolor='k')
    for container in ax.containers:
        ax.bar_label(container)
    tit = x + ' vs Death Event'
    plt.suptitle(tit)


# In[14]:


catplot(df,'anaemia')


# In[15]:


catplot(df,'diabetes')


# In[16]:


catplot(df,'high_blood_pressure')


# In[17]:


catplot(df,'sex')


# In[18]:


catplot(df,'smoking')


# In[19]:


def numplot(df,x,scale): 
    sns.set(style='whitegrid')
    fig = plt.subplots(2,1,figsize = (15,11))
 
    plt.subplot(2,1,1)
    ax=sns.histplot(data=df, x=x, kde=True,color=colors[0],edgecolor = 'k')
    ax.bar_label(ax.containers[0])
    tit=x + ' distribution'
    plt.title(tit)
    
    plt.subplot(2,1,2)
    tar=x + '_group'
    Tstr= str(scale)
    tit2=x + ' vs Death Event ( ' + Tstr + ' : 1 )'
    df[tar] = [ int(i / scale) for i in df[x]]
    ax=sns.countplot(data=df, x=tar, hue='Status',palette = colors,edgecolor = 'k')
    for container in ax.containers:
        ax.bar_label(container)
    plt.title(tit2)


# In[20]:


numplot(df,'age',5)


# In[21]:


numplot(df,'creatinine_phosphokinase',100)


# In[22]:


numplot(df,'ejection_fraction',10)


# In[23]:


numplot(df,'platelets',10**5)


# In[24]:


numplot(df,'serum_creatinine',1)


# In[25]:


numplot(df,'serum_sodium',5)


# In[26]:


numplot(df,'time',10)


# In[27]:


mms = MinMaxScaler() # Normalization
ss = StandardScaler() # Standardization
# Normalization
df['age'] = mms.fit_transform(df[['age']])
df['creatinine_phosphokinase'] = mms.fit_transform(df[['creatinine_phosphokinase']])
df['ejection_fraction'] = mms.fit_transform(df[['ejection_fraction']])
df['serum_creatinine'] = mms.fit_transform(df[['serum_creatinine']])
df['time'] = mms.fit_transform(df[['time']])
# Standardization
df['platelets'] = ss.fit_transform(df[['platelets']])
df['serum_sodium'] = ss.fit_transform(df[['serum_sodium']])
df.head()


# In[28]:


corr = data.corrwith(data['DEATH_EVENT']).sort_values(ascending = False).to_frame()
corr.columns = ['DEATH_EVENT']
plt.subplots(figsize = (5,5))
sns.heatmap(corr,annot = True,cmap = colors,linewidths = 0.4,linecolor = 'black');
plt.title('DEATH_EVENT Correlation');


# In[34]:


df1=data.copy()
df2=data.copy()

# Dataset for model based on Statistical Test :
df1 = df1.drop(columns = ['anaemia', 'diabetes', 'high_blood_pressure', 'sex', 'smoking','creatinine_phosphokinase', 'platelets'])
# Dataset for model based on General Information :
                          
df2 = df2.drop(columns=['sex', 'platelets'])


# In[36]:


over = SMOTE()
f1 = df1.iloc[:,:5].values
t1 = df1.iloc[:,5].values
f1, t1 = over.fit_resample(f1, t1)
Counter(t1)


# In[37]:


over = SMOTE()
f2 = df2.iloc[:,:10].values
t2 = df2.iloc[:,10].values
f2, t2 = over.fit_resample(f2, t2)
Counter(t2)



# In[54]:


x_train1, x_test1, y_train1, y_test1 = train_test_split(f1, t1, test_size=0.15, random_state=0)
x_train2, x_test2, y_train2, y_test2 = train_test_split(f2, t2, test_size=0.15, random_state=0)


# In[55]:


def model(classifier,x_train,y_train,x_test,y_test):
    sns.set(rc={'figure.figsize':(5,3)})
    sns.set(style='whitegrid')
    classifier.fit(x_train,y_train)
    prediction = classifier.predict(x_test)
    cv = RepeatedStratifiedKFold(n_splits = 10,n_repeats = 3,random_state = 1)
    cross_val_scores = cross_val_score(classifier, x_train, y_train, cv=cv)
    mean_cross_val_score = cross_val_scores.mean()

    print("Cross Validation Score:", '{0:.2%}'.format(mean_cross_val_score))
    #print("Cross Validation Score : ",'{0:.2%}'.format(cross_val_score(classifier,x_train
    print("ROC_AUC Score : ",'{0:.2%}'.format(roc_auc_score(y_test,prediction)))
 # plot_roc_curve(classifier, x_test,y_test)
    RocCurveDisplay.from_estimator(classifier, x_test,y_test)
    plt.title('ROC_AUC_Plot')
    plt.show()
                                                                       
def model_evaluation(classifier,x_test,y_test):
 
 # Confusion Matrix
   cm = confusion_matrix(y_test,classifier.predict(x_test))
   names = ['True Neg','False Pos','False Neg','True Pos']
   counts = [value for value in cm.flatten()]
   percentages = ['{0:.2%}'.format(value) for value in cm.flatten()/np.sum(cm)]
   labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in zip(names,counts,percentages)]
   labels = np.asarray(labels).reshape(2,2)
   sns.heatmap(cm,annot = labels,cmap = 'Blues',fmt ='')
 
 # Classification Report
   print(classification_report(y_test,classifier.predict(x_test)))


# In[56]:


classifier_xgb = XGBClassifier(random_state=1)

model(classifier_xgb,x_train1,y_train1,x_test1,y_test1)
model_evaluation(classifier_xgb,x_test1,y_test1)


# In[57]:


model(classifier_xgb,x_train2,y_train2,x_test2,y_test2)
model_evaluation(classifier_xgb,x_test2,y_test2)


# In[58]:


classifier_lgbm = LGBMClassifier(random_state=1)
model(classifier_lgbm,x_train1,y_train1,x_test1,y_test1)
model_evaluation(classifier_lgbm,x_test1,y_test1)


# In[59]:


model(classifier_lgbm,x_train2,y_train2,x_test2,y_test2)
model_evaluation(classifier_lgbm,x_test2,y_test2)


# In[60]:


classifier_lr = LogisticRegression(random_state = 1) 
model(classifier_lr,x_train1,y_train1,x_test1,y_test1)
model_evaluation(classifier_lr,x_test1,y_test1)


# In[61]:


model(classifier_lr,x_train2,y_train2,x_test2,y_test2)
model_evaluation(classifier_lr,x_test2,y_test2)


# In[62]:


classifier_svc = SVC()
model(classifier_svc,x_train1,y_train1,x_test1,y_test1)
model_evaluation(classifier_svc,x_test1,y_test1)


# In[63]:


model(classifier_svc,x_train2,y_train2,x_test2,y_test2)
model_evaluation(classifier_svc,x_test2,y_test2)


# In[64]:


classifier_grad = GradientBoostingClassifier(random_state=1)
model(classifier_grad,x_train1,y_train1,x_test1,y_test1)
model_evaluation(classifier_grad,x_test1,y_test1)


# In[65]:


model(classifier_svc,x_train2,y_train2,x_test2,y_test2)
model_evaluation(classifier_svc,x_test2,y_test2)


# In[66]:


classifier_rdf = RandomForestClassifier(random_state=1)
model(classifier_rdf,x_train1,y_train1,x_test1,y_test1)
model_evaluation(classifier_rdf,x_test1,y_test1)


# In[67]:


model(classifier_svc,x_train2,y_train2,x_test2,y_test2)
model_evaluation(classifier_svc,x_test2,y_test2)


# In[68]:


pip install flaml


# In[71]:


from flaml import AutoML
automl = AutoML()


# In[72]:


classifier_lgbm = LGBMClassifier(colsample_bytree=0.26649620250942635,
              learning_rate=0.02058909150877934, max_bin=127,
              min_child_samples=7, n_estimators=184, num_leaves=48,
              reg_alpha=0.004090180440029941, reg_lambda=0.0009765625,
              verbose=-1)


# In[73]:


model(classifier_lgbm,x_train1,y_train1,x_test1,y_test1)
model_evaluation(classifier_lgbm,x_test1,y_test1)


# In[74]:


model(classifier_lgbm,x_train2,y_train2,x_test2,y_test2)
model_evaluation(classifier_lgbm,x_test2,y_test2)


# In[ ]:


The Final Results:
After using flaml, the results of both datasets improved

