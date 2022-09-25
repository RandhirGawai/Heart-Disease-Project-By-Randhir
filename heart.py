# -*- coding: utf-8 -*-
"""
Created on Sat Sep 24 16:15:25 2022

@author: Randhir
"""
import pandas as pd

data=pd.read_csv("F:/Lnb PROGRAM/Mini Project/heart.csv")

#Taking Care of Missing Values
print(data.isnull().sum())

#Taking care of Duplicate Values

data_dup=data.duplicated().any()
print(data_dup)
data=data.drop_duplicates()
data_dup=data.duplicated().any()
print(data_dup)


#Data processing

cate_val=[]
cont_val=[]

for column in data.columns:
    if data[column].nunique()<=10:
        cate_val.append(column)
    else:
        cont_val.append(column)
print(cate_val)

#Encoding Categorical Data

print(data['cp'].unique())
cate_val.remove('sex')
cate_val.remove('target')
data=pd.get_dummies(data,columns=cate_val,drop_first=True)
print(data.head())
        

#Feature Scaling

from sklearn.preprocessing import StandardScaler

st=StandardScaler()
data[cont_val]=st.fit_transform(data[cont_val])
print(data.head())


#Splitting dataset into the training and test set

X=data.drop('target',axis=1)
y=data['target']

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)


#Logistic Regression

print(data.head())

from sklearn.linear_model import LogisticRegression

log=LogisticRegression()
log.fit(X_train,y_train)
y_pred1=log.predict(X_test)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred1))


#SVC

from sklearn import svm
svm=svm.SVC()
svm.fit(X_train,y_train)
y_pred2=svm.predict(X_test)
print(accuracy_score(y_test, y_pred2))

#KNeighbors Classifier

from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier()
knn.fit(X_train,y_train)
y_pred3=knn.predict(X_test)
print(accuracy_score(y_test, y_pred3))


score=[]
for k in range(1,40):
    knn=KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train,y_train)
    y_pred=knn.predict(X_test)
    score.append(accuracy_score(y_test, y_pred))
print(score)

knn=KNeighborsClassifier(n_neighbors=2)
knn.fit(X_train,y_train)
y_pred=knn.predict(X_test)
print(accuracy_score(y_test, y_pred))

#Non-Linear ML Algorithms

data=pd.read_csv("F:/Lnb PROGRAM/Mini project/heart.csv")
print(data.head())
data=data.drop_duplicates()
print(data.shape)

X=data.drop('target',axis=1)
y=data['target']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier()     
dt.fit(X_train,y_train)
y_pred4=dt.predict(X_test)
print(accuracy_score(y_test, y_pred4))

#Random forestclassifier

from sklearn.ensemble import RandomForestClassifier

rf=RandomForestClassifier()
rf.fit(X_train,y_train)
y_pred5=rf.predict(X_test)
print(accuracy_score(y_test, y_pred5))


#Gradient Boosting Classifier

from sklearn.ensemble import GradientBoostingClassifier
gbc=GradientBoostingClassifier()
gbc.fit(X_train,y_train)
y_pred6=gbc.predict(X_test)
print(accuracy_score(y_test, y_pred6))


final_data=pd.DataFrame({'Models':['LR','SVM','KNN','DT','RF','GB'],'ACC':[accuracy_score(y_test, y_pred1),accuracy_score(y_test, y_pred2),accuracy_score(y_test, y_pred3),accuracy_score(y_test, y_pred4),accuracy_score(y_test, y_pred5),accuracy_score(y_test, y_pred6),]})

print(final_data)

import seaborn as sns
sns.barplot(final_data['Models'],final_data['ACC'])

X=data.drop('target',axis=1)
y=data['target']
print(X.shape)

from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier()
rf.fit(X,y)


#prediction on new data

import pandas as pd
new_data=pd.DataFrame({
    'age':52,
    'sex':1,
    'cp':0,
    'trestbps':125,
    'chol':212,
    'fbs':0,
    'restecg':1,
    'thalach':168,
    'exang':0,
    'oldpeak':1.0,
    'slop':2,
    'ca':2,
    'thal':3,
    },index=[0])
print(new_data)
p=rf.predict(new_data)
if p[0]==0:
    print("No disease")
else:
    print("Disease")
    
