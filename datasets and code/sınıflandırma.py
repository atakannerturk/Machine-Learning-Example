# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 14:18:53 2020

@author: Atakan
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

veriler = pd.read_csv("veriler.csv")


x = veriler.iloc[:,1:4]
y = veriler.iloc[:,4:]


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.33,random_state = 0)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)

#Logistic sınıflandırma
from sklearn.linear_model import LogisticRegression
logr = LogisticRegression(random_state = 0)
logr.fit(X_train,y_train)
y_pred = logr.predict(X_test)
print(y_pred)
print(y_test)
 
# confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print("Logistic sınıflandırma")
print(cm)


# K Nearest Neigbors sınıflandırma
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1,metric = 'minkowski')
knn.fit(X_train,y_train)
y_pred = knn.predict(X_test)
cm = confusion_matrix(y_test,y_pred)
print("Knn sınıflandırma")
print(cm)


# SVC sınıflandırma
from sklearn.svm import SVC
svc= SVC(kernel='rbf')
svc.fit(X_train,y_train)
y_pred = svc.predict(X_test)
cm = confusion_matrix(y_test,y_pred)
print("SVC sınıflandırma")
print(cm)


#Naive Bayes Sınıflandırma
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train,y_train)
y_pred = gnb.predict(X_test)
cm = confusion_matrix(y_test,y_pred)
print("Naive Bayes sınıflandırma")
print(cm)


# decision Tree sınıflandırma algoritması
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(criterion='entropy')
dtc.fit(X_train,y_train)
y_pred = dtc.predict(X_test)
cm = confusion_matrix(y_test,y_pred)
print("Decision Tree sınıflandırma")
print(cm)


#Random Forest Sııflandırma
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=10,criterion='entropy')
rfc.fit(X_train,y_train)
y_pred = rfc.predict(X_test)
cm = confusion_matrix(y_test,y_pred)
print("Random Forest sınıflandırma")
print(cm)

































