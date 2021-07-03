# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 15:13:57 2020

@author: Atakan
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# veri yükleme
veriler = pd.read_csv("maaslar.csv")
x = veriler.iloc[:,1:2]
y = veriler.iloc[:,2:]
X = x.values
Y = y.values

# linear regression
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,Y)

plt.scatter(X,Y)
plt.plot(X,lin_reg.predict(X),color='red')
plt.title("Linear")
plt.show()


# polinom regression
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
x_poly = poly_reg.fit_transform(X)
print(x_poly)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly,y)
plt.scatter(X,Y)
plt.plot(X,lin_reg2.predict(poly_reg.fit_transform(X)),color = 'red')
plt.title("Polynom")
plt.show()

# tahminler
print(lin_reg.predict([[11]]))
print(lin_reg2.predict(poly_reg.fit_transform([[11]])))


# ölcekleme
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_olcekli = sc.fit_transform(X)
sc1 =StandardScaler()
y_olcekli = sc1.fit_transform(Y)


#SVR
from sklearn.svm import SVR
svr_reg = SVR(kernel='rbf')
svr_reg.fit(x_olcekli,y_olcekli)

plt.scatter(x_olcekli,y_olcekli,color='red')
plt.plot(x_olcekli,svr_reg.predict(x_olcekli))
plt.title("SRV")
plt.show()
print(svr_reg.predict([[11]]))


# tree decision
from sklearn.tree import DecisionTreeRegressor
r_dt = DecisionTreeRegressor(random_state = 0)
r_dt.fit(X,Y)

plt.scatter(X,Y,color="red")
plt.plot(X,r_dt.predict(X))
plt.title("Decison Tree")
plt.show()
print(r_dt.predict([[11]]))

# random forest tree
from sklearn.ensemble import RandomForestRegressor
rf_reg = RandomForestRegressor(n_estimators = 10,random_state = 0)
rf_reg.fit(X,Y.ravel())
plt.scatter(X,Y,color="red")
plt.plot(X,rf_reg.predict(X))
plt.title("random forest")
print(rf_reg.predict([[11]]))



#R2 hesaplanması
from sklearn.metrics import r2_score
print("Random Forest R2 Degeri:")
print(r2_score(Y,rf_reg.predict(X)))
print("Decision Tree R2 Degeri:")
print(r2_score(Y,r_dt.predict(X)))
























