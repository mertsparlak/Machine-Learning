# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 18:50:13 2020

@author: sadievrenseker
"""

#1.kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# veri yukleme
veriler = pd.read_csv(r"C:\Users\merts\.spyder-py3\makine-ogrenmesi\sup_vec_reg\maaslar.csv")

x = veriler.iloc[:,1:2]
y = veriler.iloc[:,2:]
X = x.values
Y = y.values


#linear regression
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,Y) #fit etmek train etmeye yarayan algoritmadır
"""
plt.scatter(X,Y,color='red')
plt.plot(x,lin_reg.predict(X), color = 'blue')
plt.show()
"""

#polynomial regression
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
x_poly = poly_reg.fit_transform(X)
print(x_poly)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly,y)
"""
plt.scatter(X,Y,color = 'red')
plt.plot(X,lin_reg2.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.show()
"""
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
x_poly = poly_reg.fit_transform(X)
print(x_poly)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly,y)
"""
plt.scatter(X,Y,color = 'red')
plt.plot(X,lin_reg2.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.show()
"""
#tahminler

print(lin_reg.predict([[11]]))
print(lin_reg.predict([[6.6]]))

print(lin_reg2.predict(poly_reg.fit_transform([[6.6]])))
print(lin_reg2.predict(poly_reg.fit_transform([[11]])))




#verilerin olceklenmesi
from sklearn.preprocessing import StandardScaler

sc1=StandardScaler()

x_olcekli = sc1.fit_transform(X)

sc2=StandardScaler()

y_olcekli = sc1.fit_transform(Y)

from sklearn.svm import SVR

svr_reg=SVR(kernel="rbf") ## başka kerneller da kullanılabilir ama en uyanı rbf 
svr_reg.fit(x_olcekli,y_olcekli)

"""
plt.scatter(x_olcekli,y_olcekli)
plt.plot(x_olcekli,svr_reg.predict(x_olcekli))
"""

## decision tree tahmini
from sklearn.tree import DecisionTreeRegressor
r_dt=DecisionTreeRegressor(random_state=0)
r_dt.fit(X,Y)

Z=X+0.5
K=X-0.4

"""
plt.scatter(X,Y,color="red")
plt.plot(X,r_dt.predict(X),color="blue")
plt.plot(X,r_dt.predict(Z),color="green")
plt.plot(X,r_dt.predict(K),color="yellow") ## decision tree de üçü de aynı çizgi üstünde kaldı
"""


print("\n")
from sklearn.ensemble import RandomForestRegressor
rf_reg=RandomForestRegressor(n_estimators=10,random_state=0) # n_estimators kaç desicion treeye bölüceğini belirler
rf_reg.fit(X,Y.ravel())
print(rf_reg.predict([[6.5]]) )#6.5 eğitim seviyesi için maaş tahmini
print(rf_reg.predict([[6.6]]))

"""
plt.scatter(X,Y,color="red")

plt.plot(X,rf_reg.predict(X), color="blue")

plt.plot(X,rf_reg.predict(Z), color="green")

plt.plot(X,rf_reg.predict(K), color="yellow")
 ##rand forestta ise Z ve K ile farklı tahminler yapılabildi çünkü frameleri böldük 
 ## ve farklı sonuçlar verebildiler algoritmamız da bir karar verdi buna göre.
 ## rand forest çoğu zaman decision tree ye göre çok daha etkilidir.
"""

from sklearn.metrics import r2_score
print("Random forest r^2 degeri:")
print(r2_score(Y,rf_reg.predict(X)))






