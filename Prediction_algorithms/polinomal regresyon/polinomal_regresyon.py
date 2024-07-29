# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 18:50:13 2020

@author: sadievrenseker
"""

#1.kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#2.veri onisleme
#2.1.veri yukleme
veriler = pd.read_csv(r"C:\Users\merts\.spyder-py3\makine-ogrenmesi\polinomal regresyon\maaslar.csv")
#pd.read_csv("veriler.csv")
#test
print(veriler)

#data frame dilimleme
x=veriler.iloc[:,1:2] #egitim seviyesini aldk
y=veriler.iloc[:,2:] # maaşı aldık
X=x.values #numpy array e dönüştürdük
Y=y.values

#Modeller

#Linear regression yaptık burada sadece
from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression()
lin_reg.fit(X,Y)


#polinomal regresyon oluşturma
from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(degree=2) #2. dereceden oluşturduk

#bu yolla direkt poly haline dönüştürebiliyoruz.
x_poly=poly_reg.fit_transform(X)
print(x_poly)

#yine lineer reg gibi fit ediyoruz zaten olay x in exponantial olmasında artık
lin_reg2=LinearRegression()
lin_reg2.fit(x_poly, y)

# derecesi farklı polinomal regresyon
poly_reg=PolynomialFeatures(degree=4) #4. dereceden oluşturduk

#bu yolla direkt poly haline dönüştürebiliyoruz.
x_poly=poly_reg.fit_transform(X)
print(x_poly)

#yine lineer reg gibi fit ediyoruz zaten olay x in exponantial olmasında artık
lin_reg2=LinearRegression()
lin_reg2.fit(x_poly, y)


# görselleştirmeler
plt.scatter(X,Y,color="red")
plt.plot(x,lin_reg.predict(X)) # doğrusal regresyon grafiği

plt.scatter(X,Y,color="red")
plt.plot(x, lin_reg2.predict(poly_reg.fit_transform(X)))
plt.show() #2. derece grafiği

plt.scatter(X,Y,color="red")
plt.plot(x, lin_reg2.predict(poly_reg.fit_transform(X)))
plt.show() #4. derece daha başarılı grafikten görüldüğü gibi







