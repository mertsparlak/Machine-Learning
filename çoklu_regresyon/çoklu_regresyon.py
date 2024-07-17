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
veriler = pd.read_csv(r"C:\Users\merts\.spyder-py3\makine-ogrenmesi\çoklu_regresyon\veriler.csv")
#pd.read_csv("veriler.csv")
#test
print(veriler)

#kategorik verileri numerikleştirme işlemi
ulke=veriler.iloc[:,0:1].values
print(ulke)

from sklearn import preprocessing
le=preprocessing.LabelEncoder()

ulke[:,0]=le.fit_transform(veriler.iloc[:,0])
print(ulke)

ohe=preprocessing.OneHotEncoder() #encoding yapıyoruz 
ulke=ohe.fit_transform(ulke).toarray()
print(ulke)

c=veriler.iloc[:,0:1].values
print(c)

from sklearn import preprocessing
le=preprocessing.LabelEncoder()

c[:,-1]=le.fit_transform(veriler.iloc[:,-1])
print(c)

ohe=preprocessing.OneHotEncoder()
c=ohe.fit_transform(c).toarray()
print(c)

Yas=veriler.iloc[:,1:4]
print(Yas)
#numpy dizileri ile dataframe oluşturma
sonuc=pd.DataFrame(data=ulke,index=range(22),columns=["fr","tr","us"])
print(sonuc)

sonuc2=pd.DataFrame(data=Yas,index=range(22),columns=["boy","kilo","yas"])
print(sonuc2)

cinsiyet=veriler.iloc[:,-1].values
print(cinsiyet)

sonuc3=pd.DataFrame(data=cinsiyet,index=range(22),columns=["cinsiyet"])
print(sonuc3)


s=pd.concat([sonuc,sonuc2],axis=1)
print(s)

s2=pd.concat([s,sonuc3],axis=1)
print(s2)























