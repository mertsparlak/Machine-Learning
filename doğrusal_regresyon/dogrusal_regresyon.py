#1.kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#2.veri onisleme
#2.1.veri yukleme
veriler = pd.read_csv(r"C:\Users\merts\.spyder-py3\makine-ogrenmesi\satislar.csv")
#pd.read_csv("veriler.csv")
#test
print(veriler)

satislar=veriler[["Satislar"]]
print(satislar)
aylar=veriler[["Aylar"]]
print(aylar)

"""
x = veriler.iloc[:,1:4].values #bağımsız değişkenler
y = veriler.iloc[:,4:].values #bağımlı değişken
"""

#verilerin egitim ve test icin bolunmesi
from sklearn.model_selection import train_test_split

x_train, x_test,y_train,y_test = train_test_split(aylar,satislar,test_size=0.33, random_state=0)
"""
#verilerin olceklenmesi
from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)

Y_train = sc.fit_transform(x_train)
Y_test = sc.transform(x_test)
"""

#model inşası (lineer regresyon)
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x_train,y_train)

tahmin=lr.predict(x_test)

x_train=x_train.sort_index()
y_train=y_train.sort_index()

plt.plot(x_train, y_train)
plt.plot(x_test,tahmin)
plt.title("aylara gore satis")
plt.xlabel("aylar")
plt.ylabel("satislar")

