import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

veriler=pd.read_csv(r"C:\Users\merts\.spyder-py3\makine-ogrenmesi\Classification_algorithms\Logicistic_regression\veriler.csv")
print(veriler)

x=veriler.iloc[:,1:4].values ## bağımsız değişkenler
y=veriler.iloc[:,4:].values #bağımlı değişken

#verilerin eğitimi
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.33,random_state=0)

#verilerin ölçeklenmesi
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()

X_train=sc.fit_transform(x_train) # fit öğretme yöntemi yani burda öğrenip uyguluyor
X_test=sc.transform(x_test) # transform uygulama yöntemi öğrendiğini uyguluyor fit e gerek yok

from sklearn.linear_model import LogisticRegression
logr=LogisticRegression(random_state=0)
logr.fit(X_train,y_train)

y_pred=logr.predict(X_test)
print(y_pred)


from sklearn.metrics import confusion_matrix
#diagon değerler (soldan sağ aşşağıya doğru) doğru sayısı, dışındakiler yanlış sonuçlar
#büyük veri gruplarına bakmak için çok kullanışlı
cm=confusion_matrix(y_test, y_pred)
print(cm)
