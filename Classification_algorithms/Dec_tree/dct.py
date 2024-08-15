import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

veriler=pd.read_csv(r"C:\Users\merts\.spyder-py3\Makine-Ogrenmesi\Lessons\Classification_algorithms\Naive_Bayes\veriler.csv")
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


from sklearn.neighbors import KNeighborsClassifier

knn=KNeighborsClassifier(n_neighbors=1,metric="minkowski")
# neighbor sayısı 5 iken 1 doğru 7 yanlış vardı neighbor sayısını 1 e düşürünce bu sayısı 7 doğru 1 yanlışa düştü
# neighbor sayısının yüksekliği her zaman en başarılı sonuca ulaştırmaz
#başka metricler de denenebilir.
knn.fit(X_train, y_train) # X_train den y_train öğretiyor

y_pred=knn.predict(X_test)

cm=confusion_matrix(y_test, y_pred)

from sklearn.naive_bayes import GaussianNB

## Naive bayes içinde multinomal ve bernoulli naive bayes olmak üzere 2 tane daha naive bayes kullanımı var
## Multinomal araba markaları, okullar gibi çoklu verileri integer değer atayarak yani 0,1,2,3 gibi, kullanılır
##bernoulli naive bayes ise binary formdadır yani 0 ve 1 e göre sınıflar
## kullandığımız gauss naive bayes ise sürekli gelen veriler için kullanılır ve sayılar tam sayı olmak zorunda da değil

gnb=GaussianNB()
gnb.fit(X_train,y_train)

y_pred=gnb.predict(X_test)
cm=confusion_matrix(y_test, y_pred)
print("GNB")
print(cm)


from sklearn.tree import DecisionTreeClassifier
dtc=DecisionTreeClassifier(criterion="entropy")

dtc.fit(X_train,y_train)
y_pred=dtc.predict(X_test)

dtc=confusion_matrix(y_test, y_pred)
print("DTC")
print(dtc)



