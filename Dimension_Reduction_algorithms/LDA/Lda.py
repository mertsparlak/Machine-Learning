import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

veriler=pd.read_csv(r"C:\Users\merts\.spyder-py3\Makine-Ogrenmesi\Lessons\Dimension_Reduction_algorithms\LDA\Wine.csv")
X=veriler.iloc[:,0:13].values
y=veriler.iloc[:,13].values

#eğitim ve test kümelerinin bölünmesi
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

#Ölçekleme
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

from sklearn.decomposition import PCA
pca=PCA(n_components=2)

#veriyi 13 boyuttan 2 boyuta indirmiş olduk
X_train2=pca.fit_transform(X_train)
X_test2=pca.transform(X_test)

#pca'den önce gelen lojistik regresyon
from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression(random_state=0)
classifier.fit(X_train,y_train)

#pca'den sonra gelen lojistik regresyon
classifier2=LogisticRegression(random_state=0)
classifier2.fit(X_train2,y_train)

#tahminler
y_pred=classifier.predict(X_test)

y_pred2=classifier2.predict(X_test2)

from sklearn.metrics import confusion_matrix
print("gercek/PCA'ssz")
cm=confusion_matrix(y_test, y_pred)
print(cm)
# 13 kolon ile 36'da 0 hata ile bulabiliyoruz

print("gercek/PCA'lı")
cm2=confusion_matrix(y_test, y_pred2)
print(cm2)
#13'den 2 kolona indirdik ve 36'da sadce 1 hata yaptık

print("PCA'SIZ/PCA'lı")
cm3=confusion_matrix(y_pred, y_pred2)
print(cm3)

#LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda=LDA(n_components=2)

#LDA sınıfları ayrıştırdığından x_train ve y_train vermemiz gerekiyor pca aksine
X_train_lda=lda.fit_transform(X_train,y_train)
X_test_lda=lda.transform(X_test)

#lda'den sonra gelen lojistik regresyon
classifier_lda=LogisticRegression(random_state=0)
classifier_lda.fit(X_train_lda,y_train)

#LDA verisini tahmin et
y_pred_lda=classifier_lda.predict(X_test_lda)

#LDA sonrası/ orjinal
print("lda ve orjinal")
cm4=confusion_matrix(y_pred, y_pred_lda)
print(cm4)
# 2 boyuta indirdi ve %100 başarı da elde etti.

"""
Pca sınıf farkı gözetmez, Lda ise sınıflara göre ayırır bu yüzden LDA daha başarılı çıktı
"""

