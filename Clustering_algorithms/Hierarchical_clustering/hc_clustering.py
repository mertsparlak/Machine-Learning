import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

veriler=pd.read_csv(r"C:\Users\merts\.spyder-py3\Makine-Ogrenmesi\Lessons\Clustering_algorithms\K-means\musteriler.csv")
X=veriler.iloc[:,3:].values # x'e veri atadık hacim ve maaşı 

from sklearn.cluster import KMeans
kmeans=KMeans(n_clusters=3,init="k-means++")
kmeans.fit(X)

print(kmeans.cluster_centers_)

sonuclar=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i,init="k-means++",random_state=123)
    kmeans.fit(X)
    sonuclar.append(kmeans.inertia_)
"""
plt.plot(range(1,11), sonuclar)
plt.show()
"""
## grafikten dirsek noktasına bakıp o noktayı almak daha sağlıklı olucaktır 
#yani burada 2,3,4 alınabilir eğime bakarak

from sklearn.cluster import AgglomerativeClustering
ac=AgglomerativeClustering(n_clusters=3,affinity="euclidean",linkage="ward")
Y_tahmin=ac.fit_predict(X)
print(Y_tahmin) #3 clustera atandılar ve print ile çıkan output hangi clusterda olduğunu gösteriyor verilerin

plt.scatter(X[Y_tahmin==0,0],X[Y_tahmin==0,1],s=100,c="red")
plt.scatter(X[Y_tahmin==1,0],X[Y_tahmin==1,1],s=100,c="blue")
plt.scatter(X[Y_tahmin==2,0],X[Y_tahmin==2,1],s=100,c="green")
plt.scatter(X[Y_tahmin==3,0],X[Y_tahmin==3,1],s=100,c="yellow")
plt.show()

import scipy.cluster.hierarchy as sch
dendrogram=sch.dendrogram(sch.linkage(X,method="ward"))
plt.show() ## dendrogramlar arası mesafe en çok 2de olduğu için k=2 alınabilir veya 4 de uzaklığı iyi gibi o da olabilir
# k-meansde de 2 dirsek noktasıydı 2 ve 4 alınabilir demiştik


