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

plt.plot(range(1,11), sonuclar)
## grafikten dirsek noktasına bakıp o noktayı almak daha sağlıklı olucaktır 
#yani burada 2,3,4 alınabilir eğime bakarak