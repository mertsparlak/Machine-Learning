import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""
reinforced learningden önce random ile bakıyoruz aradaki farkı görebilmek için
10000 sample içinden 10 farklı reklamdan sadece birisine tıklanılıyor hep
random ile çalışınca tabi normal olarak 1000 civarı bulabiliyoruz 10000 içinden
"""


veriler=pd.read_csv(r"C:\Users\merts\.spyder-py3\Makine-Ogrenmesi\Lessons\Reinforced_learning\Ads_CTR_Optimisation.csv")

#Random Selection
"""
import random
N = 10000
d=10
toplam=0
secilenler=[]

for n in range(0,N):
    ad=random.randrange(d)
    secilenler.append(ad)
    odul=veriler.values[n,ad]
    toplam=toplam+odul

# hangi ilanların seçildiğine grafikle daha rahat bakabiliyoruz
plt.hist(secilenler)
plt.show()
"""

import math
#UCB
N = 10000 #10000 tıklama
d=10 #10 ilanımız var
tiklamalar= [0]*d # o ana kadarki tıklamalar
oduller = [0]*d #ilk basta butun ilanların odulu 0
toplam=0 #toplam odul
secilenler=[]
for n in range(0,N):
    ad=0 #seçilen ilan
    max_ucb=0
    for i in range(0,d):
        if(tiklamalar[i]>0):
            ortalama=oduller[i]/tiklamalar[i] #o ana kadarki her reklamın ortalam odülü formülü
            delta=math.sqrt(3/2* math.log(n)/tiklamalar[i])# ucb için yukarı aşşağı oynama potansiyeli formülü
            ucb=ortalama+delta
        else:
            ucb=N*10
        if max_ucb < ucb: #max'tan büyük ucb çıktı
            max_ucb=ucb #en yüksek ucb değerliyi alırız
            ad=i
            
    secilenler.append(ad)
    tiklamalar[ad]=tiklamalar[ad]+1
    odul=veriler.values[n,ad]
    oduller[ad]=oduller[ad]+odul
    toplam=toplam+odul

print("Toplam Odul:")
print(toplam)

plt.hist(secilenler)
plt.show()

# random selection'a göre 1000 daha çok tıklanacak şekilde daha iyi sonuç elde etti
