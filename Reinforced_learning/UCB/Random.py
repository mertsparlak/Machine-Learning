
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

    
        
