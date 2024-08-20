import pandas as pd
import matplotlib.pyplot as plt
import random

veriler=pd.read_csv(r"C:\Users\merts\.spyder-py3\Makine-Ogrenmesi\Lessons\Reinforced_learning\Thompson_sampling\Ads_CTR_Optimisation.csv")

N = 10000 #10000 tıklama
d=10 #10 ilanımız var
oduller = [0]*d #ilk basta butun ilanların odulu 0
toplam=0 #toplam odul
secilenler=[]
birler=[0]*d
sifirlar=[0]*d

for n in range(0,N):
    ad=0 #seçilen ilan
    max_th=0
    for i in range(0,d):
        rasbeta=random.betavariate(birler[i]+1,sifirlar[i]+1)
        if rasbeta>max_th:
            max_th=rasbeta
            ad=i
    secilenler.append(ad)
    odul=veriler.values[n,ad]
    
    if odul==1:
        birler[ad]=birler[ad]+1
    else:
        sifirlar[ad]=sifirlar[ad]+1

    oduller[ad]=oduller[ad]+odul
    toplam=toplam+odul

print("Toplam Odul:")
print(toplam)

plt.hist(secilenler)
plt.show()

# thampson sampling ucb'ye göre daha iyi sonuç çıkardı yaklaşık 400 fazla
