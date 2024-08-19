import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

veriler=pd.read_csv(r"C:\Users\merts\.spyder-py3\Makine-Ogrenmesi\Lessons\Association_rule_algorithms\sepet.csv",header=None)

t= []
for i in range(0,7501):
    t.append([str(veriler.values[i,j]) for j in range (0,20)])
    
from apyori_lib import apriori
kurallar=apriori(t,min_support=0.01,min_confidence=0.2,min_lift=3,min_length=2)
print(list(kurallar))
    
    
    
             
