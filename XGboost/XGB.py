import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.compose import ColumnTransformer

veriler=pd.read_csv(r"C:\Users\merts\OneDrive\Belgeler\GitHub\Deep-Learning\Churn_Modelling.csv")

X=veriler.iloc[:,3:13].values
y=veriler.iloc[:,13].values
#Ülke ve cinsiyetleri encode etmeliyiz şimdi çünkü YSA sadece 0-1 değer alır


labelencoder_X_1=LabelEncoder()
X[:,1]=labelencoder_X_1.fit_transform(X[:,1])
labelencoder_X_2=LabelEncoder()
X[:,2]=labelencoder_X_2.fit_transform(X[:,2])
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
X=X[:,1:]


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.33,random_state=0)

from xgboost import XGBClassifier
classifier=XGBClassifier()
classifier.fit(X_train,y_train)

y_pred=classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_pred,y_test)
print(cm)