# -*- coding: utf-8 -*-
"""
Created on Sun Jun 15 07:17:45 2025

@author: fjose
"""
#%%

import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from sklearn.datasets import make_blobs
#from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay, precision_recall_curve
import numpy as np
#from sklearn.model_selection import train_test_split
from sklearn.mixture import GaussianMixture
#%%
X, _ = make_blobs(400, 2, centers=4, random_state=0, cluster_std=0.6) 
elip = [[1,2],[2,1]]
X_elip = np.dot(X, elip)

plt.scatter(X_elip[:, 0], X_elip[:, 1], s=50,c=[100/255,0/255,255/255]) #visualizamos los datos entrenamiento
#plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, s=30, cmap='coolwarm') # datos de testeo

#%% 

model = GaussianMixture(n_components=4,random_state=42)

model = model.fit(X_elip)

y = model.predict(X_elip)

plt.scatter(X_elip[:, 0], X_elip[:, 1], s=50,c=y, cmap='viridis')

y_proba = model.predict_proba(X_elip)


#%%
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_digits
digits = load_digits()

X = digits.data 

pca = PCA().fit(X)

plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance');

#%% 12 podría ser un buen numero entonces 

pca_opt=PCA(n_components=12).fit(X)

X_new =pca_opt.transform(X)

#ahora tengo un arreglo dimensional de 12 variables, no 64,
# que explica el 80% de la variabilidad original. 
# %%
