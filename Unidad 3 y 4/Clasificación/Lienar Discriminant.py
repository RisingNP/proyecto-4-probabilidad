# -*- coding: utf-8 -*-
"""
Created on Thu May 15 13:20:09 2025

@author: fjose
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from sklearn.datasets import make_blobs
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

#%% Datos

#generamos 200 datos que estén distribuidos en dos grupos, y dos dimensiones
#para mostrar como funciona el algoritmo, pero los datos normalmente serían
#de alguna base de datos que deseamos analizar. 
 
X, y = make_blobs(200, 2, centers=2, random_state=2, cluster_std=1.5) 

#X serían las variables clasificadora, 
#y son las clases

#grafiquemos para ver como están los datos
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='RdBu'); #visualizamos los datos

#%%
model = LinearDiscriminantAnalysis().fit(X,y) # luego generamos el modelo y lo ajustamos

#generemos otros datos aleatorios para clasificarlos con el modelo. 
rng = np.random.RandomState(0)#creamos un generador random 
#creamos otros datos X nuevos, es descir los features, para asignar el label
Xnew = [-6, -14] + [14, 18] * rng.rand(2000, 2)

ynew = model.predict(Xnew) #predecimos el label para los nuevos X
 
plt.scatter(X[:, 0], X[:, 1], c=y, s=100, cmap='RdBu') #graficamos los datos iniciales
#graficamos los nuevos dtos, con transparencia del 30% (alpha)
plt.scatter(Xnew[:, 0], Xnew[:, 1], c=ynew, s=100, cmap='RdBu', alpha=0.3) 

