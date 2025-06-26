# -*- coding: utf-8 -*-
"""
Created on Mon May 26 12:45:25 2025

@author: fjose
"""


import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from sklearn.datasets import make_blobs
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay, precision_recall_curve
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import zscore
from sklearn.model_selection import validation_curve, learning_curve

#%%
##Data train
x1,y1 =make_blobs(50, 2, centers=2, random_state=2, cluster_std=3.5) 
x2,y2 =  make_blobs(200, 2, centers=1, random_state=2, cluster_std=3.5) 
X_train = zscore(np.concatenate((x1,x2),axis=0))
y_train = np.concatenate((y1,y2),axis=0)

#data test

x3,y3 =make_blobs(50, 2, centers=2, random_state=2, cluster_std=2) 
x4,y4 =  make_blobs(200, 2, centers=1, random_state=2, cluster_std=2) 
X_test = zscore(np.concatenate((x3,x4),axis=0))
y_test = np.concatenate((y3,y4),axis=0)

plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, s=50, cmap='RdBu');
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, s=30, cmap='coolwarm') # datos de testeo



#%% #generación del modelo

model = KNeighborsClassifier(n_neighbors=5,metric='euclidean')

model = model.fit(X_train,y_train) # luego generamos el modelo y lo ajustamos a lo datos

#generemos otros datos aleatorios para clasificarlos con el modelo, es decir, 

y_pred = model.predict(X_test) #predecimos el label para los nuevos X usando el modelo

probs = model.predict_proba(X_test)

#%% Reporta de clasificación
#presente las metricas de: accuracy, recall, precision y F1

print("Reporte de clasificación:")

print(classification_report(y_test, y_pred))

#Matriz de confusión 

conf_matrix = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
disp.plot()
disp.ax_.grid(0)
#disp.ax_.axes.texts._axes.set_xlabel(fontsize=20)
plt.xlabel('Clase Predicha',fontsize=16)
plt.ylabel('Clase Real',fontsize=16)
plt.show()

#%% curva de validación para selección del K 

k = np.arange(1, 21) # veamos para diferentes valores de grados

train_score, val_score = validation_curve(KNeighborsClassifier(n_neighbors=5,metric='euclidean'), X_train, y_train,
                                          param_name='n_neighbors',scoring='f1',
                                          param_range=k, cv=5) # cv=5 hace un k-fold CV de 5 folds
#para saber las opciones de param_name se requeire conocer las opciones que tiene KNeighborsClassifier con el método model.get_params()
#validation_curve, retorna la metrica que se asinge en scoring, para los datos de entrenamiento y de validación

plt.plot(k, np.median(train_score, axis=1), color='blue', 
         label='training score') #graficamos el promedios del parámetro de entrenamietno, y el de validación 

plt.plot(k, np.median(val_score, axis=1), color='red', 
         label='validation score') #graficamos el promedio de los parámetros de validación o prueba
plt.legend(loc='best')
#plt.ylim(0, 1)
plt.xlabel('K')
plt.ylabel('f1')

better_K = k[np.argmax(np.median(val_score,axis=1))] #buscamos dónde ocurre el máx en la curva
# de la curva de validación

print(f'El mejor valor de K es: {better_K}')

#%% Veamos que pasa si los datos tienen diferentes escalas,
#Data train
x1,y1 =make_blobs(50, 2, centers=2, random_state=2, cluster_std=3.5) 
x2,y2 =  make_blobs(200, 2, centers=1, random_state=2, cluster_std=3.5) 
X_train = np.concatenate((x1,x2),axis=0)
X_train[:,0] = X_train[:,0]+1000
y_train = np.concatenate((y1,y2),axis=0)

#data test

x3,y3 =make_blobs(50, 2, centers=2, random_state=2, cluster_std=2) 
x4,y4 =  make_blobs(200, 2, centers=1, random_state=2, cluster_std=2) 
X_test = np.concatenate((x3,x4),axis=0)
X_test[:,0] = X_test[:,0]+1000
y_test = np.concatenate((y3,y4),axis=0)

plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, s=50, cmap='RdBu');
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, s=30, cmap='coolwarm') # datos de testeo

#%% z-score

X_train=zscore(X_train)
X_test=zscore(X_test)

plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, s=50, cmap='RdBu');
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, s=30, cmap='coolwarm') # datos de testeo
#%%
#generación del modelo

model = KNeighborsClassifier(n_neighbors=3,metric='euclidean')

model = model.fit(X_train,y_train) # luego generamos el modelo y lo ajustamos a lo datos

#generemos otros datos aleatorios para clasificarlos con el modelo, es decir, 

y_pred = model.predict(X_test) #predecimos el label para los nuevos X usando el modelo

probs = model.predict_proba(X_test)

# Reporta de clasificación
#presente las metricas de: accuracy, recall, precision y F1

print("Reporte de clasificación:")

print(classification_report(y_test, y_pred))

#Matriz de confusión 

conf_matrix = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
disp.plot()
disp.ax_.grid(0)
#disp.ax_.axes.texts._axes.set_xlabel(fontsize=20)
plt.xlabel('Clase Predicha',fontsize=16)
plt.ylabel('Clase Real',fontsize=16)
plt.show()

