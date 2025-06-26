# -*- coding: utf-8 -*-
"""
Created on Sat May 24 07:33:00 2025

@author: fjose
"""



import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from sklearn.datasets import make_blobs
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay, precision_recall_curve
import numpy as np


#%%
#Data train
x1,y1 =make_blobs(10, 2, centers=2, random_state=2, cluster_std=2.5) 
x2,y2 =  make_blobs(200, 2, centers=1, random_state=2, cluster_std=2.5) 
X_train = np.concatenate((x1,x2),axis=0)
y_train = np.concatenate((y1,y2),axis=0)

#data test

x3,y3 =make_blobs(10, 2, centers=2, random_state=2, cluster_std=3) 
x4,y4 =  make_blobs(200, 2, centers=1, random_state=2, cluster_std=3) 
X_test = np.concatenate((x3,x4),axis=0)
y_test = np.concatenate((y3,y4),axis=0)

plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, s=50, cmap='RdBu');
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, s=30, cmap='coolwarm') # datos de testeo


#%% #generación del modelo

model = GaussianNB().fit(X_train,y_train) # luego generamos el modelo y lo ajustamos a lo datos

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

#%% ROC curve

RocCurveDisplay.from_estimator(model, X_test, y_test)
#%% Precision Recall Curve 

y_scores = probs[:,1]

prec, rec, thresholds = precision_recall_curve(y_test, y_scores)

fig, ax = plt.subplots()

ax.plot(rec,prec)
ax.set_xlabel('Recall')
ax.set_ylabel('Precisión')
ax.plot([0.6,1],[0.6,1],'r')
ax.set_aspect(1)

# para elegir el umbral se puede elegir el punto que maximice la diferentes entre Precisión y Recall 
# o que minimice la distancia al punto (1,0)
# o por ensayo y error, el valor que mejor al máx la clasificiación de 1 si el costo de clasificarlos mall es muy alto.
#%%
umbral = 0.47

y_pred = (y_scores>= umbral).astype(int)


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
