# -*- coding: utf-8 -*-
"""
Created on Thu May 22 11:03:40 2025

@author: fjose
"""
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from sklearn.datasets import make_blobs
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay, precision_recall_curve
import numpy as np

#%%
X_train, y_train = make_blobs(200, 2, centers=2, random_state=2, cluster_std=4) 
X_test, y_test = make_blobs(200, 2, centers=2, random_state=2, cluster_std=5) 
#X serían las variables clasificadora, 
#y son las clases

#grafiquemos para ver como están los datos
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, s=50, cmap='RdBu'); #visualizamos los datos entrenamiento
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, s=30, cmap='coolwarm') # datos de testeo

#%% Datos


X_train, y_train = make_blobs(200, 2, centers=2, random_state=2, cluster_std=4) 


X_test, y_test = make_blobs(200, 2, centers=2, random_state=2, cluster_std=5) 
#X serían las variables clasificadora, 
#y son las clases

#grafiquemos para ver como están los datos
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, s=50, cmap='RdBu'); #visualizamos los datos entrenamiento
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, s=30, cmap='coolwarm') # datos de testeo

#%% #generación del modelo

model = GaussianNB().fit(X_train,y_train) # luego generamos el modelo y lo ajustamos a lo datos

#generemos otros datos aleatorios para clasificarlos con el modelo, es decir, 

y_pred = model.predict(X_test) #predecimos el label para los nuevos X usando el modelo
 
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, s=100, cmap='RdBu') #graficamos los datos iniciales
#graficamos los nuevos dtos, con transparencia del 30% (alpha)
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, s=100, cmap='RdBu', alpha=0.3) 

#podemos conocer la probabilidad de cada punto de estar en una clase
probs = model.predict_proba(X_test)
probs = probs.round(3)
print(probs[-12:])#imprimamos las últimas 12 probabilides. 

#%% evaluación del modelo 

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

#%% Lift Chart

y_scores = model.predict_proba(X_test)[:, 1]# la probabilidad de ser 1, no de ser cero

# se debe ordenar los datos segun su probabilidad de ser clase 1, de forma descendente,
#del más probable al menos probable
# generamos los indices para ordenar
order = np.argsort(y_scores)[::-1]

#ordenamos y_test segun el orden de probabilidad 
y_sorted = y_test[order]
#odenamos la probabilidad segun su orden 
scores_sorted = y_scores[order]

# Luego se debe calcular proporción acumulada de positivos (recall acumulado)
#para esto se debe ir calculando el recall como cambia a medida que }
#se clasifican como 1 mas datos
 
cumulative_positives = np.cumsum(y_sorted) #se genera un vector con los acumulados

total_positives = np.sum(y_test) #se calculan todos los 1, se puede hacer así porque lo que no es 1 es 0 entonces no suma
recall_cumulative = cumulative_positives / total_positives# se calcula el recall acumulado 

#se genera un vector del la fracción de la muestra usada en para el recall 
fraction = np.arange(1, len(y_test) + 1) / len(y_test)

# Calcular lift: recall acumulado / diagonal(recall de una selección por asar) que es la misma fracción
lift = recall_cumulative / fraction

# Graficar lift chart
plt.figure()
plt.plot(fraction, lift)
plt.plot(fraction, recall_cumulative)
plt.xlabel("Fracción de la muestra evaluada")
plt.ylabel("Lift")
plt.title("Lift Chart")
plt.tight_layout()
plt.show()