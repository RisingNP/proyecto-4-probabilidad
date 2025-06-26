# -*- coding: utf-8 -*-
"""
Created on Mon May 26 19:36:15 2025

@author: fjose
"""

import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from sklearn.datasets import make_blobs
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay, precision_recall_curve
import numpy as np
from sklearn.model_selection import train_test_split

#%%
X, y = make_blobs(400, 2, centers=4, random_state=0, cluster_std=1.5) 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=10)


#X serían las variables clasificadora, 
#y son las clases

#grafiquemos para ver como están los datos
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, s=50, cmap='rainbow'); #visualizamos los datos entrenamiento
#plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, s=30, cmap='coolwarm') # datos de testeo

#%%
tree = DecisionTreeClassifier().fit(X_train, y_train)

def visualize_classifier(model, X, y, ax=None, cmap='rainbow'):
    ax = ax or plt.gca()
    # Plot the training points
    ax.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=cmap,
               clim=(y.min(), y.max()), zorder=3)
    ax.axis('tight')
    ax.axis('off')
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    # fit the estimator
    model.fit(X, y)
    xx, yy = np.meshgrid(np.linspace(*xlim, num=200),
                         np.linspace(*ylim, num=200))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    # Create a color plot with the results
    n_classes = len(np.unique(y))
    contours = ax.contourf(xx, yy, Z, alpha=0.3,
                           levels=np.arange(n_classes + 1) - 0.5,
                           cmap=cmap, clim=(y.min(), y.max()),
                           zorder=1)
    ax.set(xlim=xlim, ylim=ylim)
    
visualize_classifier(DecisionTreeClassifier(max_depth=4), X_train, y_train)


#%% partición de los datos 
#podemos usar la función de partición de datos de entrenamiento y prueba
#con diferentes random_state

X_train1, X_test, y_train1, y_test = train_test_split(X_train, y_train, test_size=0.5,random_state=10)
X_train2, X_test, y_train2, y_test = train_test_split(X_train, y_train, test_size=0.5,random_state=4)
X_train3, X_test, y_train3, y_test = train_test_split(X_train, y_train, test_size=0.5,random_state=42)

visualize_classifier(DecisionTreeClassifier(max_depth=4), X_train1, y_train1)
#visualize_classifier(DecisionTreeClassifier(max_depth=4), X_train2, y_train2)
#visualize_classifier(DecisionTreeClassifier(max_depth=4), X_train3, y_train3)

#%% random forest 
from sklearn.ensemble import RandomForestClassifier

#%%
#generamos un modelo randomforest, se le define un estado aleatorio para que
#sea reproducible, y se define el numero de arboles del bosque
model = RandomForestClassifier(n_estimators=100, random_state=0, max_samples=0.8).fit(X_train,y_train)

#visualize_classifier(model, X_train, y_train);

#%% validación 
#X_test, y_test = make_blobs(200, 2, centers=4, random_state=0, cluster_std=1.5) 
#plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, s=50, cmap='rainbow'); #visualizamos los datos entrenamiento
#plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, s=30, cmap='coolwarm') # datos de testeo

y_pred =model.predict(X_test)

visualize_classifier(model, X_train, y_train);
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, s=30, cmap='rainbow') # datos de testeo

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

