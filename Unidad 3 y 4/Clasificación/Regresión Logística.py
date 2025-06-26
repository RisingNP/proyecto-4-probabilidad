# -*- coding: utf-8 -*-
"""
Created on Sun May 18 14:52:49 2025

@author: fjose
"""

from sklearn.datasets import load_iris # para cargar una base de datos de skleanr
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd

#cargamos la base de datos
iris = load_iris()
#extreamos los datos

X = iris.data

#hay tres tipos de flores, convirtamos el problema en uno de dos clases
# es Iris Virginica o no lo es, esta corresponde a la clase 2, entonces 
# y será uno si es 2, y 0 en otro caso. 
y = (iris.target == 2).astype(int)  # Clasificar si es "Iris virginica" (1) o no (0)

#dividamos los datos entre datos de entrenamiento (70%) y prueba (30%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#creamos el modelo de regresión logística 
model = LogisticRegression()

# ajustamos el modelo a los datos de entrenamiento. 
model.fit(X_train, y_train)

#usamos los datos de prueba para predecir la clase 
y_pred = model.predict(X_test)
#veamos qué tan bien la fue al modelo:
    
print("Reporte de clasificación:")

clasif = pd.DataFrame({
    'Y':y_test,
    'Y_Pred':y_pred
    })

from tabulate import tabulate # para tabular los dataframe

print(tabulate(clasif,headers='keys',tablefmt='heavy_grid'))

# también podemos estar interesados en la probabilidad de que los features
# correspondan a una clase o a la otra 
#%%
probs = model.predict_proba(X_test)
probs = probs.round(3)
print(probs)