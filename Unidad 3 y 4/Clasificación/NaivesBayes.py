# -*- coding: utf-8 -*-
"""
Created on Sun May 11 17:20:41 2025

@author: fjose
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from sklearn.datasets import make_blobs
from sklearn.naive_bayes import GaussianNB


#%% Datos

#generamos 200 datos que estén distribuidos en dos grupos, y dos dimensiones
#para mostrar como funciona el algoritmo, pero los datos normalmente serían
#de alguna base de datos que deseamos analizar. 
 
X, y = make_blobs(200, 2, centers=2, random_state=2, cluster_std=1.5) 

#X serían las variables clasificadora, 
#y son las clases

#grafiquemos para ver como están los datos
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='RdBu'); #visualizamos los datos


#%% #generación del modelo

model = GaussianNB().fit(X,y) # luego generamos el modelo y lo ajustamos a lo datos

#generemos otros datos aleatorios para clasificarlos con el modelo, es decir, 
#generemos una datos para probar el modelo.  

rng = np.random.RandomState(0)#creamos un generador random 

#creamos otros datos X nuevos, es descir los features, a los que vamos a 
#asignarles el label con el modelo

Xnew = [-6, -14] + [14, 18] * rng.rand(2000, 2)

ynew = model.predict(Xnew) #predecimos el label para los nuevos X usando el modelo
 
plt.scatter(X[:, 0], X[:, 1], c=y, s=100, cmap='RdBu') #graficamos los datos iniciales
#graficamos los nuevos dtos, con transparencia del 30% (alpha)
plt.scatter(Xnew[:, 0], Xnew[:, 1], c=ynew, s=100, cmap='RdBu', alpha=0.3) 

#podemos conocer la probabilidad de cada punto de estar en una clase
probs = model.predict_proba(Xnew)
probs = probs.round(3)
print(probs[-12:])#imprimamos las últimas 12 probabilides. 

#%%
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


"""
Supongamos que tenemos oraciones relacionadas con tres temas, y queremos
clasificarlas segun la frecuencia de aparaciones de palabras que se relacionan 
con el tema. 

Entonces cada oración debe ser representada por un vector con conteo de palabras
para poder modelarlo como una distribución multinomial
"""
# Ejemplos de oraciones
docs = [
    "El experimento fue exitoso en el laboratorio",
    "El jugador del equipo marcó dos goles en el partido",
    "El congreso aprobó una nueva ley, y el presidente la firmó",
    "Los científicos descubrieron una nueva partícula con un experimento",
    "El equipo ganó el partido con tres goles y con ello la copa del torneo",
    "El presidente dio un discurso sobre economía en el congreso",
    "científicos publicaron un artículo en una revista sobre un experimento",
    "El técnico hizo cambios estratégicos en el segundo tiempo del partido",
    "La campaña electoral para el congreso comenzó en todo el país con una nueva ley"
]

# Etiquetas (0 = ciencia, 1 = deportes, 2 = política)

labels = [0, 1, 2, 0, 1, 2, 0, 1, 2]

# Convertimos el texto en vectores de conteo de palabras
vectorizer = CountVectorizer() #instancia para convertir texto en frecuencia de ocurrencia de palabras

X = vectorizer.fit_transform(docs)  # X es una matriz  de conteos de cada palabra
# en cada oración. 
#imprimamos las palabras extraídas
print(vectorizer.get_feature_names_out())
#%%
#usemos algunos datos para entrenar u otro para probar

#esta función es de mucha utilidad para partir datos para prueba y entrenamiento
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.3, random_state=12)

#generemos el modelo y lo ajustamos con los datos de entrenamiento

model = MultinomialNB()
model.fit(X_train, y_train)

#probemos el modelo para clasificar los datos de prueba
y_pred = model.predict(X_test) 

print("Reporte de clasificación:")

clasif = pd.DataFrame({
    'Y':y_test,
    'Y_Pred':y_pred
    })

from tabulate import tabulate # para tabular los dataframe
print(tabulate(clasif,headers='keys',tablefmt='heavy_grid'))
