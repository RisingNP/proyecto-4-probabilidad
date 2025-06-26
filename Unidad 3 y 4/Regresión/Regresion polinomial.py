# -*- coding: utf-8 -*-
"""
Created on Sun Apr 27 11:44:02 2025

@author: fjose
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#from tabulate import tabulate #para mostrar el dataframe como una tabla 
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from sklearn.preprocessing import PolynomialFeatures


#%% cargar datos 

df = pd.read_excel("DRXcrudo.xlsx")

#%% grafiquemos 

fig, ax = plt.subplots(figsize=(5,5))

ax.scatter(df['Tetha'],df['counts'])
ax.set_xlabel(r'2$\theta$')
ax.set_ylabel('Counts')

#%% 
x = df['Tetha']
x = x.values.reshape(-1,1)
y = df['counts']


# Paso 1: crear polinomio
poly = PolynomialFeatures(degree=3) # se genera una clase polinomio 

x_poly = poly.fit_transform(x)#se genera una tranformación de x, para las diferentes potencias, 0, 1, 2, 3...

# Paso 2: crear el modelo lineal y se ajusta con la x transformada

modelo_f = LinearRegression().fit(x_poly,y)

# ajustar el modelo

#y_pre=modelo_f.predict(x_poly)


"""
Los coeficientes y el intercepto están en modelo_f.coef_ y modelo_f.intercept.
"""
y_pre = np.sum(x_poly*modelo_f.coef_,axis=1)+modelo_f.intercept_ # este sería el modelo

fig, ax = plt.subplots(figsize=(5,5))

ax.scatter(df['Tetha'],df['counts'])
ax.set_xlabel(r'2$\theta$')
ax.set_ylabel('Counts')
ax.plot(x,y_pre,'r')



#%% regressión spline 
from sklearn.preprocessing import SplineTransformer

# crear una clase para el polinomio a tramos 
spline_transformer = SplineTransformer(degree=2, n_knots=50)

X_splines = spline_transformer.fit_transform(x)# se tranforma la x para los difentes 
#polinomios, estos son polinomios que son diferente de cero, solo entre degree +1 knots

modelo_f = LinearRegression().fit(X_splines,y)

#y_pre=modelo_f.predict(X_splines)
"""
Los coeficientes y el intercepto están en modelo_f.coef_ y modelo_f.intercept.
"""
y_pre = np.sum(X_splines*modelo_f.coef_,axis=1)+modelo_f.intercept_ # este sería el modelo

fig, ax = plt.subplots(figsize=(5,5))

ax.scatter(df['Tetha'],df['counts'])
ax.set_xlabel(r'2$\theta$')
ax.set_ylabel('Counts')
ax.plot(x,y_pre,'r')

