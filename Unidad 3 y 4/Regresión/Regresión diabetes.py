# -*- coding: utf-8 -*-
"""
Created on Sat Apr 26 21:13:55 2025

@author: fjose
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#from tabulate import tabulate #para mostrar el dataframe como una tabla 
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score, KFold
#%% cargar datos y analizar correlaciones 

df_completo = pd.read_excel('diabetes.xlsx')

corr_matrix = df_completo.corr()

# Graficar la matriz de correlación con un heatmap
plt.figure(figsize=(8, 6))
plt.title("Matriz de Correlación ")

#no importa si la correlación es positiva o negativa, entonces se puede graficar abs
sns.heatmap(abs(corr_matrix.values), cmap='vlag', fmt=".2f", linewidths=0.5,xticklabels=corr_matrix.columns, yticklabels=corr_matrix.columns)
plt.tight_layout()



#%% Dividamos los datos en 80% para entrenamiento y 20% para prueba 

df = df_completo.sample(frac=0.8,random_state=42) #datos entrenamiento
df_prueba = df_completo.drop(df.index) #datos para prueba 

#%%


#x = df[['S1']]# un solo regresor 
#x_p=df_prueba[['S1']]
#x = x.values.reshape(-1,1)#necesario si solo hay un regresor
#x_p =x_p.values.reshape(-1,1)
#x = df[['S1','S3']]#dos regresores S3 baja correlación
#x_p = df_prueba[['S1','S3']]
#x = df[['S1','S4']]#dos regresores S4 tiene mejor correlación
#x_p = df_prueba[['S1','S4']]
#x =df[['S1','S4','S5']]#tres regresores 
#x_p =df_prueba[['S1','S4','S5']]
x = df[['S1','S3','S4','S5']] #cuatro regresores 
x_p = df_prueba[['S1','S3','S4','S5']]

y = df['S2'] #variable que se quiere describir o explicar 
y_p = df_prueba['S2']

modelo = LinearRegression().fit(x,y) # se genera y ajusta el modelo

#usamos los datos de prueba 

y_pred = modelo.predict(x_p) # se calculan los valores predichos. 

# Calcular métricas con los valores predichos 

mse = mean_squared_error(y_p, y_pred)
rmse = np.sqrt(mse) # RMSE
p=df.shape[1] #numero de predictores
RSE = np.sqrt(np.sum((y_p-y_pred)**2)/(len(y_p)-p-1))
r2 = r2_score(y_p, y_pred)
mae = mean_absolute_error(y_p, y_pred)

print(f'RMSE: {rmse:.2f}')
print(f'R²: {r2:.2f}')
print(f'RSE: {RSE:.2f}')
print(f'MAE: {mae:.2f}')

#%% k-folds cross-validation
x = df_completo[['S1']]# un solo regresor 
x = x.values.reshape(-1,1)#necesario si solo hay un regresor 
#x = df_completo[['S1','S3']]#dos regresores S3 baja correlación 
#x = df_completo[['S1','S4']]#dos regresores S4 tiene mejor correlación
x =df_completo[['S1','S4','S5']]#tres regresores 
#x = df_completo[['S1','S3','S4','S5']] #cuatro regresores 

y = df_completo['S2']

# Crear el modelo de regresión lineal
model = LinearRegression().fit(x,y) # se genera y ajusta el modelo
 
# Establecer el número de "folds" (particiones) para la validación cruzada
k = 10

# Crear el objeto KFold (k-particiones) para dividir los datos

kf = KFold(n_splits=k, shuffle=True, random_state=42) #cross-validation splitter

# Realizar k-fold cross-validation
scores1 = cross_val_score(model, x, y, cv=kf, scoring='neg_root_mean_squared_error')  # Usamos RMSE como métrica
scores = cross_val_score(model, x, y, cv=kf, scoring='r2')  # Usamos R2 como métrica
# Calcular el Error Cuadrático Medio promedio
rmse_promedio = -np.mean(scores1)  # Scikit-learn devuelve los valores negativos para RMSE, así que lo invertimos
R2_promedio = np.mean(scores) #promedio de R2 
print(f'Root Mean Squared Error promedio (RMSE): {rmse_promedio}')
print(f'Coef. de Determinación promedio (R^2): {R2_promedio}')

#%% residuors

residuos = y_p - modelo.predict(x_p) #residuos 

from scipy.stats import jarque_bera #prueba de normalidad para residuos (cientos de datos)

jb_test = jarque_bera(residuos)

print(f'Estadístico Jarque-Bera: {jb_test[0]}, p-valor: {jb_test[1]}')

# se puede revisar el qqplot para la normalidad de los residuos 
import statsmodels.api as sm 

sm.qqplot(residuos, line='45', fit=True)
plt.title("QQ-Plot de los Residuos del modelo")


from statsmodels.stats.stattools import durbin_watson #prueba para autocorrelación

DW = durbin_watson(residuos)

print(f'Estadístico Durbin-Watson: {DW:.4f}')

#%% Regresion con Interacciones 

from sklearn.preprocessing import PolynomialFeatures


# Paso 1: crear interacciones
#degree: orden de las interacciones
#interaction_only=True para que solo haga inteacciones
poly = PolynomialFeatures(degree=3, interaction_only=True)

x_poly = poly.fit_transform(x) #Construimos un nuevo X con las interacciones
 
# Paso 2: crear el modelo lineal

modelo_f = LinearRegression().fit(x_poly,y) #se ajusta el modelo para las interacciones

y_pre= np.sum(x_poly*modelo_f.coef_,axis=1)+modelo_f.intercept_ #modelo

# Calcular métricas con los valores predichos 

mse = mean_squared_error(y, y_pre)
rmse = np.sqrt(mse) # RMSE
p=df.shape[1] #numero de predictores
RSE = np.sqrt(np.sum((y-y_pre)**2)/(len(y)-p-1))
r2 = r2_score(y, y_pre)
mae = mean_absolute_error(y, y_pre)

print(f'RMSE: {rmse:.2f}')
print(f'R²: {r2:.2f}')
print(f'RSE: {RSE:.2f}')
print(f'MAE: {mae:.2f}')
