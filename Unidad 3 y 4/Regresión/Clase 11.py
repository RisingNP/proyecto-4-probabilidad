# -*- coding: utf-8 -*-
"""
Created on Sat Apr 26 16:11:35 2025

@author: fjose
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


#%% cargamos el achivo de datos

df = pd.read_excel("visceral_fat.xlsx")

#%% Miremos qué está correlacionado

#saquemos las variables categóricas 
df_numerico=df.drop(columns=['current gestational age', 'gestational age at birth'])

corr_matrix = df_numerico.corr()

# Graficar la matriz de correlación con un heatmap
plt.figure(figsize=(8, 6))

sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)

# Personalización
plt.title("Matriz de Correlación ")

#%%  
df = df.dropna() #eliminamos primero los nan
#x = df['pregnancies (number)']
#x = x.values.reshape(-1,1)
#y = df['age (years)']

#segun la matriz de correlación, el IMC está relacionado con la presión sistólica
#intentemos generar un modelo que prediga la presión sistolica a partir del IMC
#df = df.dropna() #eliminamos primero los nan

#x = df['bmi pregestational (kg/m)'] 
x = df['mean diastolic bp (mmhg)'] 
# el algoritmo de sklearn espera siempre más de 1 predictor 
#por este motivo x debe ser 2D 
x = x.values.reshape(-1,1) #esto hace que sea un arreglo de (n,1)

y = df['mean systolic bp (mmhg)']


# Crear el modelo con sklearn
modelo = LinearRegression().fit(x,y)

#Para medir la precisión del modelo, se usan los datos predichos y los medidos
y_pred = modelo.predict(x) 

# Calcular métricas
mse = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y, y_pred)
r2 = r2_score(y, y_pred)
p=1 #numero de predictores
RSE = np.sqrt(np.sum((y-y_pred)**2)/(len(y)-p-1))

print(f'MSE: {mse:.2f}')
print(f'RMSE: {rmse:.2f}')
print(f'MAE: {mae:.2f}')
print(f'R²: {r2:.2f}')
print(f'RSE: {RSE:.2f}')



"""
El R2 es muy bajo, 0.5 
Sin embargo, la varible X es significativa, esto puede deberser a que 
si describe a Y pero no por completo, faltan otras varible 

Los residuos no son normales, eso no es un buen resultado 
aunque son no autocorrelacionados lo qeu sí es positivo. 

"""


