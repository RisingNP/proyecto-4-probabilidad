# -*- coding: utf-8 -*-
"""
Created on Mon Apr 28 16:31:36 2025

@author: fjose
"""

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
import numpy as np
import matplotlib.pyplot as plt
import seaborn; seaborn.set()
from sklearn.model_selection import validation_curve, learning_curve
import pandas as pd 
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score
#from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import Lasso, Ridge, RidgeCV, LassoCV
 #%%
def PolynomialRegression(degree=2, **kwargs):
    
    return make_pipeline(PolynomialFeatures(degree), LinearRegression(**kwargs))

def make_data(N, err=1.0, rseed=1):
    # randomly sample the data
    rng = np.random.RandomState(rseed) # se crea unn objeto random
    X = rng.rand(N) ** 2# se usa el atributo rand del objeto para generar datos aleatorios, al cuadrado
    y = 10 - 1. / (X + 0.1)
    if err > 0:
        y += err * rng.randn(N) # se le suma a y un error, para que la relación no sea perfecta
    return X, y

X, y = make_data(100)
X = X.reshape(-1,1)
#%%
X_test = np.linspace(-0.1, 1.1, 500) # serán los datos para testear 
X_test = X_test.reshape(-1,1)
#%%
plt.scatter(X, y, color='black')
axis = plt.axis()
for degree in [1, 3, 5, 15]:
    y_test = PolynomialRegression(degree).fit(X, y).predict(X_test)
    plt.plot(X_test, y_test, label=f'degree={degree}')
plt.xlim(-0.1, 1.0)
plt.ylim(-2, 30)
plt.legend(loc='best');

#%% veamos la curva de validación 

degree = np.arange(0, 21) # veamos para diferentes valores de grados

train_score, val_score = validation_curve(PolynomialRegression(), X, y,
                                          param_name='polynomialfeatures__degree',
                                          param_range=degree, cv=7)
plt.plot(degree, np.median(train_score, axis=1), color='blue', 
         label='training score')
plt.plot(degree, np.median(val_score, axis=1), color='red', 
         label='validation score')
plt.legend(loc='best')
plt.ylim(0, 1)
plt.xlabel('degree')
plt.ylabel('score')

better_degree = degree[np.argmax(np.median(val_score,axis=1))]

print(f'El mejor orden del polinomio es: {better_degree}')


#%% curvas de aprendizaje. 

fig, ax = plt.subplots(1, 3, figsize=(16, 6))
 
fig.subplots_adjust(left=0.0625, right=0.95, wspace=0.1)
for i, degree in enumerate([2, 3, 5]):
    N, train_lc, val_lc = learning_curve(PolynomialRegression(degree),
    X, y, cv=7,    train_sizes=np.linspace(0.05, 1, 50))
    ax[i].plot(N, np.mean(train_lc,axis=1), color='blue', label='training score')
    ax[i].plot(N, np.mean(val_lc, axis=1), color='red', label='validation score')
    ax[i].hlines(np.mean([train_lc[-1], val_lc[-1]]), N[0], N[-1], color='gray',
    linestyle='dashed')
    ax[i].set_ylim(0, 1)
    ax[i].set_xlim(N[0], N[-1])
    ax[i].set_xlabel('training size')
    ax[i].set_ylabel('score')
    ax[i].set_title(f'degree = {degree}'.format(degree), size=14)
    ax[i].legend(loc='best')


#%% Regularización 

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

x = df[['S1','S3','S4','S5']] #cuatro regresores 
x_p = df_prueba[['S1','S3','S4','S5']]

y = df['S2'] #variable que se quiere describir o explicar 
y_p = df_prueba['S2']

modelo = LinearRegression().fit(x,y) # se genera y ajusta el modelo
modelo_Ridge=Ridge(alpha=0.5).fit(x,y) 
modelo_Lasso=Lasso(alpha=0.5).fit(x,y)
#usamos los datos de prueba 

y_pred = modelo.predict(x_p) # se calculan los valores predichos. 
y_pred_R=modelo_Ridge.predict(x_p)
y_pred_L=modelo_Lasso.predict(x_p)
# Calcular métricas con los valores predichos 

rmse = np.sqrt(mean_squared_error(y_p, y_pred))
r2 = r2_score(y_p, y_pred)

rmse_R = np.sqrt(mean_squared_error(y_p, y_pred_R))
r2_R = r2_score(y_p, y_pred_R)

rmse_L = np.sqrt(mean_squared_error(y_p, y_pred_L))
r2_L = r2_score(y_p, y_pred_L)


print(f'RMSE: {rmse:.2f}')
print(f'R²: {r2:.2f}')

print(f'RMSE_R: {rmse_R:.2f}')
print(f'R²_R: {r2_R:.2f}')

print(f'RMSE_L: {rmse_L:.2f}')
print(f'R²_L: {r2_L:.2f}')

#%% para elergir el valor de alfa, se puede usar una validación cruzada para
#diferentes alfa 
#alfas a ensayar
alfa = np.logspace(-3,3,num=7)

modelo_RidgeCV=RidgeCV(alphas=alfa,cv=10).fit(x,y)
modelo_LassoCV=LassoCV(alphas=alfa,cv=10).fit(x,y)

print("Mejor alpha_Ridge:", modelo_RidgeCV.alpha_)
print("Mejor alpha_Lasso:", modelo_LassoCV.alpha_)

