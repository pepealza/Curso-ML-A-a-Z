#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 18:38:15 2019

@author: juangabriel
"""

# SVR

# Cómo importar las librerías
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importar el data set
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values


# Dividir el data set en conjunto de entrenamiento y conjunto de testing
"""
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
"""

# Escalado de variables utilizamos reshepe -1 y 1 por sugerencia py
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y.reshape(-1,1))

# Ajustar la regresión con el dataset
#kernel puede (es por defecto "rbf"), "linear", "poly", "sigmoid", "precomputed"
from sklearn.svm import SVR
regression = SVR(kernel = "rbf")
regression.fit(X, y)

# Predicción de nuestros modelos con SVR
y_pred = sc_y.inverse_transform(regression.predict(sc_X.transform(np.array([[6.5]]))))

# Visualización de los resultados del SVR
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, y, color = "red")
plt.plot(X_grid, regression.predict(X_grid), color = "blue")
plt.title("Modelo de Regresión (SVR)")
plt.xlabel("Posición del empleado")
plt.ylabel("Sueldo (en $)")
plt.show()

# Visualización de los resultados del Modelo luego de hacer la inversa de la transformacion
#Regresar los datos a sus valores originales

X_orig = sc_X.inverse_transform(X)
y_orig = sc_y.inverse_transform(y)

#Graficar
plt.scatter(X_orig,y_orig,color="red")
plt.plot(X_orig,sc_y.inverse_transform(regression.predict(X)),color="blue")
plt.title("Modelo de regresión SVR")
plt.xlabel("Posición del empleado")
plt.ylabel("Sueldo")
plt.show()

#otra opcion para desescalar
# Visualización de los resultados del Modelo

X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)
vector = sc_y.inverse_transform(regression.predict(X_grid))
X_grid = sc_X.inverse_transform(X_grid)
X = sc_X.inverse_transform(X)
y = sc_y.inverse_transform(y)
plt.scatter(X, y, color = "red")
plt.plot(X_grid, vector, color = "blue")
plt.title("Modelo de Regresión")
plt.xlabel("Posición del empleado")
plt.ylabel("Sueldo (en $)")
plt.show()
