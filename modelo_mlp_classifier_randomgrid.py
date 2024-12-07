#####################################

#-----------Red neuronal-------------
# Desarrolla un ejemplo de modelo mlp de clasificación
# con búsqueda de los mejores hiperparámetros
# mediante grilla aleatoria.

#####################################

# Librerías
import pandas as pd
import numpy as np
import os

import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import RandomizedSearchCV
import multiprocessing

# Crea los directorios
# Data
# Directorio datos
root='C:\\Users\\camed\\OneDrive\\Documentos\\Git\\neural'
carpeta_imagenes='imagenes'
path_imagenes=os.path.join(root, carpeta_imagenes)
os.makedirs(path_imagenes, exist_ok=True)


# Datos simulados
X, y = make_blobs(
        n_samples=500,
        n_features=2,
        centers=3,
        cluster_std=1.2,
        shuffle=True,
        random_state=0
)

print(y)

# Espacio de búsqueda de hiperparámetros
param_distributions={
    'hidden_layer_sizes' : [(10), (10, 10), (20, 20)],
    'alpha' : np.logspace(-3, 3, 7),
    'learning_rate_init' : [0.001, 0.01, 0.1]
}

# Búsqueda por validación cruzada
grid = RandomizedSearchCV(
    estimator = MLPClassifier(solver='lbfgs', max_iter=50000),
    param_distributions=param_distributions,
    n_iter=50, # Máximo número de iteraciones a probar
    scoring='accuracy',
    n_jobs=multiprocessing.cpu_count() - 1,
    cv=3,
    verbose=0,
    random_state=123,
    return_train_score=True
)

grid.fit(X=X, y=y)

# Resultados del grid
resultados=pd.DataFrame(grid.cv_results_)
print(resultados.columns)

resultados = resultados.filter(regex='(param.*)|mean_t|std_t')\
    .drop(columns='params')\
    .sort_values('mean_test_score', ascending=False)\
    .head(15)


print(resultados.loc[:,['param_learning_rate_init', 
                        'param_hidden_layer_sizes', 
                        'param_alpha', 
                        'mean_test_score', 
                        'std_test_score', 
                        'mean_train_score', 
                        'std_train_score']])

# La combinación de hiperparámetros ópimos es
modelo = grid.best_estimator_
print(modelo)

# Graficamos el resultado de la clasificación para los dos estimadores
# Crea una grilla con los valores del dominio a 100 divisiones
grid_x1 = np.linspace(start=min(X[:,0]), stop=max(X[:,0]), num=100)
grid_x2 = np.linspace(start=min(X[:,1]), stop=max(X[:,1]), num=100)
xx, yy = np.meshgrid(grid_x1, grid_x2)

X_grid = np.column_stack([xx.flatten(), yy.flatten()])
predicciones = modelo.predict(X_grid)

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5,5))

for i in np.unique(predicciones):
    ax.scatter(
        x = X_grid[predicciones == i, 0],
        y = X_grid[predicciones == i, 1],
        c = plt.rcParams['axes.prop_cycle'].by_key()['color'][i],
        alpha=0.3,
        label=f"Grupo {i}"
    )

for i in np.unique(y):
    ax.scatter(
        x = X[y == i, 0],
        y = X[y == i, 1],
        c = plt.rcParams['axes.prop_cycle'].by_key()['color'][i],
        marker='o',
        edgecolor='black'
    )

ax.set_title("Regiones de clasificación - Random")
ax.legend()
plt.show()
fig.savefig(os.path.join(path_imagenes,'2_clases_modelos_random_grid.png'), dpi=300, bbox_inches='tight')