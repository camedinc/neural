#####################################

#-----------Red neuronal-------------
# Desarrolla un ejemplo de modelo mlp de clasificación
# con búsqueda de los mejores hiperparámetros
# mediante búsqueda bayesiana

#####################################

# Librerías
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

import optuna
from sklearn.datasets import make_blobs
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
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

# Crea una función objetivo para optimizar
def objetivo(trial):
    # Espacio de búsqueda de hiperparámetros
    hidden_layer_sizes = trial.suggest_categorical("hidden_layer_sizes", [(10), (10, 10), (20, 20)])
    alpha = trial.suggest_loguniform("alpha", 1e-4, 1e1)
    learning_rate_init = trial.suggest_loguniform("learning_rate_init", 1e-4, 1e-1)

    # Modelo que se entrena
    modelo = MLPClassifier(
        hidden_layer_sizes = hidden_layer_sizes,
        alpha = alpha,
        learning_rate_init = learning_rate_init,
        solver = 'lbfgs',
        max_iter=1000,
        random_state=2024
    )

    # Validación cruzada
    score = cross_val_score(modelo, X, y, cv=3, scoring='accuracy').mean()
    return score

# Crea el estudio de optimización
study = optuna.create_study(direction='maximize')
study.optimize(objetivo, n_trials=50)

# Muestra los mejores resultados del estudio de optimización
print("Mejores hiperparámetros")
print(study.best_params)
print("Mejor precisión:", study.best_value)

# Entrena el modelo con los mejores hiperparámetros entontrados por optuna
best_params = study.best_params
final_model = MLPClassifier(
    hidden_layer_sizes=best_params['hidden_layer_sizes'],
    alpha = best_params['alpha'],
    learning_rate_init = best_params['learning_rate_init'],
    solver = 'lbfgs',
    max_iter = 1000,
    random_state = 2024
)

final_model.fit(X=X, y=y)

# Resultados de proceso de optimización
# Se crea un dataframe con los resultados de los ensayos
trials_df = pd.DataFrame([
    {
    'hidden_layer_sizes' : trial.params['hidden_layer_sizes'],
    'alpha' : trial.params['alpha'],
    'learning_rate_init' : trial.params['learning_rate_init'],
    'mean_accuracy' : trial.value, # Puntaje medio de validación cruzada
    'trial_number' : trial.number
    }
    for trial in study.trials
])

# ordena por rendimiento del modelo (descendente)
trials_df = trials_df.sort_values(by='mean_accuracy', ascending=False)
print(trials_df.head(15))

# Graficamos el resultado de la clasificación para los dos estimadores
# Crea una grilla con los valores del dominio a 100 divisiones
grid_x1 = np.linspace(start=min(X[:,0]), stop=max(X[:,0]), num=100)
grid_x2 = np.linspace(start=min(X[:,1]), stop=max(X[:,1]), num=100)
xx, yy = np.meshgrid(grid_x1, grid_x2)

X_grid = np.column_stack([xx.flatten(), yy.flatten()])
predicciones = final_model.predict(X_grid)

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

ax.set_title("Regiones de clasificación - Bayes")
ax.legend()
plt.show()
fig.savefig(os.path.join(path_imagenes,'3_clases_modelos_bayes_grid.png'), dpi=300, bbox_inches='tight')
