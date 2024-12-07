#####################################

#-----------Red neuronal-------------
# Desarrolla prototipos de red neuronal multicapa con distintos
# niveles de complejidad para observar cómo su arquitectura afecta 
# su capacidad de aprendizaje sobre el mismo conjunto de datos.

#####################################

# Librerías
import numpy as np
import os

import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs
from sklearn.neural_network import MLPClassifier

# Crea los directorios
# Directorio imágenes
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

# Gráfica de los datos
fig, ax=plt.subplots(nrows=1, ncols=1, figsize=(6, 4))
for i in np.unique(y):
    ax.scatter(
        x=X[y==i, 0],
        y=X[y==i, 1],
        c=plt.rcParams['axes.prop_cycle'].by_key()['color'][i],
        marker='o',
        edgecolor='black',
        label=f"Grupo {i}"
    )
ax.set_title("Datos simulados")
ax.legend()
plt.show()
fig.savefig(os.path.join(path_imagenes,'0_datos.png'), dpi=300, bbox_inches='tight')

# Arquitectura de la red
modelo_1=MLPClassifier(
            hidden_layer_sizes=(5),
            learning_rate_init=0.01,
            solver='lbfgs',
            max_iter=10000,
            activation='relu',
            random_state=123
)

modelo_2=MLPClassifier(
            hidden_layer_sizes=(10),
            learning_rate_init=0.01,
            solver='lbfgs',
            max_iter=10000,
            activation='relu',
            random_state=123
)

modelo_3=MLPClassifier(
            hidden_layer_sizes=(20, 20),
            learning_rate_init=0.01,
            solver='lbfgs',
            max_iter=10000,
            activation='relu',
            random_state=123
)

modelo_4=MLPClassifier(
            hidden_layer_sizes=(70, 70, 70),
            learning_rate_init=0.01,
            solver='lbfgs',
            max_iter=10000,
            activation='relu',
            random_state=123
)


modelo_1.fit(X=X, y=y)
modelo_2.fit(X=X, y=y)
modelo_3.fit(X=X, y=y)
modelo_4.fit(X=X, y=y)

# Gráfico de predicciones
fig, ax=plt.subplots(nrows=2, ncols=2, figsize=(12, 8))

# Convierte la cuadrícula en un arreglo 1D para iterar
ax=ax.flatten()

# Cuadrícula de puntos para predicciones
# 100 valores equidistantes para las dimensiones 0 y 1 de los datos en X
# Genera una malla 2D de puntos meshgrid (combinaciones de grid_x1 y grid_x2)
# Combina las coordinadas de xx e yy en un arreglo de forma (10.000, 2). Cada fila es un punto de cadrícula
grid_x1 = np.linspace(start=min(X[:, 0]), stop=max(X[:, 0]), num=100)
grid_x2 = np.linspace(start=min(X[:, 1]), stop=max(X[:, 1]), num=100)
xx, yy = np.meshgrid(grid_x1, grid_x2)
X_grid = np.column_stack([xx.flatten(), yy.flatten()])

# Itera sobre cada modelo para graficar en cada celda de cuadrícula
for i, modelo in enumerate([modelo_1, modelo_2, modelo_3, modelo_4]):
    # Usa el modelo para predecir las clases de los puntos en la cuadrícula X_grid
    predicciones = modelo.predict(X_grid)
    # Grafica las predicciones
    for j in np.unique(predicciones):
        ax[i].scatter(
            # Agrupa puntos de cuadrícula según la clase predicha
            x = X_grid[predicciones == j, 0],
            y = X_grid[predicciones == j, 1],
            c = plt.rcParams['axes.prop_cycle'].by_key()['color'][j],
            # Usa colores suaves para mostrar regiones clasificadas por el modelo, cada clase tiene un color
            alpha = 0.3,
            label = f"Grupo {j}"
        )

    for j in np.unique(y):
            ax[i].scatter(
                # Superpone los puntos originales para distinguirlos del fondo
                x = X[y == j, 0],
                y = X[y == j, 1],
                c = plt.rcParams['axes.prop_cycle'].by_key()['color'][j],
                # Marcas y color
                marker = 'o',
                edgecolor = 'black'
            )
    # Muestra el número de capas ocultas y neuronas del modelo actual
    ax[i].set_title(f"Capas ocultas: {modelo.hidden_layer_sizes}")
    # Desactiva los ejes ocultos para limpieza
    ax[i].axis('off')
    # Agrega la leyenda en el primer gráfico para identificar los grupos
ax[i].legend()
plt.show()
fig.savefig(os.path.join(path_imagenes,'1_clases_modelos.png'), dpi=300, bbox_inches='tight')