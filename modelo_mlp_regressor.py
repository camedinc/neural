#####################################

#--------Red neuronal regresión------

#####################################

# Librerías
import pandas as pd
import numpy as np
import os

from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
import multiprocessing

# Crea los directorios
# Directorio imágenes
root='C:\\Users\\camed\\OneDrive\\Documentos\\Git\\neural'
carpeta_imagenes='imagenes'
path_imagenes=os.path.join(root, carpeta_imagenes)
os.makedirs(path_imagenes, exist_ok=True)

# Carga los datos
root='C:\\Users\\camed\\OneDrive\\Documentos\\Git\\neural'
carpeta = 'datos'
archivo = 'financial_regression.csv'
path_datos=os.path.join(root, carpeta, archivo)

df = pd.read_csv(path_datos)
print(df)

# Exploración
print("Dimensión:")
print(df.shape)
print(df.describe().T)

# Eliminar us_rates_%, CPI, GDP (pocos registros) y date (no es relevante)
df.drop(['date' , 'us_rates_%', 'CPI', 'GDP'], axis = 1, inplace = True)

# Elimina 185 filas de NAs
df.dropna(inplace = True)

# Calidad
print("\nPresencia de NAs:")
print(df.isna().sum())
print("\nPresencia de duplicados:")
print(df.duplicated().sum())

# Tipos
print(df.dtypes)

# Preprocesado requerido en la red neuronal: OHE y escalado
# df_num = df.select_dtypes(include = 'number', exclude = 'category')
# df_cat = df.select_dtypes(include = 'object', exclude = 'number')

#print("Variables numéricas:")
#print(df_num.head(5))

# Escalado
columnas = df.columns
def escala_num(df, columns):
    scaler=StandardScaler()
    df_escalado=scaler.fit_transform(df)
    df_escalado = pd.DataFrame(df_escalado, columns = columns).reset_index(drop=True)
    return df_escalado

df_escalado = escala_num(df, columnas)
print("Escalado:", df_escalado)

# OHE
#print("Variables categóricas:")
#print(df_cat.head(5))

# One Hot Encoding
# def ohe(df):
    # OHE categóricas
#    df_encoded = pd.get_dummies(df, dtype=int, drop_first=True)
#    df_encoded = pd.DataFrame(df_encoded).reset_index(drop=True)
#    return df_encoded

#df_encoded = ohe(df_cat)
#print("OHE:", df_encoded)

# Une el dataframe
# df_modelo = pd.concat([df_escalado, df_encoded], axis=1)

# Variables
X = df_escalado.drop('gold close', axis=1)
y = df_escalado['gold close']

# Entrenamiento y prueba
def divide_train_test(X, y, test_size):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = divide_train_test(X, y, test_size = 0.2)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

print(X_train)
print(X_test)
print(y_train)
print(y_test)

# Arquitectura de la red neuronal
# ---------------------------------------------------------

# Modelo MLP de regresión
modelo = MLPRegressor(solver = 'adam', max_iter = 10000)

# Espacio de búsqueda de hiperparámetros
param_distributions = {
    'hidden_layer_sizes' : [(5,), (10,), (20,), (30,), (30, 30), (50, 50, 50), (100, 50, 25), (100, 100, 100)],
    'alpha' : np.logspace(-4, 2, 100), # Más resolución en valores pequeños
    #'learning_rate_init' : [0.0001, 0.001, 0.01],
    'learning_rate' : ['adaptive', 'constant', 'invscaling'], # Parámetro correcto para aprendizaje adaptativo
    'activation': ['relu', 'tanh', 'logistic']  # Agregar funciones de activación
}

grid = RandomizedSearchCV(
        estimator = modelo,
        param_distributions = param_distributions,
        n_iter = 50,
        scoring = 'neg_mean_squared_error',
        n_jobs = multiprocessing.cpu_count() - 1,
        cv = 5,
        verbose = 0,
        random_state = 123,
        return_train_score = True
)

grid.fit(X = X_train, y = y_train)

# Resultados de la grilla
resultados = pd.DataFrame(grid.cv_results_)
resultados = resultados.filter(regex = 'param.*|mean_t|std_t')\
.drop(columns = 'params')\
.sort_values('mean_test_score', ascending = False)\
.head(10)
print("Resutados de la red neuronal:")
print(resultados)

# Otras métricas de evaluación para regresión
# ----------------------------------------------------------------

# MAE
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_test, grid.best_estimator_.predict(X_test))
print(f"\nMean Absolute Error (MAE): {mae}")

# R2 (mayor a 0.7 para finanzas)
from sklearn.metrics import r2_score
r2 = r2_score(y_test, grid.best_estimator_.predict(X_test))
print(f"\nR-Squared (R²): {r2}")

# RMSE
rmse = np.sqrt(-grid.best_score_)
print(f"\nRoot Mean Squared Error (RMSE): {rmse}")

# MAPE
def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

mape = mean_absolute_percentage_error(y_test, grid.best_estimator_.predict(X_test))
print(f"\nMean Absolute Percentage Error (MAPE): {mape}%")

# Distribución de errores (deben ser normales centrados en cero)
import matplotlib.pyplot as plt
errores = y_test - grid.best_estimator_.predict(X_test)
fig, ax=plt.subplots(nrows=1, ncols=1, figsize=(5,5))
plt.hist(errores, bins=30, edgecolor='k')
plt.title("Distribución de los Errores")
plt.show()
fig.savefig(os.path.join(path_imagenes,'4_regresion_distribución_errores.png'), dpi=300, bbox_inches='tight')

# Predicciones vs. valores reales
fig, ax=plt.subplots(nrows=1, ncols=1, figsize=(5,5))
plt.scatter(y_test, grid.best_estimator_.predict(X_test), alpha=0.6)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.title("Predicciones vs Valores Reales")
plt.xlabel("Valores Reales")
plt.ylabel("Predicciones")
plt.show()
fig.savefig(os.path.join(path_imagenes,'5_regresion_prediccion_real.png'), dpi=300, bbox_inches='tight')
