import pandas as pd
import gc
import numpy as np
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

# Imprimir todas las columnas y filas
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)

# Cargar datos de entrenamiento
print("Cargando datos de entrenamiento...")

## en esta version separamos datos de entrenamiento y validacion evitando hacer overfitting por el mal manejo de las fechas de las observaciones

# Cargar parte de los datos de entrenamiento y entrenar arbol con ellos
data = pd.concat([pd.read_csv(f"ctr_{i}.csv") for i in range(15, 16)])
indices = list(range(len(data)))
np.random.shuffle(indices)
print()


# creo el conjunto de entrenamiento
train_indices = indices[:int(0.05*len(data))]
train_data = data.iloc[train_indices]
print("reemplazando NAs con la media...")
# Separar las columnas numéricas y categóricas

# Rellenar los NAs en las columnas numéricas con la media
numeric_cols = train_data.select_dtypes(include=['number']).columns
train_data[numeric_cols] = train_data[numeric_cols].fillna(train_data[numeric_cols].mean())

# Rellenar los NAs en las columnas categóricas con la moda
categorical_cols = train_data.select_dtypes(exclude=['number', 'bool']).columns
train_data[categorical_cols] = train_data[categorical_cols].apply(lambda x: x.fillna(x.mode()[0]))

# Si tienes columnas booleanas, puedes rellenarlas con el valor más frecuente (moda)
boolean_cols = train_data.select_dtypes(include=['bool']).columns
train_data[boolean_cols] = train_data[boolean_cols].apply(lambda x: x.fillna(x.mode()[0]))
print("train data NAs listo")

# entrenar un árbol con los datos de entrenamiento
y_train = train_data["Label"]
X_train = train_data.drop(columns=["Label"])
X_train = X_train.select_dtypes(include='number')
print("Train data listo")
print()


# Crear el conjunto de validación
print("Creando conjunto de validación...")
val_indices = indices[int(0.6*len(data)):int(0.65*len(data))]
val_data = data.iloc[val_indices]
y_val = val_data["Label"]
X_val = val_data.drop(columns=["Label"])
X_val = X_val.select_dtypes(include='number')
print("Val data listo")
print()


# Cargar los datos de prueba
eval_data = pd.read_csv("ctr_test.csv")
del train_data, val_data, train_indices, val_indices, data, indices
gc.collect() 


# Definir el modelo XGBoost
print("Definiendo el modelo XGBoost...")
xgb_clf = xgb.XGBClassifier(objective='binary:logistic', eval_metric='auc', random_state=42)

# Diccionario de parámetros para la búsqueda aleatoria
param_dist_xgb = {
    'n_estimators': np.random.randint(110, 160 , size=15),  # Rango para el número de árboles (100 a 300)
    'max_depth': np.random.randint(0, 50, size=10),         # Rango para la profundidad máxima (4 a 8)
    'learning_rate': np.random.uniform(0.008, 0.05, size=50),  # Rango para la tasa de aprendizaje (0.01 a 0.05)
    'subsample': np.random.uniform(0.55, 0.85, size=20),     # Rango para el muestreo (0.6 a 0.8)
    'colsample_bytree': np.random.uniform(0.55, 0.8, size=20),  # Rango para el muestreo de columnas (0.6 a 0.8)
    'gamma': np.random.uniform(0, 0.1, size=20),           # Rango para la regularización (0 a 0.1)
    'min_child_weight': np.random.randint(1, 10, size=10),  # Rango para el peso mínimo de las muestras (1 a 10)
    'reg_lambda': np.random.uniform(0.5, 1.0, size=20)     # Rango para la regularización L2 (0.5 a 1.0)
}

gc.collect() 

# Búsqueda exhaustiva de los mejores parámetros
print("Iniciando Random Search para optimización de hiperparámetros...")
random_search_xgb = RandomizedSearchCV(estimator=xgb_clf, param_distributions=param_dist_xgb, n_iter=5, cv=5, n_jobs=-1, verbose=1, random_state=42)  # Cambié cv=5 a cv=3
random_search_xgb.fit(X_train, y_train)

# Obtener el mejor modelo
print("Random Search completado. Seleccionando mejor modelo...")
best_xgb = random_search_xgb.best_estimator_
print(f"Mejores parámetros encontrados: {random_search_xgb.best_params_}")


# Calcular el AUC-ROC en el conjunto de validación
print(f"score de la ibreia es: {random_search_xgb.best_score_}") ## OJO con esto (dudoso)
print("Calculando AUC-ROC en el conjunto de validación...")
y_val_preds_xgb = best_xgb.predict_proba(X_val)[:, 1]
val_auc_roc_xgb = roc_auc_score(y_val, y_val_preds_xgb)
print(f"AUC-ROC en el conjunto de validación con XGBoost: {val_auc_roc_xgb}")

# Predecir en el conjunto de evaluación
print("Realizando predicciones sobre el conjunto de evaluación...")
eval_data = eval_data.select_dtypes(include='number')
y_preds_xgb = best_xgb.predict_proba(eval_data.drop(columns=["id"]))[:, 1]

# Crear el archivo de envío
print("Creando archivo de envío...")
submission_df_xgb = pd.DataFrame({"id": eval_data["id"], "Label": y_preds_xgb})
submission_df_xgb["id"] = submission_df_xgb["id"].astype(int)
submission_df_xgb.to_csv("xgboost_model.csv", sep=",", index=False)
print("¡Proceso completado! Archivo de envío guardado como 'xgboost_model.csv'.")

# relleno los nas con media, 
# cambio los parametros de grilla --> mas valores
# cambio el cv de 5 a 3
# cambio el n_iter de 25 a 35
# agrego min_child_weight
# mas datos en los sets train y val

## n_iter = 1
## train size 30%
