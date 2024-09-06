import pandas as pd
import gc
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score

# Imprimir todas las columnas y filas
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)

# Cargar parte de los datos de entrenamiento
print("Cargando datos de entrenamiento...")
train_data = pd.read_csv("ctr_21.csv")

# Cargar los datos de validación 
print("Cargando datos de validación...")
val_data = pd.read_csv("ctr_19.csv")
val_data = val_data.sample(frac=1/10)
y_val = val_data["Label"]
x_val = val_data.drop(columns=["Label"])
x_val = x_val.select_dtypes(include='number')

# Cargar los datos de prueba
print("Cargando datos de evaluación...")
eval_data = pd.read_csv("ctr_test.csv")

# Entrenar un árbol con los datos de entrenamiento
print("Preparando datos de entrenamiento...")
train_data = train_data.sample(frac=1/10)
y_train = train_data["Label"]
X_train = train_data.drop(columns=["Label"])
X_train = X_train.select_dtypes(include='number')
del train_data
gc.collect()

# Definir el modelo XGBoost
print("Definiendo el modelo XGBoost...")
xgb_clf = xgb.XGBClassifier(objective='binary:logistic', eval_metric='auc', random_state=42)

# Grilla de parámetros para la búsqueda (mejorada)
param_grid_xgb = {
    'n_estimators': [100, 200, 300],  # Aumenta el número de árboles
    'max_depth': [4, 6, 8],             # Aumenta la profundidad máxima
    'learning_rate': [0.01, 0.05],     # Varias tasas de aprendizaje
    'subsample': [0.6, 0.8],           # Opciones de muestreo
    'colsample_bytree': [0.6, 0.8],    # Opciones para el muestreo de columnas
    'gamma': [0, 0.1],                  # Variaciones en la regularización
    'reg_lambda': [0.5, 1.0]            # Variaciones en la regularización L2
}

# Búsqueda exhaustiva de los mejores parámetros
print("Iniciando Grid Search para optimización de hiperparámetros...")
grid_search_xgb = GridSearchCV(estimator=xgb_clf, param_grid=param_grid_xgb, cv=3, n_jobs=-1, verbose=2)  # Cambié cv=5 a cv=3
grid_search_xgb.fit(X_train, y_train)

# Obtener el mejor modelo
print("Grid Search completado. Seleccionando mejor modelo...")
best_xgb = grid_search_xgb.best_estimator_
print(f"Mejores parámetros encontrados: {grid_search_xgb.best_params_}")
print(f"Mejor puntaje durante la búsqueda: {grid_search_xgb.best_score_}")

# Calcular el AUC-ROC en el conjunto de validación
print("Calculando AUC-ROC en el conjunto de validación...")
y_val_preds_xgb = best_xgb.predict_proba(x_val)[:, 1]
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
