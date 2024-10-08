import pandas as pd
import gc
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score

# Imprimir todas las columnas y filas
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)

# Cargar parte de los datos de entrenamiento
data = pd.read_csv("ctr_21.csv")

# Cargar los datos de validacion 
indices = list(range(len(data)))
np.random.shuffle(indices)

# creo el conjunto de entrenamiento
train_indices = indices[:int(0.1*len(data))]
train_data = data.iloc[train_indices]
# entrenar un árbol con los datos de entrenamiento
y_train = train_data["Label"]
X_train = train_data.drop(columns=["Label"])
X_train = X_train.select_dtypes(include='number')

# creo el conjunto de validacion
val_indices = indices[int(0.1*len(data)):int(0.2*len(data))]
val_data = data.iloc[val_indices]
y_val = val_data["Label"]
x_val = val_data.drop(columns=["Label"])
x_val = x_val.select_dtypes(include='number')

# Cargar los datos de prueba
eval_data = pd.read_csv("ctr_test.csv")

del train_data, val_data, train_indices, val_indices, data, indices
gc.collect()


# Definir el modelo
rf = RandomForestClassifier(random_state=42)
# grilla de parámetros para la búsqueda
param_grid = {
	'n_estimators': [160],  		# Número de árboles en el bosque
	'max_depth': [6],  				# Profundidad máxima del árbol
	'min_samples_split': [6],   	# Número mínimo de muestras requeridas para dividir un nodo
	'min_samples_leaf': [1]  	  	# Número mínimo de muestras requeridas en un nodo hoja
}

# hacemos la búsqueda exhausitiva de los parámetros con sckill-learn
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=1) #parametros: cv=#divisiones; n_jobs=-1 para usar mas procesadores; verbose=1 para ver el progreso de busqueda 
grid_search.fit(X_train, y_train)

# Obtener el mejor modelo
best_rf = grid_search.best_estimator_
print(grid_search.best_params_) 	# vemos cuales son los param que optimizan el modelo
#print(grid_search.best_score_) 	# vemos el score del mejor modelo (creeo que hay un error aqui, desconfiar de este valor)

# Calcular el AUC-ROC en el conjunto de validacion
y_val_preds = best_rf.predict_proba(x_val)[:, 1] #predicciones de la clase positiva , solo queremos la columna 2
val_auc_roc = roc_auc_score(y_val, y_val_preds)  
print("AUC-ROC en el conjunto de validacion:", val_auc_roc)

# Predecir en el conjunto de evaluación
eval_data = eval_data.select_dtypes(include='number')
y_preds = best_rf.predict_proba(eval_data.drop(columns=["id"]))[:, 1]

# Crear el archivo de envío
submission_df = pd.DataFrame({"id": eval_data["id"], "Label": y_preds})
submission_df["id"] = submission_df["id"].astype(int)
submission_df.to_csv("basic_model.csv", sep=",", index=False)

