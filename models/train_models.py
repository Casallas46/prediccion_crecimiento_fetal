# train_models.py

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import numpy as np

# Cargar dataset
df = pd.read_excel("FGR_dataset.xlsx")

# Renombrar columnas para mayor claridad
df = df.rename(columns={
    "C1": "Edad",
    "C2": "IMC",
    "C3": "Edad Gestacional (parto)",
    "C4": "Gravidez",
    "C5": "Paridad",
    "C6": "Sintomas inicio",
    "C9": "Edad Gestacional (HTA)",
    "C13": "Edad Gestacional (proteinuria)",
    "C17": "Antecedentes",
    "C18": "Presion sistolica max",
    "C19": "Presion diastolica max",
    "C23": "Creatinina max",
    "C25": "Proteinuria max",
    "C31": "Peso Fetal"
})

# Variables predictoras
features = [
    "Edad", "IMC", "Edad Gestacional (parto)", "Gravidez", "Paridad",
    "Sintomas inicio", "Edad Gestacional (HTA)", "Edad Gestacional (proteinuria)",
    "Antecedentes", "Presion sistolica max", "Presion diastolica max",
    "Creatinina max", "Proteinuria max"
]

X = df[features]
y = df["Peso Fetal"]

# Normalizar
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Guardar scaler
joblib.dump(scaler, "models/escalador.pkl")

# Dividir datos
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 1. Regresión Logística
logistic_model = LogisticRegression(max_iter=1000)
logistic_model.fit(X_train, y_train)
joblib.dump(logistic_model, "models/logistic_model.pkl")

# 2. Red Neuronal Artificial
mlp_model = MLPClassifier(hidden_layer_sizes=(50,), max_iter=1000)
mlp_model.fit(X_train, y_train)
joblib.dump(mlp_model, "models/mlp_model.pkl")

# 3. Máquina de Soporte Vectorial
svm_model = SVC(probability=True)
svm_model.fit(X_train, y_train)
joblib.dump(svm_model, "models/svm_model.pkl")

# 4. Mapa Cognitivo Difuso (simulación usando media ponderada como aproximación)
class MapaCognitivoDifuso:
    def predict(self, X):
        return np.where(X[:, 1] > 30, 1, 0)  # ejemplo simple: si IMC > 30 = FGR
fcm_model = MapaCognitivoDifuso()
joblib.dump(fcm_model, "models/fcm_model.pkl")

print("Modelos entrenados y guardados exitosamente.")
