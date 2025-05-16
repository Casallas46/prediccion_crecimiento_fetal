from flask import Flask, render_template, request
import pandas as pd
import joblib
import os
import numpy as np

# ✅ Clase usada para el modelo FCM simulado
class MapaCognitivoDifuso:
    def predict(self, X):
        X = np.array(X)
        return np.where(X[:, 1] > 30, 1, 0)  # Ejemplo simple: si IMC > 30 = FGR

# ✅ Inicializar Flask app
app = Flask(__name__)

# ✅ Definir ruta base relativa al archivo app.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# ✅ Función para cargar modelo con validación
def cargar_modelo(path, custom=False):
    full_path = os.path.join(MODELS_DIR, path)
    if not os.path.exists(full_path):
        print(f"[❌] Modelo no encontrado: {full_path}")
        return None if not custom else MapaCognitivoDifuso()
    print(f"[✅] Modelo cargado: {full_path}")
    return joblib.load(full_path) if not custom else MapaCognitivoDifuso()

# ✅ Cargar modelos de forma segura
modelos = {
    "Regresión logística": cargar_modelo("logistic_model.pkl"),
    "Red neuronal artificial": cargar_modelo("mlp_model.pkl"),
    "Máquina de vector soporte": cargar_modelo("svm_model.pkl"),
    "Mapa cognitivo difuso": cargar_modelo("fcm_model.pkl", custom=True)
}

# ✅ Definir columnas esperadas
columnas = [
    "Age", "BMI", "Gestational age of delivery", "Gravidity", "Parity",
    "Initial onset symptoms", "Gestational age of hypertension onset",
    "Gestational age of proteinuria onset", "Past history",
    "Maximum systolic blood pressure", "Maximum diastolic blood pressure",
    "Maximum values of creatinine", "Maximum proteinuria value"
]

# ✅ Rutas
@app.route('/')
def index():
    return render_template("index.html")

@app.route('/manual', methods=['GET', 'POST'])
def manual_prediction():
    if request.method == 'POST':
        try:
            datos = [
                float(request.form['Age']),
                float(request.form['BMI']),
                float(request.form['Gestational age of delivery']),
                float(request.form['Gravidity']),
                float(request.form['Parity']),
                int(request.form['Initial onset symptoms']),
                float(request.form['Gestational age of hypertension onset']),
                float(request.form['Gestational age of proteinuria onset']),
                int(request.form['Past history']),
                float(request.form['Maximum systolic blood pressure']),
                float(request.form['Maximum diastolic blood pressure']),
                float(request.form['Maximum values of creatinine']),
                float(request.form['Maximum proteinuria value'])
            ]
            modelo_nombre = request.form['modelo']
            modelo = modelos.get(modelo_nombre)
            if modelo is None:
                return f"Modelo no cargado o inexistente: {modelo_nombre}"
            prediccion = modelo.predict([datos])[0]
            resultado = "Normal" if prediccion == 0 else "FGR"
            return render_template("result.html", resultado=resultado, modelo=modelo_nombre)
        except Exception as e:
            return f"Error en los datos: {str(e)}"
    return render_template("manual_prediction.html")

@app.route('/batch', methods=['GET', 'POST'])
def batch_prediction():
    if request.method == 'POST':
        file = request.files['dataset']
        if not file:
            return "No se subió ningún archivo"
        try:
            df = pd.read_excel(file)
            modelo_nombre = request.form['modelo']
            modelo = modelos.get(modelo_nombre)
            if modelo is None:
                return f"Modelo no cargado o inexistente: {modelo_nombre}"
            X = df[columnas]
            y = df["Fetal weight"]
            y_pred = modelo.predict(X)
            exactitud = (y == y_pred).mean()
            return render_template("result.html", resultado=f"Exactitud del modelo: {exactitud:.2%}", modelo=modelo_nombre)
        except Exception as e:
            return f"Error procesando el archivo: {str(e)}"
    return render_template("batch_prediction.html")

# ✅ Ejecutar la app en modo debug (para desarrollo)
if __name__ == '__main__':
    app.run(debug=True)
