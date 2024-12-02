from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

# Cargar el modelo KMeans desde el archivo proporcionado
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

app = Flask(__name__)

# Ruta para la página principal
@app.route('/')
def index():
    return render_template('index.html')

# Ruta para predecir clústeres
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Obtener datos JSON de la solicitud
        data = request.get_json()
        points = np.array(data['points'])

        # Predecir clústeres usando el modelo KMeans cargado
        clusters = model.predict(points)

        # Retornar respuesta
        return jsonify({'clusters': clusters.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)
