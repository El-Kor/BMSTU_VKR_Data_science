import flask
from flask import render_template, request
import pickle
import numpy as np
import tensorflow as tf
import os

app = flask.Flask(__name__, template_folder='templates')

base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, 'model.pkl')

# Загрузка модели
with open(model_path, 'rb') as f:
    model = pickle.load(f)

MIN_VALUES = np.array([1731.76, 2.43, 17.74, 14.25, 100.0, 0.60, 64.05, 1036.85, 33.80, 0.0, 0.0, 0.0])
MAX_VALUES = np.array([2207.77, 1911.53, 198.95, 33.0, 413.27, 1399.54, 82.68, 3848.43, 414.59, 90.0, 14.44, 103.98])

@app.route('/', methods=['GET', 'POST'])
@app.route('/index', methods=['GET', 'POST'])
def main():
    if request.method == 'GET':
        return render_template('main.html', result=None)

    if request.method == 'POST':
        
        try:
            input_list = [
                request.form['density'],
                request.form['elasticity_modulus'],
                request.form['hardener_qty'],
                request.form['epoxy_groups'],
                request.form['flash_point'],
                request.form['surface_density'],
                request.form['tensile_modulus'],
                request.form['tensile_strength'],
                request.form['resin_consumption'],
                request.form['angle'],
                request.form['step'],
                request.form['patch_density']
            ]
            
            
            features = np.array([float(x) for x in input_list])
            features_scaled = (features - MIN_VALUES) / (MAX_VALUES - MIN_VALUES)
            features_scaled = features_scaled.reshape(1, -1)

            # Прогноз
            prediction = model.predict(features_scaled)
            res = round(float(prediction[0][0]), 3)

            return render_template('main.html', result=res)
        
        except Exception as e:
            return render_template('main.html', result=f"Ошибка: {e}")

if __name__ == '__main__':
    app.run(debug=True)