from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load the trained model
model = joblib.load('logistic_regression_model.pkl')

# Load the dataset
df = pd.read_csv('Crop_recommendation.csv')

# Function to preprocess user input
def preprocess_input(input_data):
    input_data = [float(val) for val in input_data]
    return np.array(input_data).reshape(1, -1)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        input_data = request.form.to_dict(flat=False)['input_data[]']
        input_scaled = preprocess_input(input_data)
        # Predict crop label
        prediction = model.predict(input_scaled)[0]
        return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
