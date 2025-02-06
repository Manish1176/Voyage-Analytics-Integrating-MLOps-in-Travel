from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np
from datetime import datetime

app = Flask(__name__)

# Load the model and preprocessing objects
with open('flight_model.pkl', 'rb') as f:
    model_data = pickle.load(f)
    
model = model_data['model']
scaler = model_data['scaler']
label_encoders = model_data['label_encoders']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        from_location = request.form['from']
        to_location = request.form['to']
        flight_type = request.form['flightType']
        time = float(request.form['time'])
        distance = float(request.form['distance'])
        agency = request.form['agency']
        date = request.form['date']

        # Create a DataFrame with the input data
        input_data = pd.DataFrame({
            'from': [from_location],
            'to': [to_location],
            'flightType': [flight_type],
            'time': [time],
            'distance': [distance],
            'agency': [agency],
            'date': [date]
        })

        # Convert date to datetime and then to int64
        input_data['date'] = pd.to_datetime(input_data['date']).astype(np.int64)

        # Apply label encoding to categorical columns
        categorical_columns = ['from', 'to', 'flightType', 'agency']
        for column in categorical_columns:
            try:
                input_data[column] = label_encoders[column].transform(input_data[column])
            except ValueError:
                return render_template('index.html', 
                    error=f"Invalid value for {column}. Please use one of the known values.")

        # Scale the features
        input_scaled = scaler.transform(input_data)

        # Make prediction
        prediction = model.predict(input_scaled)[0]

        return render_template('index.html', prediction=f"{prediction:.2f}")

    except Exception as e:
        return render_template('index.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True)
