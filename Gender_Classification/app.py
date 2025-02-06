from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

#Load model and preprocessing objects
with open('gender_classification_model.pkl', 'rb') as f:
    data = pickle.load(f)
    model = data['model']
    scaler = data['scaler']
    label_encoders = data['label_encoders']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from the form
        userCode = request.form['userCode']
        company = request.form['company']
        name = request.form['name']
        age = request.form['age']

        # Create a DataFrame with the input data
        input_data = pd.DataFrame([[userCode, company, name, age]], 
                                columns=['userCode', 'company', 'name', 'age'])
        
        # Apply label encoding to categorical variables
        for column, encoder in label_encoders.items():
            input_data[column] = encoder.transform([input_data[column].iloc[0]])[0]
        
        # Convert all values to numeric
        input_data = input_data.astype(float)
        
        # Scale the features
        input_scaled = scaler.transform(input_data)
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        
        # Return the prediction to the template
        return render_template('index.html', prediction=prediction)
        
    except Exception as e:
        # Log the error (in a production environment)
        print(f"Error during prediction: {str(e)}")
        return render_template('index.html', error="An error occurred during prediction. Please try again.")

if __name__ == '__main__':
    app.run(debug=True)