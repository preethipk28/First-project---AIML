from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the trained model
model = joblib.load('model.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        pclass = int(request.form['pclass'])
        sex = 0 if request.form['sex'] == 'male' else 1
        age = float(request.form['age'])
        sibsp = int(request.form['sibsp'])
        parch = int(request.form['parch'])
        fare = float(request.form['fare'])

        input_data = np.array([[pclass, sex, age, sibsp, parch, fare]])
        prediction = model.predict(input_data)[0]
        result = "Survived" if prediction == 1 else "Did not survive"
    except Exception as e:
        result = f"Error: {str(e)}"
    
    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)