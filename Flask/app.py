import numpy as np
import pickle
import joblib
import matplotlib
import matplotlib.pyplot as plt
import time
import pandas as pd
import os
from flask import Flask, request, jsonify, render_template
from markupsafe import escape  # Use this import if the previous one fails
import sklearn
print(sklearn.__version__)

app = Flask(__name__)

# Print the current working directory
print("Current Working Directory:", os.getcwd())

# Use binary mode for opening the model file
with open("c:/Users/LENOVO/Downloads/TrafficTelligence - Advanced Traffic Volume Estimation with Machine Learning/Flask/HRF_Model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

# Use binary mode for opening the scaler file
with open("c:/Users/LENOVO/Downloads/TrafficTelligence - Advanced Traffic Volume Estimation with Machine Learning/Flask/scaler.pkl", "rb") as scale_file:
    scaler = pickle.load(scale_file)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=["POST"])
def predict():
    input_feature = [float(x) for x in request.form.values()]
    features_values = [np.array(input_feature)]
    names = ['holiday', 'temp', 'rain', 'snow', 'weather', 'Day', 'Month', 'Year', 'Hours', 'Minutes', 'Seconds']
    data = pd.DataFrame(features_values, columns=names)
    data_scaled = scaler.transform(data)
    
    prediction = model.predict(data_scaled)
    print(prediction)
    text = "Estimated Traffic volume is: " + str(prediction[0])
    
    return render_template("index.html", prediction_text=text)

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(port=port, debug=True, use_reloader=False)
